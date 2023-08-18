import numpy as np
import itertools
import tensorflow as tf
from keras import layers

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas


# 'spookyAuthor' or 'movieReview' or 'small'
dataset = 'spookyAuthor'

# prepare the dataset
if dataset == 'spookyAuthor':
    dataSet = pandas.read_csv('../../GNNTextClassification/data/spookyAuthor/spookyAuthor.csv')
    inputAxis = dataSet['text']
    outputAxis = dataSet['author']
elif dataset == 'movieReview':
    dataSet = pandas.read_csv('../../GNNTextClassification/data/movieReview/movieReview.csv')
    inputAxis = dataSet['text']
    outputAxis = dataSet['tag']
elif dataset == 'small':
    dataSet = pandas.read_csv('../../GNNTextClassification/data/spookyAuthor/small.csv')
    inputAxis = dataSet['text']
    outputAxis = dataSet['author']
else:
    raise ValueError("Unusable dataset. Use 'spookyAuthor', 'movieReview' or 'small'.")

# takes the output classes (authors) and gives them values 0,1,2
labelEncoder = LabelEncoder()
outputAxis = labelEncoder.fit_transform(outputAxis)
# show a sample of the dataset
print(dataSet.sample(5))
# 80-20 splitting the dataset (80%->Training and 20%->Validation), random number for reproducible outcomes
textTrain, textTest, outputTrain, outputTest = train_test_split(inputAxis, outputAxis, test_size=0.2, random_state=1234)

# Instantiate the model
model = tf.keras.Sequential()

# Convert text to sequences of characters
tokenizer = tf.keras.layers.TextVectorization()
tokenizer.adapt(np.concatenate([textTrain, textTest]).astype(str))
transformedTrain = tokenizer(textTrain)
transformedTest = tokenizer(textTest)

# Model architecture
model.add(layers.Embedding(input_dim=len(tokenizer.get_vocabulary()), output_dim=50, mask_zero=True))
model.add(layers.Conv1D(64, 15, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(64, activation='relu'))

if dataset == 'movieReview':
    model.add(layers.Dense(2))
else:
    model.add(layers.Dense(3))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(transformedTrain, outputTrain, epochs=5,
                    validation_data=(transformedTest, outputTest))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(transformedTest, outputTest, verbose=2)

print(test_acc)

# Train the model on the reshaped input
model.fit(transformedTrain, outputTrain)

print("Training Accuracy: ", model.evaluate(transformedTrain, outputTrain)[1])
print("Test Accuracy: ", model.evaluate(transformedTest, outputTest)[1])

# Getting the predictions of the Validation Set
predictions = model.predict(transformedTest)
# Getting the Precision, Recall, F1-Score
print(classification_report(outputTest, np.argmax(predictions, axis=1)))

cm = confusion_matrix(outputTest, np.argmax(predictions, axis=1))
plt.figure()

print('Confusion matrix, without normalization')

print(cm)
classes = [0, 1, 2]
title = 'Confusion Matrix'
cmap = plt.cm.Blues

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()





