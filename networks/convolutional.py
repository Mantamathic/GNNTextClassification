import numpy as np
import itertools
import tensorflow as tf
from keras import layers

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


def trainAndEvaluateConvolutional(model, transformedTrain, transformedTest, outputTrain, outputTest, hiddenLayers):

    transformedTrain = transformedTrain.toarray()
    transformedTest = transformedTest.toarray()

    max_sequence_length = 1155
    padding_value = 0

    padded_train = tf.keras.preprocessing.sequence.pad_sequences(transformedTrain, maxlen=max_sequence_length, padding='post', value=padding_value)
    padded_test = tf.keras.preprocessing.sequence.pad_sequences(transformedTest, maxlen=max_sequence_length, padding='post', value=padding_value)

    padded_train = np.expand_dims(padded_train, axis=2)
    padded_test = np.expand_dims(padded_test, axis=2)

    model.add(layers.Conv1D(10, 3, activation='relu', input_shape=(max_sequence_length, 1)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(10, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(10, 3, activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(padded_train, outputTrain, epochs=5,
                        validation_data=(padded_test, outputTest))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(padded_test, outputTest, verbose=2)

    print(test_acc)

    # Train the model on the reshaped input
    model.fit(padded_train, outputTrain)

    print("Training Accuracy: ", model.evaluate(padded_train, outputTrain)[1])
    print("Test Accuracy: ", model.evaluate(padded_test, outputTest)[1])

    # Getting the predictions of the Validation Set
    predictions = model.predict(padded_test)
    # Getting the Precision, Recall, F1-Score
    print(classification_report(outputTest, np.argmax(predictions, axis=1)))

    cm = confusion_matrix(outputTest, np.argmax(predictions, axis=1))
    plt.figure()
    plotConfusionMatrix(cm, classes=[0, 1, 2], normalize=True, title='Confusion Matrix')


def plotConfusionMatrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
