from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from GNNTextClassification.dataPrep import dataPrepper
from GNNTextClassification.methods import methods
from GNNTextClassification.modifications import modifications
from spookyAuthorClassifier.src.Utils import utils

# configure the desired experiment variables

# 'spookyAuthor' or 'movieReview' or 'small'
dataset = 'small'
# leave empty and/or 'punctuation' and/or 'lemmatize' and/or 'stopwords'
modification = list(['punctuation', 'lemmatize', 'stopwords'])
# 'BOW' or 'TF-IDF' or 'subWord' or 'charLevel' or 'embed'
method = 'TF-IDF'


# prepare the dataset
textTrain, textTest, outputTrain, outputTest = dataPrepper.prepData(dataset)

# apply modifications
modifications.editModifications(modification)

# applying the method
transformedTrain, transformedTest = methods.apply(method, textTrain, textTest)


# instantiating the model with Multinomial Naive Bayes..
model = MultinomialNB()

# Scale the vectors to a non-negative range only needed for naive bayes
# TODO see if subwordEncoder matrix works for NNs
scaler = MinMaxScaler()
transformedTrain = scaler.fit_transform(transformedTrain)
transformedTest = scaler.transform(transformedTest)

# training the model...
model = model.fit(transformedTrain, outputTrain)

print("Training Accuracy: ", model.score(transformedTrain, outputTrain))

print("Test Accuracy: ", model.score(transformedTest, outputTest))

# getting the predictions of the Validation Set
predictions = model.predict(transformedTest)
# getting the Precision, Recall, F1-Score
print(classification_report(outputTest, predictions))

cm = confusion_matrix(outputTest, predictions)
plt.figure()
utils.plot_confusion_matrix(cm, classes=[0, 1, 2], normalize=True, title='Confusion Matrix')
