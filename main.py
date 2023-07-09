from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from GNNTextClassification.dataPrep import dataPrepper
from GNNTextClassification.methods import methods
from GNNTextClassification.modifications import modifications
from spookyAuthorClassifier.src.Utils import utils

# configure the desired experiment variables

# 'spookyAuthor' or 'movieReview' or 'small'
dataset = 'spookyAuthor'
# leave empty and/or 'punctuation' and/or 'lemmatize' and/or 'stopwords'
modification = list(['punctuation', 'lemmatize', 'stopwords'])
# 'BOW' or 'TF-IDF' or 'subWord' or 'charLevel' or 'embed'
method = 'subWord'
# how many characters for the subWords should be used (min, max)
subWordCharSize = (2, 7)


# prepare the dataset
textTrain, textTest, outputTrain, outputTest = dataPrepper.prepData(dataset)

# edit modifications
modifications.editModifications(modification)
if method == 'subWord':
    methods.editSubWordCharSize(subWordCharSize)

# applying the modifications and the method
transformedTrain, transformedTest = methods.apply(method, textTrain, textTest)


# instantiating the model with Multinomial Naive Bayes..
model = MultinomialNB()

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
