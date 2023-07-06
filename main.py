from sklearn.feature_extraction.text import CountVectorizer
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
modifications.editModifications(modification)
# 'BOW' or 'TF-IDF' or 'subWord' or 'charLevel' or 'embed'
method = 'BOW'


# prepare the dataset
textTrain, textTest, authorTrain, authorTest = dataPrepper.prepData(dataset)

# apply modifications
transformer = CountVectorizer(analyzer=modifications.modify).fit(textTrain)

# applying the method
transformedTrain, transformedTest = methods.apply(method, transformer, textTrain, textTest)


# instantiating the model with Multinomial Naive Bayes..
model = MultinomialNB()

# training the model...
model = model.fit(transformedTrain, authorTrain)

print("Training Accuracy: ", model.score(transformedTrain, authorTrain))

print("Test Accuracy: ", model.score(transformedTest, authorTest))

# getting the predictions of the Validation Set
predictions = model.predict(transformedTest)
# getting the Precision, Recall, F1-Score
print(classification_report(authorTest, predictions))

cm = confusion_matrix(authorTest, predictions)
plt.figure()
utils.plot_confusion_matrix(cm, classes=[0, 1, 2], normalize=True, title='Confusion Matrix')
