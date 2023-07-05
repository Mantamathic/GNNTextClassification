from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from GNNTextClassification.dataPrep import dataPrepper
from spookyAuthorClassifier.src.Utils import utils

# configure the desired experiment variables

# 'spookyAuthor' or 'movieReview' or 'small'
dataset = 'small'
# method =


# prepare the dataset
text_train, text_test, author_train, author_test = dataPrepper.prepData(dataset)


# defining the bag-of-words transformer on the text-processed corpus
bagOfWordsTransformer = CountVectorizer(analyzer=utils.text_process).fit(text_train)
# transforming into Bag-of-Words and hence textual data to numeric
bagOfWordsTrain = bagOfWordsTransformer.transform(text_train)  # ONLY TRAINING DATA
# transforming into Bag-of-Words and hence textual data to numeric
bagOfWordsTextTest = bagOfWordsTransformer.transform(text_test)  # TEST DATA

# instantiating the model with Multinomial Naive Bayes..
model = MultinomialNB()

# training the model...
model = model.fit(bagOfWordsTrain, author_train)

print("Training Accuracy: ", model.score(bagOfWordsTrain, author_train))

print("Test Accuracy: ", model.score(bagOfWordsTextTest, author_test))

# getting the predictions of the Validation Set
predictions = model.predict(bagOfWordsTextTest)
# getting the Precision, Recall, F1-Score
print(classification_report(author_test, predictions))

cm = confusion_matrix(author_test, predictions)
plt.figure()
utils.plot_confusion_matrix(cm, classes=[0, 1, 2], normalize=True, title='Confusion Matrix')
