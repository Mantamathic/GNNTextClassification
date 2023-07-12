from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from keras import models


from GNNTextClassification.dataPrep import dataPrepper
from GNNTextClassification.methods import methods
from GNNTextClassification.modifications import modifications
from GNNTextClassification.networks.convolutional import trainAndEvaluateConvolutional
from spookyAuthorClassifier.src.Utils import utils

# configure the desired experiment variables

# 'spookyAuthor' or 'movieReview' or 'small'
dataset = 'spookyAuthor'
# leave empty and/or 'punctuation' and/or 'lemmatize' and/or 'stopwords'
modification = list(['punctuation', 'lemmatize', 'stopwords'])
# 'BOW' or 'TF-IDF' or 'subWord' or 'charLevel' or 'embed'
method = 'BOW'
# 'naiveBayes' or 'NN' or 'convolutionalNN' or 'graphNN'
network = 'convolutionalNN'
# how many hidden layers to use for the NN
hiddenLayers = 20
# how many characters for the subWords should be used (min, max)
subWordCharSize = (2, 7)

start_time = time.time()

# prepare the dataset
textTrain, textTest, outputTrain, outputTest = dataPrepper.prepData(dataset)

# edit modifications
modifications.editModifications(modification)
if method == 'subWord':
    methods.editSubWordCharSize(subWordCharSize)

# applying the modifications and the method
transformedTrain, transformedTest = methods.apply(method, textTrain, textTest, network)

# instantiating model
if network == 'naiveBayes':
    model = MultinomialNB()
elif network == 'NN':
    model = MLPClassifier(hidden_layer_sizes=(hiddenLayers,), max_iter=1000)
elif network == 'convolutionalNN':
    model = models.Sequential()
elif network == 'graphNN':
    model = MultinomialNB()
else:
    raise ValueError("Unusable network. Use 'naiveBayes'', 'NN', 'convolutionalNN' or 'graphNN'")

# training and evaluating the model
if network == 'naiveBayes' or network == 'NN':
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
elif network == 'convolutionalNN':
    trainAndEvaluateConvolutional(model, transformedTrain, transformedTest, outputTrain, outputTest, hiddenLayers)

# Print the elapsed time
print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))
