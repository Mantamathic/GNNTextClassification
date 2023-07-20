from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import time
from keras import models

from GNNTextClassification.dataPrep import dataPrepper
from GNNTextClassification.methods import methods
from GNNTextClassification.modifications import modifications
from GNNTextClassification.networks.convolutional import trainAndEvaluateConvolutional
from GNNTextClassification.networks.basic import trainAndEvaluateBasic
from GNNTextClassification.networks.graph import trainAndEvaluateGraph

# configure the desired experiment variables

# 'spookyAuthor' or 'movieReview' or 'small'
dataset = 'movieReview'
# leave empty and/or 'punctuation' and/or 'lemmatize' and/or 'stopwords'
modification = list(['punctuation', 'lemmatize', 'stopwords'])
# 'BOW' or 'TF-IDF' or 'subWord' or 'charLevel' or 'embed'
method = 'BOW'
# 'naiveBayes' or 'NN' or 'convolutionalNN' or 'graphNN'
network = 'naiveBayes'
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
    trainAndEvaluateBasic(model, transformedTrain, outputTrain, transformedTest, outputTest)
elif network == 'NN':
    model = MLPClassifier(hidden_layer_sizes=(hiddenLayers,), max_iter=1000)
    trainAndEvaluateBasic(model, transformedTrain, outputTrain, transformedTest, outputTest)
elif network == 'convolutionalNN':
    model = models.Sequential()
    trainAndEvaluateConvolutional(model, transformedTrain, transformedTest, outputTrain, outputTest, hiddenLayers)
elif network == 'graphNN':
    if dataset == 'movieReview':
        trainAndEvaluateGraph(transformedTrain, transformedTest, outputTrain, outputTest, hiddenLayers, num_classes=2)
    else:
        trainAndEvaluateGraph(transformedTrain, transformedTest, outputTrain, outputTest, hiddenLayers, num_classes=3)
else:
    raise ValueError("Unusable network. Use 'naiveBayes'', 'NN', 'convolutionalNN' or 'graphNN'")

# Print the elapsed time
print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))
