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

# 'naiveBayes' or 'NN' or 'convolutionalNN' or 'graphNN'
network = 'convolutionalNN'
# 'spookyAuthor' or 'movieReview' or 'small'
dataset = 'spookyAuthor'
# 'BOW' or 'TF-IDF' or 'charLevel' or 'subWord' or 'embed'
method = 'BOW'
# 'punctuation' and/or 'lemmatize' and/or 'stopwords' and/or 'nouns' and/or 'verbs' and/or 'adjectives'
modification = list([])
# how many hidden layers to use for the NNs
hiddenLayers = 1
# how many characters for the subWords should be used (min, max)
subWordCharSize = (3, 3)

start_time = time.time()

# prepare the dataset
textTrain, textTest, outputTrain, outputTest = dataPrepper.prepData(dataset)

# edit modifications
modifications.editModifications(modification)
if method == 'subWord':
    methods.editSubWordCharSize(subWordCharSize)

# applying the modifications and the method
transformedTrain, transformedTest = methods.apply(method, textTrain, textTest, network, dataset)

# instantiating model, training and evaluating
if network == 'naiveBayes':
    model = MultinomialNB()
    trainAndEvaluateBasic(model, transformedTrain, outputTrain, transformedTest, outputTest)
elif network == 'NN':
    model = MLPClassifier(hidden_layer_sizes=(hiddenLayers,), max_iter=1500)
    trainAndEvaluateBasic(model, transformedTrain, outputTrain, transformedTest, outputTest)
elif network == 'convolutionalNN':
    model = models.Sequential()
    trainAndEvaluateConvolutional(model, transformedTrain, transformedTest, outputTrain, outputTest, method, hiddenLayers)
elif network == 'graphNN':
    if dataset == 'movieReview':
        trainAndEvaluateGraph(transformedTrain, transformedTest, outputTrain, outputTest, hiddenLayers, num_classes=2)
    else:
        trainAndEvaluateGraph(transformedTrain, transformedTest, outputTrain, outputTest, hiddenLayers, num_classes=3)
else:
    raise ValueError("Unusable network. Use 'naiveBayes'', 'NN', 'convolutionalNN' or 'graphNN'")

# Print the elapsed time
print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))
