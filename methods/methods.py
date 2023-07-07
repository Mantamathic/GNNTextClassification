from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from GNNTextClassification.modifications import modifications
from gensim.models import FastText
import numpy as np



def apply(method, textTrain, textTest):
    if method == 'BOW':
        return bagOfWords(textTrain, textTest)
    elif method == 'TF-IDF':
        return termFrequencyInverseDocumentFrequency(textTrain, textTest)
    elif method == 'subWord':
        return subWordEncoding(textTrain, textTest)
    elif method == 'charLevel':
        return characterLevelEncoding()
    elif method == 'embed':
        return wordEmbedding()
    else:
        raise ValueError("Unusable method. Use 'BOW', 'TF-IDF', 'subWord', 'charLevel' or 'embed'")


def bagOfWords(textTrain, textTest):
    # Create a count vectorizer to convert the text data into a matrix of word counts
    bagOfWordsTransformer = CountVectorizer(analyzer=modifications.modify).fit(textTrain)
    # transforming into Bag-of-Words and hence textual data to numeric
    bagOfWordsTrain = bagOfWordsTransformer.transform(textTrain)
    # transforming into Bag-of-Words and hence textual data to numeric
    bagOfWordsTextTest = bagOfWordsTransformer.transform(textTest)
    return bagOfWordsTrain, bagOfWordsTextTest


def termFrequencyInverseDocumentFrequency(textTrain, textTest):
    # Create a TF-IDF vectorizer to calculate the TF-IDF scores for each word
    vectorizer = TfidfVectorizer(analyzer=modifications.modify)
    # Fit the vectorizer
    vectorizer.fit(textTrain)
    # Transform the training and test text into TF-IDF representations
    tfIdfTrain = vectorizer.transform(textTrain)
    tfIdfTest = vectorizer.transform(textTest)
    return tfIdfTrain, tfIdfTest


def subWordEncoding(textTrain, textTest, vector_size=100, window=5, min_count=5):
    # Tokenize the training and test text
    tokenizedTrain = [modifications.modify(text) for text in textTrain]
    tokenizedTest = [modifications.modify(text) for text in textTest]

    model = FastText(sentences=tokenizedTrain, vector_size=vector_size, window=window, min_count=min_count)

    # Vectorize the training text
    subwordVecTrain = []
    for words in tokenizedTrain:
        subwords = [model.wv[word] for word in words if word in model.wv]
        if subwords:
            subwordVecTrain.append(np.mean(subwords, axis=0))
        else:
            subwordVecTrain.append(np.zeros(vector_size))

    subwordVecTrain = np.vstack(subwordVecTrain)

    # Vectorize the test text
    subwordVecTest = []
    for words in tokenizedTest:
        subwords = [model.wv[word] for word in words if word in model.wv]
        if subwords:
            subwordVecTest.append(np.mean(subwords, axis=0))
        else:
            subwordVecTest.append(np.zeros(vector_size))

    subwordVecTest = np.vstack(subwordVecTest)

    return subwordVecTrain, subwordVecTest


def characterLevelEncoding():
    return "null", "null"


def wordEmbedding():
    return "null", "null"
