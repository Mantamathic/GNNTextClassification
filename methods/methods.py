from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Reshape

from GNNTextClassification.modifications import modifications
import numpy as np
from gensim.models import Word2Vec

subWordCharSize = ()


def editSubWordCharSize(newSubWordCharSize):
    global subWordCharSize
    subWordCharSize = newSubWordCharSize


def apply(method, textTrain, textTest, network):

    # apply the text modifications first
    textTrain = [modifications.modify(text) for text in textTrain]
    textTest = [modifications.modify(text) for text in textTest]

    # apply the methods
    if method == 'BOW':
        return bagOfWords(textTrain, textTest)
    elif method == 'TF-IDF':
        return termFrequencyInverseDocumentFrequency(textTrain, textTest)
    elif method == 'subWord':
        return subWordEncoding(textTrain, textTest)
    elif method == 'charLevel':
        return characterLevelEncoding(textTrain, textTest)
    elif method == 'embed':
        return wordEmbedding(textTrain, textTest, network)
    else:
        raise ValueError("Unusable method. Use 'BOW', 'TF-IDF', 'subWord', 'charLevel' or 'embed'")


def bagOfWords(textTrain, textTest):
    # Create a count vectorizer to convert the text data into a matrix of word counts
    bagOfWordsTransformer = CountVectorizer().fit(textTrain)
    # transforming into Bag-of-Words and hence textual data to numeric
    bagOfWordsTrain = bagOfWordsTransformer.transform(textTrain)
    # transforming into Bag-of-Words and hence textual data to numeric
    bagOfWordsTextTest = bagOfWordsTransformer.transform(textTest)
    return bagOfWordsTrain, bagOfWordsTextTest


def termFrequencyInverseDocumentFrequency(textTrain, textTest):
    # Create a TF-IDF vectorizer to calculate the TF-IDF scores for each word
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer
    vectorizer.fit(textTrain)
    # Transform the training and test text into TF-IDF representations
    tfIdfTrain = vectorizer.transform(textTrain)
    tfIdfTest = vectorizer.transform(textTest)
    return tfIdfTrain, tfIdfTest


def subWordEncoding(textTrain, textTest):
    # Create a subWord vectorizer to convert the text data into a matrix of character counts
    subWordVectorizer = CountVectorizer(analyzer='char_wb', ngram_range=subWordCharSize).fit(textTrain)
    # Transforming into subWord encoding and hence textual data to numeric
    subWordEncodedTrain = subWordVectorizer.transform(textTrain)
    # Transforming into subWord encoding and hence textual data to numeric
    subWordEncodedTest = subWordVectorizer.transform(textTest)
    return subWordEncodedTrain, subWordEncodedTest


def characterLevelEncoding(textTrain, textTest):
    # Create a character-level vectorizer to convert the text data into a matrix of character counts
    charVectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1)).fit(textTrain)
    # Transforming into character-level encoding and hence textual data to numeric
    charEncodedTrain = charVectorizer.transform(textTrain)
    # Transforming into character-level encoding and hence textual data to numeric
    charEncodedTest = charVectorizer.transform(textTest)
    return charEncodedTrain, charEncodedTest


def wordEmbedding(textTrain, textTest, network):

    # Train the Word2Vec model
    model = Word2Vec(sentences=textTrain, seed=1234)

    # Transform the text data to word embeddings
    embeddingTrain = []
    embeddingTest = []

    for words in textTrain:
        embeddings = [model.wv[word] for word in words if word in model.wv]
        if embeddings:
            embeddingTrain.append(np.mean(embeddings, axis=0))

    for words in textTest:
        embeddings = [model.wv[word] for word in words if word in model.wv]
        if embeddings:
            embeddingTest.append(np.mean(embeddings, axis=0))

    # Convert the embedding lists to arrays
    embeddingTrain = np.array(embeddingTrain)
    embeddingTest = np.array(embeddingTest)

    # Truncate negative values to 2 if naive Bayes
    if network == 'naiveBayes':
        embeddingTrain[embeddingTrain < 0] = 2
        embeddingTest[embeddingTest < 0] = 2

    return embeddingTrain, embeddingTest

