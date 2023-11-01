from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from GNNTextClassification.modifications import modifications
import numpy as np
from gensim.models import Word2Vec
import pandas

subWordCharSize = ()


def editSubWordCharSize(newSubWordCharSize):
    global subWordCharSize
    subWordCharSize = newSubWordCharSize


def apply(method, textTrain, textTest, network, dataset):
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
        return wordEmbedding(textTrain, textTest, network, dataset)
    else:
        raise ValueError("Unusable method. Use 'BOW', 'TF-IDF', 'subWord', 'charLevel' or 'embed'")


def bagOfWords(textTrain, textTest):
    # Create a count vectorizer to convert the text data into a matrix of word counts
    bagOfWordsTransformer = CountVectorizer().fit(textTrain)
    # transforming into Bag-of-Words and hence textual data to numeric
    bagOfWordsTrain = bagOfWordsTransformer.transform(textTrain)
    bagOfWordsTextTest = bagOfWordsTransformer.transform(textTest)
    return bagOfWordsTrain, bagOfWordsTextTest


def termFrequencyInverseDocumentFrequency(textTrain, textTest):
    # Create a TF-IDF vectorizer to calculate the TF-IDF scores for each word
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer
    vectorizer.fit(textTrain)
    # Transform the training and test text into TF-IDF representations (use *10 against dying ReLu)
    tfIdfTrain = vectorizer.transform(textTrain)
    tfIdfTest = vectorizer.transform(textTest)
    return tfIdfTrain, tfIdfTest


def subWordEncoding(textTrain, textTest):
    # Create a subWord vectorizer to convert the text data into a matrix of character counts
    subWordVectorizer = CountVectorizer(analyzer='char_wb', ngram_range=subWordCharSize).fit(textTrain)
    # Transforming into subWord encoding and hence textual data to numeric
    subWordEncodedTrain = subWordVectorizer.transform(textTrain)
    subWordEncodedTest = subWordVectorizer.transform(textTest)
    return subWordEncodedTrain, subWordEncodedTest


def characterLevelEncoding(textTrain, textTest):
    # Create a character-level vectorizer to convert the text data into a matrix of character counts
    charVectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1)).fit(textTrain)
    # Transforming into character-level encoding and hence textual data to numeric
    charEncodedTrain = charVectorizer.transform(textTrain)
    charEncodedTest = charVectorizer.transform(textTest)
    return charEncodedTrain, charEncodedTest


def wordEmbedding(textTrain, textTest, network, dataSet):
    # for the CNN, this value also needs to be adjusted
    vectorSize = 15

    if dataSet == 'movieReview':
        dataSet = pandas.read_csv('../GNNTextClassification/data/movieReview/movieReviewFull.csv')
        inputAxis = dataSet['text'].tolist()
        # Combine textTrain and inputAxis to create a unified corpus for training Word2Vec
        combined_text = inputAxis
    else:
        dataSet = pandas.read_csv('../GNNTextClassification/data/spookyAuthor/additional.csv')
        inputAxis = dataSet['text'].tolist()
        # Combine textTrain and inputAxis to create a unified corpus for training Word2Vec
        combined_text = textTrain + inputAxis

    tokenized_corpus = [text.split() for text in combined_text]

    # Train the Word2Vec model on the tokenized_corpus
    model = Word2Vec(sentences=tokenized_corpus, vector_size=vectorSize, window=14, min_count=1, sg=1)

    # Get the word embeddings for each word in the textTrain and textTest data
    def get_sentence_embedding(text):
        embeddings = [model.wv[word] if word in model.wv else np.zeros(vectorSize) for word in text.split()]
        return embeddings

    # Create sequences of word embeddings for training and testing data
    sequencesTrain = [get_sentence_embedding(text) for text in textTrain]
    sequencesTest = [get_sentence_embedding(text) for text in textTest]

    # Convert the sequences to numpy arrays
    sequencesTrain = np.array(sequencesTrain, dtype=object)
    sequencesTest = np.array(sequencesTest, dtype=object)

    # Truncate to positive values
    if network == "naiveBayes":
        sequencesTrain = np.clip(sequencesTrain, 0, None)
        sequencesTest = np.clip(sequencesTest, 0, None)
    return sequencesTrain, sequencesTest

