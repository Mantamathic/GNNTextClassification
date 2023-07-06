from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from GNNTextClassification.modifications import modifications


def apply(method, textTrain, textTest):
    if method == 'BOW':
        return bagOfWords(textTrain, textTest)
    elif method == 'TF-IDF':
        return termFrequencyInverseDocumentFrequency(textTrain, textTest)
    elif method == 'subWord':
        return subWordEncoding()
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
    vectorizer = CountVectorizer(analyzer=modifications.modify)
    trainCounts = vectorizer.fit_transform(textTrain)
    testCounts = vectorizer.transform(textTest)

    # Create a TF-IDF transformer to calculate the IDF scores for each word
    tfidf_transformer = TfidfTransformer()
    tfIdfTrain = tfidf_transformer.fit_transform(trainCounts)
    tfIdfTest = tfidf_transformer.transform(testCounts)
    return tfIdfTrain, tfIdfTest


def subWordEncoding():
    return "null", "null"


def characterLevelEncoding():
    return "null", "null"


def wordEmbedding():
    return "null", "null"
