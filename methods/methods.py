

def apply(method, transformer, textTrain, textTest):
    if method == 'BOW':
        return bagOfWords(transformer, textTrain, textTest)
    elif method == 'TF-IDF':
        return termFrequencyInverseDocumentFrequency()
    elif method == 'subWord':
        return subWordEncoding()
    elif method == 'charLevel':
        return characterLevelEncoding()
    elif method == 'embed':
        return wordEmbedding()
    else:
        raise ValueError("Unusable method. Use 'BOW', 'TF-IDF', 'subWord', 'charLevel' or 'embed'")


def bagOfWords(transformer, textTrain, textTest):
    # transforming into Bag-of-Words and hence textual data to numeric
    bagOfWordsTrain = transformer.transform(textTrain)
    # transforming into Bag-of-Words and hence textual data to numeric
    bagOfWordsTextTest = transformer.transform(textTest)
    return bagOfWordsTrain, bagOfWordsTextTest


def termFrequencyInverseDocumentFrequency():
    return "null", "null"


def subWordEncoding():
    return "null", "null"


def characterLevelEncoding():
    return "null", "null"


def wordEmbedding():
    return "null", "null"
