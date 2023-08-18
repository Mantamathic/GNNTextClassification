from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from nltk import pos_tag
import nltk
from nltk.tokenize import word_tokenize


modifications = []


def editModifications(newModifications):
    global modifications
    modifications = newModifications


def modify(text):
    if 'punctuation' in modifications:
        text = removePunctuation(text)
    if 'lemmatize' in modifications:
        text = lemmatize(text)
    if 'stopwords' in modifications:
        text = removeStopwords(text)
    if 'adjectives' in modifications:
        text = removeAdjectives(text)
    if 'verbs' in modifications:
        text = removeVerbs(text)
    if 'nouns' in modifications:
        text = removeNouns(text)
    return text


def removePunctuation(text):
    noPunctuation = [char for char in text if char not in string.punctuation]
    noPunctuation = ''.join(noPunctuation)
    return noPunctuation


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    lemmatizedText = ''
    for word in text.split():
        lemmatizedWord = lemmatizer.lemmatize(word, pos="v")
        lemmatizedText += lemmatizedWord + ' '
    return lemmatizedText


def removeStopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
    return ' '.join(filtered_words) if filtered_words else 'empty'


def removeAdjectives(text):
    # nltk.download('averaged_perceptron_tagger')
    tagged_words = pos_tag(text.split())
    filtered_words = [word for word, pos in tagged_words if pos != 'JJ']  # JJ represents adjectives in NLTK POS tagging
    # print(' '.join(filtered_words) if filtered_words else 'empty')
    return ' '.join(filtered_words) if filtered_words else 'empty'


def removeVerbs(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    filtered_words = [word for word, pos in pos_tags if pos not in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
    # print(' '.join(filtered_words))
    return ' '.join(filtered_words)


def removeNouns(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    filtered_words = [word for word, pos in pos_tags if pos not in ['NN', 'NNS', 'NNP', 'NNPS']]
    # print(' '.join(filtered_words))
    return ' '.join(filtered_words)