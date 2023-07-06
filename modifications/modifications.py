from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

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
    return [word for word in text.split()]


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
    return ' '.join([word for word in text.split() if word.lower() not in stopwords.words('english')])
