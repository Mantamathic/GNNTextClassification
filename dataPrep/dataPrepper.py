from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas
import matplotlib.pyplot as plt


def prepData(dataSetString):
    if dataSetString == 'spookyAuthor':
        dataSet = pandas.read_csv('../GNNTextClassification/data/spookyAuthor/spookyAuthor.csv')
        inputAxis = dataSet['text']
        outputAxis = dataSet['author']
    elif dataSetString == 'movieReview':
        dataSet = pandas.read_csv('../GNNTextClassification/data/movieReview/movieReview.csv')
        inputAxis = dataSet['text']
        outputAxis = dataSet['tag']
    elif dataSetString == 'small':
        dataSet = pandas.read_csv('../GNNTextClassification/data/spookyAuthor/small.csv')
        inputAxis = dataSet['text']
        outputAxis = dataSet['author']
    else:
        raise ValueError("Unusable dataset. Use 'spookyAuthor', 'movieReview' or 'small'.")

    # takes the output classes (authors) and gives them values 0,1,2
    labelEncoder = LabelEncoder()
    outputAxis = labelEncoder.fit_transform(outputAxis)
    # show a sample of the dataset
    print(dataSet.sample(5))
    # 80-20 splitting the dataset (80%->Training and 20%->Validation), random number for reproducible outcomes
    return train_test_split(inputAxis, outputAxis, test_size=0.2, random_state=1234)


def generateWordClouds(textAxis):
    wordCloudEAP = WordCloud().generate(textAxis[0])  # for EAP
    wordCloudHPL = WordCloud().generate(textAxis[1])  # for HPL
    wordCloudMWS = WordCloud().generate(textAxis[3])  # for MWS

    print(textAxis[0])
    # print(df['author'][0])
    plt.imshow(wordCloudEAP, interpolation='bilinear')
    plt.show()

    print(textAxis[1])
    # print(df['author'][1])
    plt.imshow(wordCloudHPL, interpolation='bilinear')
    plt.show()

    print(textAxis[3])
    # print(df['author'][3])
    plt.imshow(wordCloudMWS, interpolation='bilinear')
    plt.show()
