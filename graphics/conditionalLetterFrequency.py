import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

dataSet = pd.read_csv('../../GNNTextClassification/data/spookyAuthor/spookyAuthor.csv')
labelEncoder = LabelEncoder()
dataSet['author_encoded'] = labelEncoder.fit_transform(dataSet['author'])

def calculate_conditional_letter_frequencies(data, class_labels):
    letter_counts = {label: np.zeros(26) for label in class_labels}
    vectorizer = CountVectorizer(analyzer='char')

    for label in class_labels:
        class_texts = data[data['author'] == label]['text']
        char_counts = vectorizer.fit_transform(class_texts)
        total_char_counts = np.sum(char_counts, axis=0).A1
        total_char_counts = total_char_counts.astype(float)
        total_char_counts /= total_char_counts.sum()
        total_char_counts /= total_char_counts.sum()
        letter_counts[label] = total_char_counts
    return letter_counts

class_labels = dataSet['author'].unique()
conditional_letter_frequencies = calculate_conditional_letter_frequencies(dataSet, class_labels)

for label in class_labels:
    print(f"\nConditional Letter Frequencies for class '{label}':")
    letters = list('abcdefghijklmnopqrstuvwxyz')
    frequencies = conditional_letter_frequencies[label]
    for letter, frequency in zip(letters, frequencies):
        print(f"{letter}: {frequency:.4f}")

