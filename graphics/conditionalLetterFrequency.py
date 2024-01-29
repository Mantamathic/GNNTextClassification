import pandas as pd
from collections import Counter

# Read the dataset
dataset_path = '../../GNNTextClassification/data/spookyAuthor/spookyAuthor.csv'
dataSet = pd.read_csv(dataset_path)

# Define classes
classes = dataSet['author'].unique()

# Create a list of characters from a-z
alphabet = list("abcdefghijklmnopqrstuvwxyz")

# Initialize counters for each class
eap_letter_count = Counter({char: 0 for char in alphabet})
hpl_letter_count = Counter({char: 0 for char in alphabet})
mws_letter_count = Counter({char: 0 for char in alphabet})
total_letters = {'eap': 0, 'hpl': 0, 'mws': 0}

# Function to update letter counts for a given class
def update_letter_count(text, author):
    letter_count = Counter(text.lower())
    if author == 'EAP':
        eap_letter_count.update(letter_count)
        total_letters['eap'] += len(text)
    elif author == 'HPL':
        hpl_letter_count.update(letter_count)
        total_letters['hpl'] += len(text)
    elif author == 'MWS':
        mws_letter_count.update(letter_count)
        total_letters['mws'] += len(text)

# Process each row in the dataset
for index, row in dataSet.iterrows():
    update_letter_count(row['text'], row['author'])

# Calculate conditional frequencies for 'pos' class
eap_letter_freq = {char: count / total_letters['eap'] for char, count in eap_letter_count.items()}

# Calculate conditional frequencies for 'neg' class
hpl_letter_freq = {char: count / total_letters['hpl'] for char, count in hpl_letter_count.items()}

# Calculate conditional frequencies for 'neg' class
mws_letter_freq = {char: count / total_letters['mws'] for char, count in mws_letter_count.items()}

# Display the results
print("Conditional Letter Frequencies for 'eap' class:")
print(eap_letter_freq)

print("\nConditional Letter Frequencies hpl 'hpl' class:")
print(hpl_letter_freq)

print("\nConditional Letter Frequencies for 'mws' class:")
print(mws_letter_freq)
