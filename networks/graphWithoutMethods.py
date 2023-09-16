import numpy as np
import tensorflow as tf
import torch
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data

# Load your dataset
# Assuming your CSV file has columns: 'id', 'text', 'author'
dataset = pd.read_csv('../GNNTextClassification/data/spookyAuthor/spookyAuthor.csv')

# Preprocess the input data
inputAxis = dataset['text']
outputAxis = dataset['author']

# Convert text to sequences of characters
tokenizer = tf.keras.layers.TextVectorization()
tokenizer.adapt(inputAxis.astype(str))
transformedData = tokenizer(inputAxis)

# Convert the transformed data to PyTorch tensors
transformedData = torch.tensor(transformedData.numpy(), dtype=torch.float32)

# Create a graph representation of the data
edge_index = torch.tensor(list(itertools.combinations(range(len(inputAxis)), 2))).t().contiguous()

# Instantiate the GNN model
class GNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Convert the PyTorch Geometric Data object to the appropriate format
input_dim = len(tokenizer.get_vocabulary())
hidden_dim = 2
output_dim = len(outputAxis.unique())  # Number of unique classes
data = Data(x=transformedData, edge_index=edge_index)

# Instantiate the GNN model
gnn_model = GNNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

# Encode the labels using LabelEncoder
labelEncoder = LabelEncoder()
outputLabels = labelEncoder.fit_transform(outputAxis)

# Train the GNN model
def train_gnn(model, data, labels, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()

# Train the GNN model on your data
train_gnn(gnn_model, data, torch.tensor(outputLabels), optimizer, num_epochs=5)

# Evaluate the GNN model
gnn_model.eval()
with torch.no_grad():
    output = gnn_model(data)
    predictions = output.argmax(dim=1)

# Calculate metrics and visualization
print(classification_report(outputLabels, predictions))

cm = confusion_matrix(outputLabels, predictions)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# Add visualization settings

# Show the confusion matrix
plt.show()
