from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from gensim import corpora
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes):
        super().__init__()
        torch.manual_seed(11)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        return x


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def trainAndEvaluateGraph(transformedTrain, transformedTest, outputTrain, outputTest, hiddenLayers, num_classes):
    # Convert the sparse matrix to a dense matrix
    transformedTrain_dense = transformedTrain.toarray()

    # Convert the dense matrix to a list of documents
    documents = [[str(word) for word in document] for document in transformedTrain_dense]

    # Create a dictionary
    dictionary = corpora.Dictionary(documents)

    # Create an empty graph
    graph = nx.Graph()

    # Add nodes to the graph using the word indices from the dictionary
    for words in documents:
        indices = [dictionary.token2id[word] for word in words]
        graph.add_nodes_from(indices)

    # Add edges to the graph based on the window size
    window_size = 5
    for words in documents:
        indices = [dictionary.token2id[word] for word in words]
        for i, word in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    graph.add_edge(word, indices[j])

    # Now you have the graph object to use in your further processing

    # Example usage: get the adjacency matrix as a NumPy array
    adj_matrix = nx.to_numpy_array(graph)

    num_features = transformedTrain.shape[1]
    model = GCN(hidden_channels=hiddenLayers, num_features=num_features, num_classes=num_classes)
    model.eval()

    x = torch.tensor(transformedTrain.toarray(), dtype=torch.float)
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
    y = torch.tensor(outputTrain, dtype=torch.long)

    out = model(x, edge_index)

    visualize(out, color=y)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    criterion = torch.nn.CrossEntropyLoss()

    mask_train = torch.tensor(outputTrain, dtype=torch.long)
    mask_test = torch.tensor(outputTest, dtype=torch.long)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[mask_train], y[mask_train])
        loss.backward()
        optimizer.step()
        return loss

    def test():
        model.eval()
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[mask_test] == y[mask_test]
        test_acc = int(test_correct.sum()) / len(outputTest)
        return test_acc

    for epoch in range(1, 101):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

    model.eval()
    out_test = model(torch.tensor(transformedTest.toarray(), dtype=torch.float), edge_index)
    visualize(out_test, color=torch.tensor(outputTest, dtype=torch.long))

    # Getting the predictions of the Validation Set
    out_test = out_test.argmax(dim=1)
    # Getting the Precision, Recall, F1-Score
    print(classification_report(outputTest, out_test))
    cm = confusion_matrix(outputTest, out_test)
    plt.figure()
    plotConfusionMatrix(cm, classes=[0, 1, 2], normalize=True, title='Confusion Matrix')


def plotConfusionMatrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):

    num_classes = len(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(num_classes), range(num_classes)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

