from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import torch_geometric.transforms as T
from torch_geometric.data import Data


def trainAndEvaluateGraph(transformedTrain, outputTrain, hiddenLayers, num_classes, method):
    if method == 'embed':
        documents = []
        for document_embeddings in transformedTrain:
            doc_embedding = np.mean(document_embeddings, axis=0)
            documents.append(doc_embedding)

        transformedTrainSim = np.array(documents)

        similarity_matrix = cosine_similarity(transformedTrainSim)

        num_documents = len(transformedTrain)
        num_words = transformedTrainSim.shape[1]

        doc_embeddings = [np.sum(embeddings, axis=0) for embeddings in transformedTrain]
        x = torch.tensor(np.array(doc_embeddings), dtype=torch.float)
        x = x.view(num_documents, -1)
    else:
        num_documents, num_words = transformedTrain.shape
        similarity_matrix = cosine_similarity(transformedTrain)
        x = torch.tensor(transformedTrain.toarray(), dtype=torch.float)

    y_train = torch.tensor(outputTrain, dtype=torch.long)

    # Create an empty graph
    graph = nx.Graph()

    for i in range(num_documents):
        label = outputTrain[i]
        graph.add_node(label)

    threshold = 0.99
    # Build the edge_index using consecutive integer indices
    edges_to_add = []
    for i in range(num_documents):
        for j in range(i + 1, num_documents):
            if similarity_matrix[i, j] > threshold:
                edges_to_add.append((i, j))

    graph.add_edges_from(edges_to_add)

    # Visualization
    # plt.figure(figsize=(9, 7))
    # y = torch.tensor(outputTrain, dtype=torch.long)
    # nx.draw_spring(graph, node_size=30, arrows=False, node_color=y)
    # plt.show()

    # Convert networkx graph to PyTorch Geometric Data object
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=y_train)

    # Apply RandomNodeSplit transform to the Data object
    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    data = split(data)

    # Print the resulting Data object (optional)
    print(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcn = GCN(num_features=num_words, num_classes=num_classes, hiddenLayers=hiddenLayers).to(device)
    optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    gcn = train_node_classifier(gcn, data, optimizer_gcn, criterion, n_epochs=1000)

    test_acc = eval_node_classifier(gcn, data, data.test_mask)
    print(f'Test Acc: {test_acc:.3f}')

    # Getting the predictions for the test set
    with torch.no_grad():
        gcn.eval()
        out_test = gcn(data)[data.test_mask].argmax(dim=1)
        y_true = data.y[data.test_mask]

    # Getting the Precision, Recall, F1-Score
    print(classification_report(y_true, out_test))
    cm = confusion_matrix(y_true, out_test)

    plotConfusionMatrix(cm, classes=np.unique(y_true), normalize=False, title='Confusion Matrix')


def train_node_classifier(model, graph, optimizer, criterion, n_epochs=100):

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = eval_node_classifier(model, graph, graph.val_mask)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

    return model


def eval_node_classifier(model, graph, mask):

    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    return acc


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hiddenLayers):
        super().__init__()
        self.conv1 = GCNConv(num_features, hiddenLayers)
        self.conv2 = GCNConv(hiddenLayers, 100)
        self.conv3 = GCNConv(100, 27)
        self.conv4 = GCNConv(27, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.tanh(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        output = self.conv4(x, edge_index)

        return output


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