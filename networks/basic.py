from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np


def trainAndEvaluateBasic(model, transformedTrain, outputTrain, transformedTest, outputTest):
    model = model.fit(transformedTrain, outputTrain)

    print("Training Accuracy: ", model.score(transformedTrain, outputTrain))
    print("Test Accuracy: ", model.score(transformedTest, outputTest))

    # getting the predictions of the Validation Set
    predictions = model.predict(transformedTest)
    # getting the Precision, Recall, F1-Score
    print(classification_report(outputTest, predictions))

    cm = confusion_matrix(outputTest, predictions)
    plt.figure()
    plot_confusion_matrix(cm, classes=[0, 1, 2], normalize=True)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
