import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from utils import plot_confusion_matrix


def train_decision_tree(dataset):

    X = dataset.drop(columns=['labels'])
    y = dataset['labels']

    X_dev, X_eval, y_dev, y_eval = train_test_split(
        X,
        y,
        random_state=1,
        test_size=0.1
    )

    depths = [1, 2, 3, 5, 10]

    kf = KFold(n_splits=5)

    results = np.zeros((5, len(depths)))

    for i, (train_idx, test_idx) in enumerate(kf.split(X_dev)):

        X_train = X_dev.iloc[train_idx]
        X_test = X_dev.iloc[test_idx]

        y_train = y_dev.iloc[train_idx]
        y_test = y_dev.iloc[test_idx]

        for j, depth in enumerate(depths):

            model = DecisionTreeClassifier(
                max_depth=depth
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            cm = confusion_matrix(
                y_test,
                predictions
            )

            accuracy = np.trace(cm) / np.sum(cm)

            results[i, j] = accuracy

    mean_accuracy = results.mean(axis=0)

    plt.plot(
        depths,
        mean_accuracy,
        marker='o'
    )

    plt.title('Decision Tree Accuracy')
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')

    plt.show()

    final_model = DecisionTreeClassifier(
        max_depth=10,
        criterion='entropy'
    )

    final_model.fit(X_dev, y_dev)

    final_predictions = final_model.predict(X_eval)

    final_accuracy = accuracy_score(
        y_eval,
        final_predictions
    )

    final_cm = confusion_matrix(
        y_eval,
        final_predictions
    )

    print(f'Final Accuracy: {final_accuracy:.4f}')

    plot_confusion_matrix(
        final_cm,
        'Final Decision Tree Confusion Matrix'
    )