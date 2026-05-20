import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from utils import plot_confusion_matrix


def train_knn_selected_features(dataset):

    subset = dataset[
        (dataset['labels'] == 0) |
        (dataset['labels'] == 1)
    ]

    X = subset.drop(columns=['labels'])
    y = subset['labels']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    feature_groups = [
        [156, 438, 566],
        [141, 300, 500],
        [90, 516, 460]
    ]

    accuracies = []

    for group in feature_groups:

        model = KNeighborsClassifier(n_neighbors=5)

        model.fit(X_train[group], y_train)

        predictions = model.predict(X_test[group])

        accuracy = metrics.accuracy_score(y_test, predictions)

        cm = confusion_matrix(y_test, predictions)

        accuracies.append(accuracy)

        plot_confusion_matrix(
            cm,
            f'KNN Confusion Matrix - Features {group}'
        )

    plt.plot(range(len(accuracies)), accuracies, marker='o')

    plt.title('KNN Accuracy by Feature Group')
    plt.xlabel('Feature Group')
    plt.ylabel('Accuracy')

    plt.show()


def train_knn_random_features(dataset):

    subset = dataset[
        (dataset['labels'] == 0) |
        (dataset['labels'] == 1)
    ]

    X = subset.drop(columns=['labels'])
    y = subset['labels']

    k_values = [3, 5, 7, 10]
    n_features = [3, 10, 50, 100]

    results_test = np.zeros((len(n_features), len(k_values)))

    for i, n in enumerate(n_features):

        sampled_columns = X.sample(
            n=n,
            axis=1,
            random_state=42
        )

        X_selected = sampled_columns.values

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected,
            y,
            test_size=0.2,
            random_state=42
        )

        for j, k in enumerate(k_values):

            model = KNeighborsClassifier(n_neighbors=k)

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = metrics.accuracy_score(
                y_test,
                predictions
            )

            results_test[i, j] = accuracy

    for i, n in enumerate(n_features):

        plt.plot(
            k_values,
            results_test[i],
            marker='o',
            label=f'n={n}'
        )

    plt.title('KNN Accuracy')
    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()