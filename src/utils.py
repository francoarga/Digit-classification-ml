import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, title):

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(title)
    plt.xlabel('Prediction')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()