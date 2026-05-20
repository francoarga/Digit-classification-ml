from data_preprocessing import load_dataset
from data_preprocessing import crop_images

from exploratory_analysis import plot_digit_image
from exploratory_analysis import plot_statistics

from knn_models import train_knn_selected_features
from knn_models import train_knn_random_features

from decision_tree_models import train_decision_tree


DATA_PATH = '../data/TMNIST_Data.csv'


dataset = load_dataset(DATA_PATH)

plot_digit_image(dataset)

cropped_dataset = crop_images(dataset)

plot_statistics(
    cropped_dataset.iloc[:, :-1],
    size=24
)

train_knn_selected_features(cropped_dataset)

train_knn_random_features(cropped_dataset)

train_decision_tree(cropped_dataset)