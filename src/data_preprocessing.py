import pandas as pd
import numpy as np


def load_dataset(path):
    return pd.read_csv(path)


def crop_images(dataset):
    def crop_image(row):
        img = np.array(row[2:]).reshape((28, 28))
        return img[2:-2, 2:-2].flatten()

    cropped = dataset.apply(crop_image, axis=1)
    cropped = pd.DataFrame(cropped.tolist())
    cropped['labels'] = dataset['labels']

    return cropped