import matplotlib.pyplot as plt
import numpy as np


def plot_digit_image(dataset):
    pixels = dataset.iloc[:, 2:]

    img = np.array(pixels.iloc[1]).reshape((28, 28))

    plt.imshow(img, cmap='gray')
    plt.title('Example Digit')
    plt.show()


def plot_statistics(images, size=28):

    mean_pixels = images.mean()
    std_pixels = images.std()
    var_pixels = images.var()

    plt.imshow(np.array(mean_pixels).reshape((size, size)), cmap='gray')
    plt.title('Mean Pixel Intensity')
    plt.show()

    plt.imshow(np.array(std_pixels).reshape((size, size)), cmap='gray')
    plt.title('Pixel Standard Deviation')
    plt.show()

    plt.imshow(np.array(var_pixels).reshape((size, size)), cmap='gray')
    plt.title('Pixel Variance')
    plt.show()