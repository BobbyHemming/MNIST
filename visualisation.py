import matplotlib.pyplot as plt
import numpy as np


def plot_digit_prediction(data, label, prediction):
    string = 'No'
    if prediction:
        string = "Yes"
    string = f'Does SGD Classifier think the image instance is {label}? {string}'
    plt.text(0, 0, string)
    plt.imshow(data.reshape(28, 28), cmap='Greys', interpolation="nearest")
    plt.axis("off")


def plot_multiple_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap="Greys", **options)
    plt.axis('off')




