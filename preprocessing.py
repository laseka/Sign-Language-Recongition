import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

letters = [
    'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]


def load_data(path):  # reads csv and returns it as square grayscale images with labels
    dataset = pd.read_csv(path)
    images = dataset.drop(labels='label', axis=1)
    images = np.reshape(images.values, (-1, 28, 28, 1))
    labels = dataset['label'].values
    labels = np.reshape(labels, (-1, 1))
    return images, labels


def show_image(index, images, labels, label_names):  # plots image of index from dataset
    plt.imshow(images[index], cmap="gray")
    plt.axis("off")
    plt.title(label_names[labels[index][0]])
    plt.show()
    plt.clf()
