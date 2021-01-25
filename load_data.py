import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def load_data(path):  # reads csv and returns it as square grayscale images with labels
    dataset = pd.read_csv(path)
    size = int(math.sqrt(dataset.shape[1]))
    images = np.reshape(dataset.values[:, 1:], (-1, size, size))
    labels = dataset.values[:, 0]

    return images, labels


def show_image(index, images, labels, label_names):  # plots image of index from dataset
    plt.imshow(images[index], cmap="gray")
    plt.axis("off")
    plt.title(label_names[labels[index]])
    plt.show()
    plt.clf()


letters = [
    'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

# convert csv into images
(train_images, train_labels) = load_data("MNIST-Dataset/sign_mnist_train.csv")
(test_images, test_labels) = load_data("MNIST-Dataset/sign_mnist_test.csv")

# show something to check correctness
show_image(11, train_images, train_labels, letters)
