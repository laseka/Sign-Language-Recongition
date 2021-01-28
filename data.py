import pandas as pd
import numpy as np


def load_data(name):
    dataset = pd.read_csv("Dataset/sign_mnist_%s.csv" % name)
    images = np.reshape(dataset.values[:, 1:], (-1, 28, 28, 1))
    labels = dataset['label'].values
    return images, labels


train_images, train_labels = load_data("train")
test_images, test_labels = load_data("test")
