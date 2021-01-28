import random
import letters
import matplotlib.pyplot as plt


def random_from_set(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        n = random.randint(0, len(labels))
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[n], cmap='gray')
        plt.xlabel(letters.names[labels[n]])
    plt.show()
