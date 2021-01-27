import numpy as np
from PIL import Image, ImageFilter
import cv2
import random
from matplotlib import pyplot as plt

letters = [
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'nothing',
    'K', 'L', 'M', 'N', 'O',
    'P', 'R', 'S', 'T', 'Q',
    'U', 'V', 'W', 'X', 'Y',
]

max_letters_num = 3000
letter_repeats = 3000
img_size = 50

train_images = np.ndarray(shape=(letter_repeats * len(letters), img_size, img_size, 3), dtype=float)
train_labels = np.ndarray(shape=(letter_repeats * len(letters),), dtype=int)

for n in range(len(letters)):
    print(letters[n])
    for i in range(letter_repeats):
        image = Image.open("ASL/train/%s/%s%d.jpg" % (letters[n], letters[n], random.randint(1, max_letters_num)))
        index = i + n * letter_repeats
        # print(image.shape)

        # processed_image = image.convert("L")

        image = np.asarray(image)
        image = cv2.resize(image, (img_size, img_size))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # processed_image = Image
        # processed_image = Image.fromarray(processed_image)
        # processed_image = processed_image.filter(ImageFilter.FIND_EDGES)
        # processed_image = np.asarray(processed_image)


        # image = np.expand_dims(image, axis=2)
        train_images[index] = image / 255.0
        train_labels[index] = n

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i * letter_repeats], cmap='gray')
    plt.xlabel(letters[train_labels[i * letter_repeats]])
plt.show()