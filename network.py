import tensorflow as tf
from tensorflow.keras import models

import numpy as np
import matplotlib.pyplot as plt

import preprocessing

# preprocess data
train_images = preprocessing.train_images
train_labels = preprocessing.train_labels
# test_images = preprocessing.test_images / 255.0

letters = preprocessing.letters

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap='gray')
    plt.xlabel(letters[train_labels[i]])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(preprocessing.img_size, preprocessing.img_size, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(letters))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)

models.save_model(model,"Saved-Model-1")

# predictions = probability_model.predict(test_images)
