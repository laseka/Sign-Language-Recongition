from tensorflow.keras import models, layers
import tensorflow as tf


def create_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(25))

    return model


def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


def train_model(model, train_images, train_labels, test_images, test_labels, epochs):
    return model.fit(train_images, train_labels,
                        epochs=epochs,
                        validation_data=(test_images, test_labels))
