import data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator

cnn = Sequential()

cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(25))

cnn.summary()

cnn.compile(optimizer='adam',
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

data_gen = ImageDataGenerator(rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest'
                              )

data_gen.fit(data.train_images)

history = cnn.fit(data_gen.flow(data.train_images, data.train_labels, batch_size=32),
                  steps_per_epoch=len(data.train_images) / 32,
                  epochs=25
                  )

test_loss, test_acc = cnn.evaluate(data.test_images,  data.test_labels, verbose=2)
print("test accuracy = " + str(test_acc))

cnn.save("Saved-Model")
