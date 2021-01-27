import cv2
from tensorflow.keras import models
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter

letters = [
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'nothing',
    'K', 'L', 'M', 'N', 'O',
    'P', 'R', 'S', 'T', 'Q',
    'U', 'V', 'W', 'X', 'Y',
]

model = models.load_model("Saved-Model-1")
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
img_size = 50
camera = cv2.VideoCapture(0)

while True:
    captured, image = camera.read()
    if captured:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image = image[:,:,1]

        # image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)

        # image = Image.fromarray(image)
        # image = image.convert("L")
        # image = image.filter(ImageFilter.FIND_EDGES)
        # image = np.asarray(image)
        image = cv2.resize(image, (img_size, img_size))

        prediction = probability_model.predict(np.reshape(image, (1, img_size, img_size, 3)))
        index = np.argmax(prediction)
        print(letters[index])

        image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("image", image)


camera.release()
cv2.destroyAllWindows()
