import PIL
from PIL import Image, ImageTk
import cv2
from tkinter import *
from tensorflow.keras import models
import tensorflow as tf
import numpy as np

import data
import plotting
import letters

# plots 25 samples from training dataset
# plotting.random_from_set(data.train_images, data.train_labels)
print(cv2.__version__)
model = models.load_model("Saved-Model")
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

width, height = 800, 600
cap = cv2.VideoCapture(0)

root = Tk()
root.geometry("800x600+10+20")
root.bind('<Escape>', lambda e: root.quit())
root.title("Rozpoznawanie języka migowego")
lmain = Label(root)
lmain.pack(side=LEFT)
label = Label(text="Zidentyfikowana litera:")
label.place(x=595, y=250)
label = Label(text="")
label.config(font=("Courier", 50))
label.place(x=600, y=300)


def identify_sign(cv2image):  # REPLACE identifySign with proper function
    prediction = probability_model.predict(
        np.reshape(cv2image, (1, 28, 28, 1)))
    index = np.argmax(prediction)
    return letters.names[index]


def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    image = image[:, :480]
    to_recognise = cv2.resize(image, (28, 28))

    global label
    label["text"] = identify_sign(to_recognise)

    image = cv2.resize(image, (500, 500),
                       interpolation=cv2.INTER_NEAREST)
    image = PIL.Image.fromarray(image)
    image = ImageTk.PhotoImage(image=image)
    lmain.image = image
    lmain.configure(image=image)
    lmain.after(10, show_frame)


def run_gui():
    show_frame()
    root.mainloop()


if __name__ == "__main__":
    run_gui()
