import PIL
from PIL import Image, ImageTk
import pytesseract
import cv2
from tkinter import *

width, height = 800, 600
cap = cv2.VideoCapture(0)

root = Tk()
root.geometry("800x600+10+20")
root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)
lmain.pack(side=LEFT)

label = Label(text="")
label.place(x=600, y=300)


def identifySign(cv2image):   # REPLACE identifySign with proper function
    return "CAN'T RECOGNIZE"


def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2image = cv2.resize(cv2image, (28, 28), interpolation=cv2.INTER_CUBIC)

    # SENDING DATA TO AI
    global label
    label["text"] = identifySign(cv2image)

    cv2image = cv2.resize(cv2image, (500, 500),
                          interpolation=cv2.INTER_NEAREST)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


def runGui():
    show_frame()
    root.mainloop()


if __name__ == "__main__":
    runGui()
