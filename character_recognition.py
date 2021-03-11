from tkinter import Canvas, Tk, Button, ROUND, TRUE, RAISED, SUNKEN
from PIL import ImageGrab, Image
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('model_handwritten.h5')
blank = np.asarray(Image.open('blank.jpeg'))

def make_prediction(image):
    pred = model.predict(image)
    ans = []
    for _ in np.arange(0,3):
        ch = np.argmax(pred)
        pr = pred[0, ch]
        pred[0, ch] = -1
        ch = chr(65+ch)
        ans.append([ch, pr])
    return ans

class Paint(object):

    def __init__(self):
        self.root = Tk()
        self.root.geometry("+400+100")
        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=5)

        self.c = Canvas(self.root, bg='white', width=420, height=420)
        self.c.grid(row=1, columnspan=8)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 40
        self.color = 'black'
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.line_width = 40
        self.activate_button(self.pen_button)

    def use_eraser(self):
        self.line_width = 80
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

        # getting image
        image = ImageGrab.grab((410, 160, 410+420, 160+420))
        # resize image to 28x28 pixel
        pxl_size = 28, 28
        np_image = np.array(image.resize(pxl_size, Image.ANTIALIAS))
        # converting image from rgb to gray
        r, g, b = np_image[:, :, 0], np_image[:, :, 1], np_image[:, :, 2]
        img = 255 - (0.2989*r + 0.5870*g + 0.1140*b)
        # saving image
        Image.fromarray(img).convert('RGB').save("character.jpeg")
        # preparing image for CNN prediction
        X = img.reshape(1, 28, 28, 1)
        # prediction and output
        pred = make_prediction(X)
        if pred[0][1]>0.5 and (img>1.0255).sum()>50:
            print("Character is recognized!")
            for p in pred:
                character = p[0]
                probability = p[1]
                print("Written character is " + character + " with probability of", probability)
        else:
            print("Character is not recognized! Try again.")
        print("..........................................................................")
        

if __name__ == '__main__':
    Paint()