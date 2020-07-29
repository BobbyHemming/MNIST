from tkinter import *
from tkinter import filedialog

from PIL import ImageTk, Image, ImageOps
from imprep import ImageTransformer
from modelprep import load_model
import matplotlib.pyplot as plt


root = Tk()

# Set Title as Image Loader
root.title("Handwritten Digit Classifier")
root.geometry("220x275")
root.resizable(width=False, height=False)


def open_img():

    x = openfilename()
    im = Image.open(x)  # Open image file
    im1 = im
    img = im.resize((28, 28), Image.ANTIALIAS)

    im_transformer = ImageTransformer(threshold=90)
    data = im_transformer.transform(img)

    deeper_analysis_needed = False
    if deeper_analysis_needed:   # To understand where the classifier might be going wrong/ changes in image preparation
        print((data))
        plt.figure()
        plt.imshow(data.reshape((28, 28)), cmap='Greys', interpolation="nearest")
        plt.show()

    kn_clf = load_model('models/KNeighborsClassifier')
    y_pred = kn_clf.predict([data])

    img = ImageTk.PhotoImage(im1.resize((200, 200), Image.ANTIALIAS))
    panel = Label(root, image=img)
    panel2 = Label(root, text=f'Predicted hand-written digit: {y_pred}')
    panel.image = img
    panel.grid(row=2)
    panel.place(x=10, y=10)
    panel2.grid(row=3)
    panel2.place(x=10, y=210)


def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    root.update()
    filename = filedialog.askopenfilename(title='Digit image')

    return filename


# Create a button and place it into the window using grid layout
btn = Button(root, text='open image', command=open_img)
btn.grid(row=4, columnspan=4)
btn.place(x=55, y=240)

root.mainloop()





