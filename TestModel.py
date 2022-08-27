import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import models
import tensorflow as tf
import tkinter as tk 
from tkinter.filedialog import askopenfilename

def TestModel():
    modelToLoad = "FaceNoFace.h5"

    model = models.load_model(modelToLoad)

    tk.Tk().withdraw()
    filename = askopenfilename()
    print(filename)

    image = tf.keras.preprocessing.image.load_img(filename, target_size=(250, 250))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = tf.expand_dims(input_arr, 0)
    pred = model.predict(input_arr)

    probFace = 100 - pred[0,0] * 100

    print("Probability face: " + str(probFace) + "%")

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(str(probFace) + "%")
    plt.axis("off")
    plt.show()