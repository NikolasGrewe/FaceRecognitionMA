import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import models
import tensorflow as tf
import tkinter as tk 
from tkinter.filedialog import askopenfilename
from piCameraFunction import *

def trainModelCMD():
    print("Wie viele Epochs? ")

    valid = False

    while(valid != True):
        try:
            epochs = int(input())
        except:
            print("No valid integer, try again")
        else:
            valid = True
            print("Training with " + str(epochs) + " epochs")
            return epochs

def retrieveModel():
    print("Welches Modell soll trainiert werden? unspecific oder specific?")

    valid = False

    while(valid != True):
        modelToTrain = str(input())

        if modelToTrain in ["specific", "unspecific", "s", "u"]:
            valid = True
            
            if modelToTrain == "s":
                modelToTrain = "specific"
            elif modelToTrain == "u":
                modelToTrain = "unspecific"

            return modelToTrain
        else:
            print("Kein gültiges Modell, nochmal versuchen")

def processImage(image, modelToTest):
    modelToLoad = modelToTest + ".h5"
    model = models.load_model("models/saves/" + modelToLoad)
    
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = tf.expand_dims(input_arr, 0)
    pred = model.predict(input_arr)

    if modelToTest == "unspecific":
        probFace = round((100 - pred[0,0] * 100), ndigits=2)

        print("Probability face: " + str(probFace) + "%")
    else:
        probFace = 0

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(str(probFace) + "%")
    plt.axis("off")
    plt.show()

    return probFace

def testModel(modelToTest):

    tk.Tk().withdraw()
    filename = askopenfilename()
    print(filename)

    image = tf.keras.preprocessing.image.load_img(filename, target_size=(250, 250))
    probFace = processImage(image, modelToTest)

    return probFace

def takePhoto(filename):
    takePicture(filename)
    
    image = tf.keras.preprocessing.image.load_img(filename, target_size=(250, 250))

    return image
