import matplotlib.pyplot as plt
import tensorflow as tf
import tkinter as tk
from utility import *
from createDiagrams import createGraphValTrainLoss, createGraphValTrainAcc
from tkinter.filedialog import askopenfilename, askdirectory
from keras import models
from modelBuilder import *
from subprocess import call
import os

# Commands
commands = ["train", "predict", "help", "quit", "photo", "continue training", "evaluate"]

# Evaluiert eine Bildeingabe
def processImage(image, modelToTest):
    model = models.load_model("models/saves/" + modelToTest)
    
    # Bereitet das Bild bzw. den entsprechenden Tensor auf das Modell vor
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = tf.expand_dims(input_arr, 0)

    pred = model.predict(input_arr)

    if modelToTest == "unspecific":
        # Stellt die Wahrscheinlichkeit eines Gesichts dar, gerundet auf zwei
        # Nachkommastellen
        probFace = round((100 - pred[0,0] * 100), ndigits=2)

        print("Probability face: " + str(probFace) + "%")
    else:
        probFace = 0
        biggest = 0

        for detected in pred[0]:
            if detected > biggest:
                biggest = detected
        
#       Fordert den Index der grössten Zahl
        for i in range(len(pred[0])):
            if pred[0,i] == biggest:
                probFace = i
        
#       Namen der Personen in einer Liste darstellen
        try:
            people = [x[1] for x in os.walk('./pictures/specific/TrainingSet')]
            npeople = len([x[0] for x in os.walk('./pictures/specific/TrainingSet')])
            people = people[0][:npeople]
            print(people)
        
    #   Dictionary aller Personennamen erstellen
            namesPeople = {}
            index = 0
            for names in people:
                namesPeople[index] = names
                index += 1
        
    #   Den Index im Dictionary abrufen
            print(namesPeople[probFace])
        
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.title(str(probFace))
            plt.axis("off")
            plt.show()
        
            return probFace
        
        except:
            print("Der notwendige Datensatz existiert nicht.")

# Testet das Modell
def testModel(modelToTest):
    # Oeffnet Fenster, um ein Bild auszuwählen
    filename = askopenfilename()
    tk.Tk().withdraw()
    print(filename)

    image = tf.keras.preprocessing.image.load_img(filename, target_size=(250, 250))
    probFace = processImage(image, modelToTest)

    return probFace

def evaluateModel(modelToTest, path):
    modelToLoad = modelToTest #+ ".h5"
    model = models.load_model("models/saves/" + modelToLoad)
    
    dataPath = askdirectory()
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
         dataPath,
         labels="inferred",
         image_size=(250, 250)
    )
    
    results = model.evaluate(dataset)
    
    name = path + "Evaluation_" + modelToTest + ".txt"
    
    file = open(name, "a")
    file.write("Test loss: %f; Test accuracy: %f \n" % (results[0], results[1]))
    file.close()
    
def takePhoto(filename):
    try:
        call(['libcamera-still -o %s.jpg --ev 0.5 --awbgains 1,1 --width 250 --height 350' % filename], shell=True)
        
        image = tf.keras.preprocessing.image.load_img(filename, target_size=(250, 250))

        return image
    except:
        print("An Error has occurred utilizing the camera. Make sure your computer supports libcamera/has a camera attached. ")