import matplotlib.pyplot as plt
from keras import models
import tensorflow as tf
import tkinter as tk 
import modelBuilder
from tkinter.filedialog import askopenfilename, askdirectory
from piCameraFunction import *

# Commands
commands = ["train", "predict", "help", "quit", "photo", "continue training"]

# Datensätze erzeugen aus Ordnern
def create_dss(directory, labels, image_size, batch_size):
    trainDs = tf.keras.preprocessing.image_dataset_from_directory(
         directory,
         labels=labels,
         validation_split=0.2,
         subset="training",
         image_size=image_size,
         batch_size=batch_size)
     
    valDs = tf.keras.preprocessing.image_dataset_from_directory(
         directory,
         labels=labels,
         validation_split=0.2,
         subset="validation",
         image_size=image_size,
         batch_size=batch_size)
    return trainDs, valDs

def retrieveEpochs():
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

def continueTraining(shape, batch_size):
    modelToLoad = retrieveModel()
    dataPath = askdirectory()
    epochs = retrieveEpochs()

    #Erzeugt neue Datensätze basierend auf den Daten im angegebenen Ordner
    model = models.load_model("models/saves/" + modelToLoad + ".h5")
    trainData, valData = create_dss(dataPath, 'inferred', shape, batch_size)

    if modelToLoad == "unspecific":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['acc']
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['acc']
        )

    # Trainiert das Modell weiter: kein Datenverlust
    model.fit(trainData, epochs=epochs, validation_data=valData)

    #Speichert und kennzeichnet das alte Modell; Das neue Modell wird gespeichert
    os.rename("models/saves/" + modelToLoad + ".h5", "models/saves/" + modelToLoad + "old.h5")
    model.save("models/saves/" + modelToLoad + ".h5")

def retrieveModel():
    print("Welches Modell soll getestet/trainiert werden? unspecific oder specific?")

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
        biggest = 0
# TODO: Sortiermechanismus einbauen: Sortieren nach Ähnlichkeit
        for detected in pred[0]:
            if detected > biggest:
                biggest = detected
        
        for i in range(len(pred[0])):
            if pred[0,i] == biggest:
                probFace = i
        
        people = [x[0] for x in os.walk('./pictures/specific/lfw')]

        namesPeople = {}
        index = 0
        for names in people:
            namesPeople[index] = names
            index += 1

        print(namesPeople[probFace])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(str(probFace))
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

# Neun Bilder Der Trainingsdatensätze darstellen
def showPicturesFromDataset(dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
            
    plt.show()

def trainNewModel(model, epochs, shape, batch_size):
    people = 0

    if model == "unspecific":
        # Basis-Trainingsdatensätze erzeugen
        trainDsUnspec, valDsUnspec = create_dss("pictures/unspecific/TrainingSet", 'inferred', shape, batch_size)

        showPicturesFromDataset(trainDsUnspec)
    else:
        trainDsSpec, valDsSpec = create_dss("pictures/specific/lfw/", "inferred", shape, batch_size)

        people = [x[0] for x in os.walk('./pictures/specific/lfw')]
        npeople = len(people) - 1

        showPicturesFromDataset(trainDsSpec)

    if model == "unspecific":
        # Erstes Modell erstellen
        model1 = modelBuilder.create_model_basic(input_shape=shape + (3,))

        model1.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['acc']
        )

        # Trainieren und Testen
        print("Training model1")
        history = model1.fit(trainDsUnspec, epochs=epochs, validation_data=valDsUnspec)

        # Modell speichern
        model1.save('models/saves/unspecific.h5')

        # Visualisierung der Modelle erstellen
        tf.keras.utils.plot_model(model1, show_shapes=True, to_file="models/diagrams/model1.png")
        model1.summary()
    
    else:
        # Zweites Modell erstellen
        model2 = modelBuilder.create_specific_model(input_shape=shape + (3,), outs=npeople)

        model2.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['acc']
        )

        # Trainieren und Testen
        print("Training model2")
        history = model2.fit(trainDsSpec, epochs=epochs, validation_data=valDsSpec)

        # Modell speichern
        model2.save('models/saves/specific.h5')

        # Visualisierung der Modelle erstellen
        tf.keras.utils.plot_model(model2, show_shapes=True, to_file="models/diagrams/model2.png")
        model2.summary()

    return history
