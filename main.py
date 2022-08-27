# Imports
from cgi import test
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import models
from commands import *

# Commands
commands = ["train", "predict", "help", "quit", "photo"]

# Wichtige Bildeigenschaften
image_size_train = (250, 250)
batch_size_train = 33

# Datensätze erzeugen aus Ordnern
def create_dss(directory, labels, image_size, batch_size):
    trainDs = tf.keras.preprocessing.image_dataset_from_directory(
         directory,
         labels=labels,
         validation_split=0.2,
         subset="training",
         seed=8967,
         image_size=image_size,
         batch_size=batch_size)
     
    valDs = tf.keras.preprocessing.image_dataset_from_directory(
         directory,
         labels=labels,
         validation_split=0.2,
         subset="validation",
         seed=8967,
         image_size=image_size_train,
         batch_size=batch_size_train)
    return trainDs, valDs

def trainModel(model, epochs):
    # Basis-Trainingsdatensätze erzeugen
    trainDsOs, valDsOs = create_dss("pictures/unspecific/TrainingSet", 'inferred', image_size_train, batch_size_train)

    # Neun Bilder Der Trainingsdatensätze darstellen
    plt.figure(figsize=(10, 10))
    for images, labels in trainDsOs.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")

    if model == "unspecific":
        # Erstes Modell erstellen
        model1 = models.create_model_basic(input_shape=image_size_train + (3,))

        model1.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=['acc']
        )

        # Trainieren und Testen
        print("Training model1")
        history = model1.fit(trainDsOs, epochs=epochs, validation_data=valDsOs)

        # Modell speichern
        model1.save('models/saves/unspecific.h5')

        # Visualisierung der Modelle erstellen
        keras.utils.plot_model(model1, show_shapes=True, to_file="models/diagrams/model1.png")
        model1.summary()
    
    else:
        # Zweites Modell erstellen
        model2 = models.create_specific_model(input_shape=image_size_train, outputs=3)

        model2.compile(
            optimizer=keras.optimizers.Adam(1e-3), 
            loss=keras.losses.BinaryCrossentropy(),
            metrics=['acc'])

        # Trainieren und Testen
        print("Training model2")
        history = model2.fit(trainDsOs, epochs=epochs, validation_data=valDsOs)

        # Modell speichern
        model2.save('models/saves/specific.h5')

        # Visualisierung der Modelle erstellen
        keras.utils.plot_model(model2, show_shapes=True, to_file="models/diagrams/model2.png")
        model2.summary()

    return history

quit = False
while quit == False:
    action = input("Tippe help für Befehle oder gebe einen Befehl ein: ")

    if action in commands:
        if action == "train":
            epochs = trainModelCMD()
            history = trainModel(retrieveModel(), epochs)

        if action == "predict":
            pred = testModel(retrieveModel())

        if action == "help":
            print("train: Trainiere ein Modell\npredict: Evaluiere ein Bild\nphoto: Mache ein Bild mit der PiCamera und evaluiere es (setzt PiCamera voraus)\nquit: Verlasse das Programm\n")

        if action == "quit":
            quit = True    

        if action == "photo":
            try:
                pred = processImage(takePhoto(), retrieveModel())
            except:
                print("There has been an error utilizing the PiCamera")
    else:
        print("Kein gültiger Command\n")

