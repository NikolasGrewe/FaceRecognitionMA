import tensorflow as tf
import os
from keras import models

# Datensätze erzeugen aus Ordnern
def create_dss(directory, labels, image_size, batch_size):
    trainDs = tf.keras.preprocessing.image_dataset_from_directory(
         directory,
         labels=labels,
         validation_split=0.2,
         subset="training",
         seed=8967,
         image_size=image_size,
         batch_size=batch_size
    )
     
    valDs = tf.keras.preprocessing.image_dataset_from_directory(
         directory,
         labels=labels,
         validation_split=0.2,
         subset="validation",
         seed=8967,
         image_size=image_size,
         batch_size=batch_size
    )
    
    return trainDs, valDs

# Speichern des besten Checkpoints als Hauptdatei
def saveCheckpoints(model, modelToSave):
    checkpoints = ["./models/callbacks/" + model + "/" + name for name in os.listdir("./models/callbacks/" + model)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        modelToSave = models.load_model(latest_checkpoint)
        modelToSave.save('models/saves/%s' % model)
    else:
        modelToSave.save('models/saves/%s' % model)
        
# Fragt das aufzurufende Modell ab
def retrieveModel():
    print("Welches Modell soll getestet/trainiert werden? unspecific oder specific?")

    valid = False

    while(valid != True):
        modelToTrain = str(input()).lower()

        if modelToTrain in ["specific", "unspecific", "s", "u"]:
            valid = True
            
            if modelToTrain == "s":
                modelToTrain = "specific"
            elif modelToTrain == "u":
                modelToTrain = "unspecific"

            return modelToTrain
        else:
            print("Kein gültiges Modell, nochmal versuchen")
            
# Fragt die Epochs ab
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
            
    