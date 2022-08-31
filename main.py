# Imports
from commands import *

# Wichtige Bildeigenschaften
image_size_train = (250, 250)
batch_size_train = 33

# Interaktionen
quit = False
while quit == False:
    action = input("Tippe help für Befehle oder gebe einen Befehl ein: ")

    if action in commands:
        if action == "train":
            epochs = retrieveEpochs()
            history = trainNewModel(retrieveModel(), epochs, image_size_train, batch_size_train)

        if action == "predict":
            pred = testModel(retrieveModel())

        if action == "help":
            print("train: Trainiere ein Modell\npredict: Evaluiere ein Bild\nphoto: Mache ein Bild mit der PiCamera und evaluiere es (setzt PiCamera voraus)\ncontinue training: helpSetze das Training eines Modells fort\nquit: Verlasse das Programm\n")

        if action == "quit":
            quit = True    

        if action == "photo":
            try:
                pred = processImage(takePhoto(), retrieveModel())
            except:
                print("There has been an error utilizing the PiCamera")

        if action == "continue training":
            continueTraining(image_size_train, batch_size_train)
    else:
        print("Kein gültiger Command\n")

