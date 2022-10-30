# Imports
from commands import *
from utility import *
from training import *

# Wichtige Bildeigenschaften
image_size_train = (250, 250)
batch_size_train = 33

# Interaktionen
quit = True
while quit:
    action = input("Tippe help für Befehle oder gebe einen Befehl ein: ")
    action = action.lower()

    if action in commands:
        if action == "train":
            epochs = retrieveEpochs()
            history = trainNewModel(retrieveModel(), epochs, image_size_train, batch_size_train)

        if action == "predict":
            pred = testModel(retrieveModel())

        if action == "help":
            print("train: Trainiere ein Modell\npredict: Evaluiere ein Bild\nphoto: Mache ein Bild mit der PiCamera und evaluiere es (setzt PiCamera voraus)\ncontinue training: Setze das Training eines Modells fort\nevaluate: Evaluiere ein Netzwerk\nquit: Verlasse das Programm\n")

        if action == "quit":
            quit = False    

        if action == "photo":
            try:
                pred = processImage(takePhoto('photo'), retrieveModel())
            except:
                print("There has been an error utilizing the PiCamera")

        if action == "continue training":
            continueTraining(image_size_train, batch_size_train)
            
        if action == "evaluate":
            evaluateModel(retrieveModel(), "./Tabels/")
    else:
        print("Kein gültiger Command\n")
