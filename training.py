import os
import modelBuilder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
from utility import *
from createDiagrams import *

# Erzeugt ein neues Modell und trainiert es
def trainNewModel(model, epochs, shape, batch_size):

    if model == "unspecific":
        # Basis-Trainingsdatensätze erzeugen
        trainDsUnspec, valDsUnspec = create_dss("./pictures/unspecific/TrainingSet/", 'inferred', shape, batch_size)

        trainDsUnspec = trainDsUnspec.prefetch(buffer_size=32)
        valDsUnspec = valDsUnspec.prefetch(buffer_size=32)

    else:
        trainDsSpec, valDsSpec = create_dss("./pictures/specific/TrainingSet2/", 'inferred', shape, batch_size)
        
        trainDsSpec = trainDsSpec.prefetch(buffer_size=32)

        people = [x[0] for x in os.walk('./pictures/specific/TrainingSet2')]
        npeople = len(people) - 1

# Training model unspecific
    if model == "unspecific":
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath="./models/callbacks/unspecific/model_{epoch}",
                save_best_only=True,
                monitor="val_loss",
                verbose=1
            ),
            keras.callbacks.CSVLogger("./Tabels/UnspecificCSVs/unspecific.csv")
        ]
        # Erstes Modell erstellen
        model1 = modelBuilder.create_model_basic(input_shape=shape + (3,))

        model1.compile(
            optimizer=keras.optimizers.RMSprop(0.5e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[metrics.BinaryAccuracy(), metrics.Precision()]
        )

        # Trainieren
        print("Training model1")
        history = model1.fit(trainDsUnspec, epochs=epochs, validation_data=valDsUnspec, callbacks=callbacks)

        # Modell speichern
        saveCheckpoints("unspecific", model1)

        # Visualisierung der Modelle erstellen
        tf.keras.utils.plot_model(model1, show_shapes=True, to_file="models/diagrams/model1.png")
        model1.summary()
        createGraphValTrainLoss(history, epochs, "unspecificLosses.png")
        createGraphValTrainAcc(history, epochs, "unspecificAccs.png")
    
    else:
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath="./models/callbacks/specific/model_{epoch}",
                save_best_only=True,
                monitor="val_loss",
                verbose=1
            ), 
            keras.callbacks.CSVLogger("./Tabels/SpecificCSVs/specific.csv")
        ]
        # Zweites Modell erstellen
        model2 = modelBuilder.create_specific_model(input_shape=shape + (3,), outs=npeople)

        model2.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.5e-3), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['acc', 'categorical_accuracy']
        )
        
        # Trainieren
        print("Training model2")
        history = model2.fit(trainDsSpec, epochs=epochs, validation_data=valDsSpec, callbacks=callbacks)

        # Modell speichern
        saveCheckpoints("specific", model2)

        # Visualisierung der Modelle erstellen
        tf.keras.utils.plot_model(model2, show_shapes=True, to_file="models/diagrams/model2.png")
        model2.summary()
        createGraphValTrainLoss(history, epochs, "specificLosses.png")
        createGraphValTrainAcc(history, epochs, "specificAccs.png")

    return history

# Setzt mit dem Training eines Modells fort
def continueTraining(shape, batch_size):
    modelToLoad = retrieveModel()
    dataPath = askdirectory()
    epochs = retrieveEpochs()

    #Erzeugt neue Datensätze basierend auf den Daten im angegebenen Ordner
    model = models.load_model("models/saves/" + modelToLoad)
    trainData, valData = create_dss(dataPath, "inferred", shape, batch_size)

    if modelToLoad == "unspecific":
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath="./models/callbacks/unspecific/model_{epoch}",
                save_best_only=True,
                monitor="val_loss",
                verbose=1
            )
        ]
        
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['acc']
        )
    else:
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath="./models/callbacks/specific/model_{epoch}",
                save_best_only=True,
                monitor="val_loss",
                verbose=1
            )
        ]
        
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(1e-3), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['acc']
        )

    # Trainiert das Modell weiter: kein Datenverlust
    history = model.fit(trainData, epochs=epochs, validation_data=valData, callbacks=callbacks)
    
    createGraphValTrainAcc(history, epochs, "Continued")

    saveCheckpoints("specific", model)