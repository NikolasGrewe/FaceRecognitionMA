# -*- coding: utf-8 -*-

# Imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Commands (Für Interaktionen; WIP)
commands = ["train", "predict", "load", "help", "exit"]

# Wichtige Bildeigenschaften
image_size_train = (250, 250)
batch_size_train = 33

# Behältnis für alle wichtigen Funktionen
class functions():
    
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
    
    # Netzwerk 1 (Sequential model) erstellen (TODO: optimieren)
    def create_model_basic(input_shape):
        model = models.Sequential()
        
        model.add(layers.Rescaling(1.0 / 255, input_shape=input_shape))
        model.add(layers.Conv2D(32, (9,9), activation='relu'))
        model.add(layers.Conv2D(32, (9,9), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.MaxPooling2D((2,2)))

        model.add(layers.Conv2D(64, (9,9), activation='relu'))
        model.add(layers.Conv2D(64, (9,9), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.MaxPooling2D((2,2)))

        model.add(layers.Conv2D(64, (9,9), activation='relu'))
        model.add(layers.Conv2D(64, (9,9), activation='relu'))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        
        return model
    
    # Netzwerk 2 (Functional Model) erstellen (TODO: optimieren)
    def create_specific_model(input_shape, outputs):
        inputs = keras.Input(shape=image_size_train + (3,))
        
        x = layers.Conv2D(32, (9,9), activation='relu')(inputs)
        output1 = layers.Conv2D(64, (9,9), activation='relu')(x)
        
        x = layers.Conv2D(64, (9,9), activation='relu', padding='same')(output1)
        x = layers.Conv2D(64, (9,9), activation='relu', padding='same')(x)
        output2 = layers.add([x, output1])
        
        x = layers.Flatten()(output2)
        x = layers.Dense(64, activation='relu')(x)
        outputFinal = layers.Dense(outputs, activation='softmax')(x)
        
        model = keras.Model(inputs, outputFinal)
        return model


# Basis-Trainingsdatensätze erzeugen
trainDsOs, valDsOs = functions.create_dss("pictures/unspecific/TrainingSet", 
                                          'inferred', 
                                          image_size_train, 
                                          batch_size_train)

TestDs = tf.keras.preprocessing.image_dataset_from_directory(
            "pictures/unspecific/TestSets/UnsortedNetTest",
            labels=None, 
            image_size=image_size_train)
    
# Erstes Modell erstellen
model1 = functions.create_model_basic(input_shape=image_size_train + (3,))

model1.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)]
)

# Zweites Modell erstellen
model2 = functions.create_specific_model(input_shape=image_size_train, outputs=3)

model2.compile(
    optimizer=keras.optimizers.RMSprop(), 
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.Accuracy()])

''' Wird benutzt für Interaktionen (WIP)
action = input("Command: ")

if action in commands:
    if action == "train":
       
'''

# Trainieren und Testen
print("Training model1")
model1.fit(trainDsOs, epochs=1)

print("Evaluation model1")
model1.evaluate(valDsOs)

print("Saving model1")
model1.save('saves/BasicFaceRecognition')

print("Prediction model1")
pred = model1.predict(TestDs)

# Darstellung der Ergebnisse von Modell 1
pred = np.argmax(pred, axis=1)[:]

plt.figure(figsize=(10, 10))
for images in TestDs.take(1):
    for i in range(3):
        ax = plt.subplot(1, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(pred[i]))
        plt.axis("off")

print(pred)
    
# Visualisierung der Modelle erstellen
keras.utils.plot_model(model1, show_shapes=True, to_file="model1.png")
model1.summary()

keras.utils.plot_model(model2, show_shapes=True, to_file="model2.png")
model2.summary()
             