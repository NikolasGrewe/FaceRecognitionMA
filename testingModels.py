# Imports
import keras
import tensorflow as tf
from keras import layers

# Model 1: Unspezifische Gesichtserkennung: Kleines XCeption Model
def create_model_basic(input_shape):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = layers.RandomRotation(factor=0.3)(inputs)
    x = layers.RandomFlip(mode="horizontal")(x)
    
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    '''
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    '''
    previous_block_activation = x

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        
        x = layers.add([x, residual])
        previous_block_activation = x 

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

def create_dss(directory, labels, image_size, batch_size):
    trainDs = tf.keras.preprocessing.image_dataset_from_directory(
         directory,
         labels=labels,
         validation_split=0.2,
         subset="training",
         seed=2004,
         image_size=image_size,
         batch_size=batch_size
    )
     
    valDs = tf.keras.preprocessing.image_dataset_from_directory(
         directory,
         labels=labels,
         validation_split=0.2,
         subset="validation",
         seed=2004,
         image_size=image_size,
         batch_size=batch_size
    )
    return trainDs, valDs

datasetTrain, dataVal = create_dss("./pictures/unspecific/TrainingSet/", "inferred", (250, 250), 32)

# Erstes Modell erstellen
model1 = create_model_basic(input_shape=(250, 250) + (3,))

model1.compile(
    optimizer="rmsprop",
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['acc']
)

# Trainieren und Testen
print("Training model1")
history = model1.fit(datasetTrain, epochs=10, validation_data=dataVal)
