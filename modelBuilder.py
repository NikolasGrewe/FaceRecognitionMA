# Imports
import keras
from keras import layers

# Model 1: Unspezifische Gesichtserkennung: Kleines XCeption Model
def create_model_basic(input_shape):
    '''
    Erstellt das unspezifische Modell. "input_shape" ist die Auflösung des Bildes
    '''
    inputs = keras.Input(input_shape)
    # Image augmentation block
    x = layers.RandomRotation(factor=0.2)(inputs)
    x = layers.RandomFlip(mode="horizontal")(x)
    
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

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

# Model 2: Spezifische Gesichterkennung
def create_specific_model(input_shape, outs):
    '''
    Erstellt das spezifische Modell. "outs" ist die Anzahl Gesichter, "input_shape" die Auflösung des Bildes.
    '''
    inputs = keras.Input(input_shape)
    # Image augmentation block
    x = layers.RandomRotation(factor=0.2)(inputs)
    x = layers.RandomFlip(mode="horizontal")(x)
    
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x 

    for size in [128, 256, 512, 1024]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1152, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(outs, activation="softmax")(x)
    return keras.Model(inputs, outputs)
