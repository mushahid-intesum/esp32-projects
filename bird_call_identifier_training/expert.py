import tensorflow as tf
from tensorflow import keras

def build_expert_model(latent_dim, num_classes=10):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=latent_dim),
        keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        
        # Depthwise Separable Convolution Block 1
        keras.layers.DepthwiseConv2D((3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Conv2D(16, (1, 1), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        
        # Depthwise Separable Convolution Block 2
        keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Conv2D(24, (1, 1), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        
        # Depthwise Separable Convolution Block 3
        keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Conv2D(40, (1, 1), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        
        # Depthwise Separable Convolution Block 4
        keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Conv2D(80, (1, 1), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        
        # Global Average Pooling
        keras.layers.GlobalAveragePooling2D(),
        
        # Final classification layer  
        keras.layers.Dense(num_classes, activation='softmax')
    ], name='expert_model')

    return model
