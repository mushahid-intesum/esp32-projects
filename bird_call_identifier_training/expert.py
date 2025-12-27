import tensorflow as tf
from tensorflow import keras

def build_expert_model(latent_dim, num_classes=10):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=latent_dim),
        keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=keras.regularizers.l2(1e-2)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Dropout(0.2),
        
        # Depthwise Separable Convolution Block 1
        keras.layers.DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=keras.regularizers.l2(1e-2)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Conv2D(16, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-2)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Dropout(0.2),
        
        # Depthwise Separable Convolution Block 2
        keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', depthwise_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Conv2D(24, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Dropout(0.3),
        
        keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', depthwise_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Conv2D(40, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Dropout(0.3),
        
        keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', depthwise_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Conv2D(80, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Dropout(0.4),
        
        # Global Average Pooling
        keras.layers.GlobalAveragePooling2D(),
        
        # Final classification layer  
        keras.layers.Dense(num_classes, activation='softmax')
    ], name='expert_model')

    return model

import tensorflow as tf
from tensorflow import keras

def build_student_expert_model(latent_dim, num_classes=10):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=latent_dim),
        keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=keras.regularizers.l2(1e-2)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Dropout(0.2),
        
        # Depthwise Separable Convolution Block 1
        keras.layers.DepthwiseConv2D((3, 3), padding='same', depthwise_regularizer=keras.regularizers.l2(1e-2)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Conv2D(16, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-2)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('swish'),
        keras.layers.Dropout(0.2),
        
        # Depthwise Separable Convolution Block 2
        # keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', depthwise_regularizer=keras.regularizers.l2(1e-4)),
        # keras.layers.BatchNormalization(),
        # keras.layers.Activation('swish'),
        # keras.layers.Conv2D(24, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4)),
        # keras.layers.BatchNormalization(),
        # keras.layers.Activation('swish'),
        # keras.layers.Dropout(0.3),
        
        # keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', depthwise_regularizer=keras.regularizers.l2(1e-4)),
        # keras.layers.BatchNormalization(),
        # keras.layers.Activation('swish'),
        # keras.layers.Conv2D(40, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4)),
        # keras.layers.BatchNormalization(),
        # keras.layers.Activation('swish'),
        # keras.layers.Dropout(0.3),
        
        # keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', depthwise_regularizer=keras.regularizers.l2(1e-4)),
        # keras.layers.BatchNormalization(),
        # keras.layers.Activation('swish'),
        # keras.layers.Conv2D(80, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4)),
        # keras.layers.BatchNormalization(),
        # keras.layers.Activation('swish'),
        # keras.layers.Dropout(0.4),
        
        # Global Average Pooling
        keras.layers.GlobalAveragePooling2D(),
        
        # Final classification layer  
        keras.layers.Dense(num_classes, activation='softmax')
    ], name='expert_model')

    return model
