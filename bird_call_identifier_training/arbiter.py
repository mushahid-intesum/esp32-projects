import tensorflow as tf
from tensorflow import keras

def build_feature_extractor_arbiter_with_value(img_size, latent_dim, latent_filters, num_experts):
    num_actions = num_experts + 1
    
    inputs = keras.layers.Input(shape=(img_size, img_size, 3))
    
    x = keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('swish')(x)
    
    x = keras.layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('swish')(x)
    x = keras.layers.Conv2D(16, (1, 1), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('swish')(x)
    
    x = keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('swish')(x)
    x = keras.layers.Conv2D(24, (1, 1), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('swish')(x)
    
    x = keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('swish')(x)
    x = keras.layers.Conv2D(40, (1, 1), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('swish')(x)
    
    x = keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('swish')(x)
    x = keras.layers.Conv2D(latent_dim, (1, 1), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    latent_features = keras.layers.Activation('swish')(x)

    print(latent_features.shape)
    
    pooled = keras.layers.GlobalAveragePooling2D()(x)
    
    policy = keras.layers.Dense(64, activation='relu')(pooled)
    policy_logits = keras.layers.Dense(num_actions, activation=None, name='policy_logits')(policy)
    
    value = keras.layers.Dense(64, activation='relu')(pooled)
    value_output = keras.layers.Dense(1, activation=None, name='value')(value)
    
    model = keras.Model(
        inputs=inputs, 
        outputs=[latent_features, policy_logits, value_output],
        name='arbiter_feature_extractor_ac'
    )
    
    return model

def sample_action(policy_logits, temperature=1.0):
    policy_logits = policy_logits / temperature
    action = tf.random.categorical(policy_logits, 1)
    
    return action