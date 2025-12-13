import tensorflow as tf
from tensorflow import keras
import os

IMG_SIZE = 224
BATCH_SIZE = 32

def create_datasets(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Load and prepare train/test datasets"""
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Data augmentation for training
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    train_ds = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_ds = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    test_ds = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    num_classes = len(train_ds.class_indices)
    
    return train_ds, val_ds, test_ds, num_classes
