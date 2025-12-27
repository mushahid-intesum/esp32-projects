import tensorflow as tf
import librosa
import numpy as np
import os
from constants import MFCC_TIMESTEPS, MFCC_FEATURES

def load_wav_to_mfcc(file_path, n_mfcc, sr, max_len):
    audio, _ = librosa.load(file_path.numpy().decode('utf-8'), sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    mfcc_transposed = mfcc.T
    
    return mfcc_transposed.astype(np.float32)

def process_file(file_path, label, n_mfcc=MFCC_FEATURES, sr=22050, max_len=MFCC_TIMESTEPS):
    mfcc = tf.py_function(
        func=lambda x: load_wav_to_mfcc(x, n_mfcc, sr, max_len),
        inp=[file_path],
        Tout=tf.float32
    )
    if max_len is not None:
        mfcc.set_shape([max_len, n_mfcc])
    else:
        mfcc.set_shape([None, n_mfcc])
    
    return mfcc, label

def create_bird_dataset(base_path, n_mfcc=MFCC_FEATURES, sr=22050, max_len=MFCC_TIMESTEPS, batch_size=8, shuffle=True, validation_split=0.2):
    file_paths = []
    labels = []
    class_names = []
    
    for class_idx, folder_name in enumerate(sorted(os.listdir(base_path))):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            class_names.append(folder_name)
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    file_paths.append(os.path.join(folder_path, file_name))
                    labels.append(class_idx)
    
    total_samples = len(file_paths)
    indices = np.arange(total_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(total_samples * (1 - validation_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_paths = [file_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [file_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=len(train_paths))
    
    train_dataset = train_dataset.map(
        lambda x, y: process_file(x, y, n_mfcc, sr, max_len),
        num_parallel_calls=1
    )
    
    val_dataset = val_dataset.map(
        lambda x, y: process_file(x, y, n_mfcc, sr, max_len),
        num_parallel_calls=1
    )
    
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(1)
    
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(1)
    
    return (train_dataset, val_dataset, class_names, total_samples)