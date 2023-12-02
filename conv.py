import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyedflib
from pathlib import Path
import pickle
import os
import json
from keras.layers import LeakyReLU
from keras import optimizers
import math
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model

print('Starting 2auto')
def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    print('MY MAX: ', max_val)
    print('...............')
    print("MY MIN VALUE ", min_val)
    print('...............')
    threshold = 1e-10
    if np.any((max_val-min_val)<threshold): 
        min_val *=0.95
    normalized_data = (data - min_val) / (max_val - min_val)
    nan_indices = np.isnan(normalized_data)
    # Replace NaN values with 0
    normalized_data[nan_indices] = 0
    return normalized_data
        

data_files = ['chb01_01.edf'
]
data = []

for d in data_files:
    file_path = d
    edf_file = pyedflib.EdfReader(file_path)
    channel_labels = edf_file.getSignalLabels()
    for channel_label in channel_labels:
        channel_data = edf_file.readSignal(channel_labels.index(channel_label))
        data.append(channel_data)
    edf_file.close()
    print('Done reading', d)

data = np.array(data)


data = data / 255.0
data = normalize(data)

print('MY DATA IS ', data)
print('...................')
train_data, test_data = data[:20], data[20:]  
# Assuming your input data has a shape of (23, 912600)
# Adjust these values based on your actual data
samples, data_length = len(data), len(data[0])


x_train = data[:20].reshape((20, data_length, 1))
# Input layer
input_window = Input(shape=(data_length, 1))


# Encoder
x = Conv1D(16, 3, activation="relu", padding="same")(input_window)  
x = MaxPooling1D(2, padding="same")(x)  # 5 dims
x = Conv1D(1, 3, activation="relu", padding="same")(x)  # 5 dims
encoded = MaxPooling1D(2, padding="same")(x)  # 3 dims

# Encoder model
encoder = Model(input_window, encoded)

# Decoder
x = Conv1D(1, 3, activation="relu", padding="same")(encoded)  # 3 dims
x = UpSampling1D(2)(x)  # 6 dims
x = Conv1D(16, 2, activation='relu', padding="same")(x)  # 5 dims
x = UpSampling1D(2)(x)  # 10 dims
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)  # 10 dims

# Autoencoder model
autoencoder = Model(input_window, decoded)
autoencoder.summary()

# Compile and train the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
epochs = 10  # Adjust as needed
autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=1, shuffle=True)
