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
samples, data_length = len(data), len(data[0][:200])


x_train = data[5:10].reshape((-1, data_length, 1))
x_test = data[15:20].reshape((-1, data_length, 1))
# Input layer
input_window = Input(shape=(data_length, 1))


# Encoder
x = Conv1D(24, 8, activation="relu", padding="same")(input_window) 
print('1 Encoded: ', np.shape(x)) 
x = MaxPooling1D(2, padding="same")(x)  # 5 dims
print('2 Encoded: ', np.shape(x)) 
x = Conv1D(12, 8, activation="relu", padding="same")(x)  # 5 dims
print('3 Encoded: ', np.shape(x)) 
x = MaxPooling1D(2, padding="same")(x)  # 5 dims
print('4 Encoded: ', np.shape(x)) 
x = Conv1D(2, 8, activation="relu", padding="same")(x)  # 5 dims
print('5 Encoded: ', np.shape(x)) 
encoded = MaxPooling1D(2, padding="same")(x)  # 3 dims

# Encoder model
encoder = Model(input_window, encoded)
print('ENCODED se FINAL shape:' , np.shape(encoded))


# Decoder
x = Conv1D(2, 8, activation="relu", padding="same")(encoded)  # 3 dims
print('1 Decoded: ', np.shape(x)) 
x = UpSampling1D(2)(x) 
print('1 Decoded: ', np.shape(x)) 
x = Conv1D(12, 8, activation="relu", padding="same")(x)  # 6 dims (not encoded)
print('1 Decoded: ', np.shape(x)) 
x = UpSampling1D(2)(x)  # 12 dims
print('1 Decoded: ', np.shape(x)) 
x = Conv1D(24, 8, activation='relu', padding="same")(x)  # 12 dims
print('1 Decoded: ', np.shape(x)) 
x = UpSampling1D(2)(x)  # 24 dims
print('1 Decoded: ', np.shape(x)) 
decoded = Conv1D(1, 4, activation='sigmoid', padding='same')(x)  # 24 dims
print('DECODED se FINAL shape:' , np.shape(decoded))


# Autoencoder model
autoencoder = Model(input_window, decoded)
autoencoder.summary()

# Compile and train the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
epochs = 10  # Adjust as needed
history =autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=10, shuffle=True)
decoded_eeg = autoencoder.predict(x_test)
result_file = "actual_vs_predicted.txt"

with open(result_file, "w") as file:
    file.write("Actual\tPredicted\n")
    for i in range(len(x_test)):
        actual = x_test[i].flatten()
        predicted = decoded_eeg[i].flatten()
        file.write("\t".join([str(val) for val in actual]) + "\t" + "\t".join([str(val) for val in predicted]) + "\n")

# Save the loss history to a text file
loss_history_file = "loss_history.txt"
with open(loss_history_file, "w") as file:
    file.write("Epoch\tLoss\n")
    for i in range(epochs):
        file.write(f"{i + 1}\t{history.history['loss'][i]}\n")