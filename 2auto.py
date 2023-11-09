import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pyedflib
from pathlib import Path
import pickle
import os
import json
from keras.layers import LeakyReLU
from keras import optimizers
import math

print('Starting 2auto')
def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    print('MY MAX: ', max_val)
    print('...............')
    print("MY MIN VALUE ", min_val)
    print('...............')
    normalized_data = (data - min_val) / (max_val - min_val)
    for i in range(0, len(normalized_data)):
        for j in range(0, len(normalized_data[i])): 
            if math.isnan(normalized_data[i][j]): 
                normalized_data[i][j] = 0 
        
def getFile(_start): 
    for i in range(_start,100):
        file = "relusigmoid_normalized"+str(i)+".pkl"
        model_file = "relusigmoid_normalized"+str(i)+".json"
        if not os.path.exists(file): 
            Path(file).touch()
            return file, model_file
        elif os.path.getsize(file)/(1024*1024)>9: 
            file, model_file = getFile(i+1)
            return file, model_file
        else:
            return file, model_file

# data_files = ['chb01_01.edf', 'chb01_02.edf', 'chb01_03.edf', 'chb01_04.edf', 'chb01_05.edf','chb01_06.edf',
#               'chb01_07.edf','chb01_08.edf',
#               'chb01_09.edf','chb01_10.edf', 
#               'chb02_01.edf', 'chb02_02.edf', 'chb02_03.edf', 'chb02_04.edf', 'chb02_05.edf', 
#               'chb02_06.edf',
#             'chb02_07.edf', 'chb02_08.edf', 'chb02_09.edf', 'chb02_10.edf', 'chb02_11.edf', 
#             'chb02_12.edf', 'chb02_13.edf', 'chb02_14.edf', 'chb02_15.edf'
# ]

data_files = ['chb01_01.edf', 'chb01_02.edf', 'chb01_03.edf', 'chb01_04.edf', 'chb01_05.edf','chb01_06.edf',
              'chb01_07.edf','chb01_08.edf',
              'chb01_09.edf','chb01_10.edf', 
              'chb02_01.edf'
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
train_data, test_data = data[:230], data[230:]  
# Define the autoencoder architecture
input_dim = train_data.shape[1]
#latent_dim = input_dim // 2  
# first_layer = int(latent_dim * 1.3)  
# second_layer = int(latent_dim * 1.2)  
# third_layer = int(latent_dim * 1.1) 
first_layer = 2048
second_layer = 1024
third_layer = 512
latent_dim = 256


# Encoder
encoder_input = keras.Input(shape=(input_dim,))
encoder = layers.Dense(first_layer, activation='relu')(encoder_input)
encoder = layers.Dense(second_layer, activation='relu')(encoder)
encoder = layers.Dense(third_layer, activation='relu')(encoder)
encoder_output = layers.Dense(latent_dim, activation='relu')(encoder)

# Decoder
decoder_input = keras.Input(shape=(latent_dim,))
decoder = layers.Dense(third_layer, activation='relu')(decoder_input)
decoder = layers.Dense(second_layer, activation='relu')(decoder)
decoder = layers.Dense(first_layer, activation='relu')(decoder)
decoder_output = layers.Dense(input_dim, activation='sigmoid')(decoder)

# Models
encoder_model = keras.Model(encoder_input, encoder_output, name="encoder")
decoder_model = keras.Model(decoder_input, decoder_output, name="decoder")
autoencoder_model = keras.Model(encoder_input, decoder_model(encoder_output), name="autoencoder")


custom_optimizer = optimizers.Adam(lr=0.001)  
autoencoder_model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

# Train the autoencoder
autoencoder_model.fit(train_data, train_data, epochs=10, batch_size=32, validation_data=(test_data, test_data))

#Evaluate the model if needed
autoencoder_model.evaluate(test_data, test_data)

file, file_model = getFile(0)
with open(file, "wb") as file:
    saved_info = {
        "model_weights": autoencoder_model.get_weights(), 
    }
    pickle.dump(saved_info, file)
with open(file_model, "wb") as file:
    saved_info = {
        "model_architecture": autoencoder_model.to_json()
    }
    json_string = json.dumps(saved_info)
    json_bytes = json_string.encode('utf-8')
    file.write(json_bytes)


print("Testing to JSON: ", autoencoder_model.to_json())
print("DONNNEEEE")
