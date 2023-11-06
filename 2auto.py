import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pyedflib
from pathlib import Path
import pickle
import os
import json

print('Starting 2auto')
def getFile(_start): 
    for i in range(_start,100):
        file = "2auto_"+str(i)+".pkl"
        model_file = "2auto_"+str(i)+".json"
        if not os.path.exists(file): 
            Path(file).touch()
            return file, model_file
        elif os.path.getsize(file)/(1024*1024)>9: 
            file, model_file = getFile(i+1)
            return file, model_file
        else:
            return file, model_file

data_files = ['chb01_01.edf', 'chb01_02.edf', 'chb01_03.edf', 'chb01_04.edf', 'chb01_05.edf','chb01_06.edf',
              'chb01_07.edf','chb01_08.edf',
              'chb01_09.edf','chb01_10.edf', 
              'chb02_01.edf', 'chb02_02.edf', 'chb02_03.edf', 'chb02_04.edf', 'chb02_05.edf', 
              'chb02_06.edf',
            'chb02_07.edf', 'chb02_08.edf', 'chb02_09.edf', 'chb02_10.edf', 'chb02_11.edf', 
            'chb02_12.edf', 'chb02_13.edf', 'chb02_14.edf', 'chb02_15.edf'
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


train_data, test_data = data[:460], data[460:]  
# Define the autoencoder architecture
input_dim = train_data.shape[1]
latent_dim = input_dim // 2  # Halve the complexity as per your requirement
first_layer = input_dim//1.25
second_layer = input_dim //1.5
third_layer = input_dim //1.75
# Encoder
encoder_input = keras.Input(shape=(input_dim,))
encoder = layers.Dense(first_layer, activation='elu')(encoder_input)
encoder = layers.Dense(second_layer, activation='elu')(encoder)
encoder = layers.Dense(third_layer, activation='elu')(encoder)
encoder_output = layers.Dense(latent_dim, activation='elu')(encoder)

# Decoder
decoder_input = keras.Input(shape=(latent_dim,))
decoder = layers.Dense(third_layer, activation='elu')(decoder_input)
decoder = layers.Dense(second_layer, activation='elu')(decoder)
decoder = layers.Dense(first_layer, activation='elu')(decoder)
decoder_output = layers.Dense(input_dim, activation='elu')(decoder)

# Models
encoder_model = keras.Model(encoder_input, encoder_output, name="encoder")
decoder_model = keras.Model(decoder_input, decoder_output, name="decoder")
autoencoder_model = keras.Model(encoder_input, decoder_model(encoder_output), name="autoencoder")

# Compile the autoencoder
autoencoder_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder_model.fit(train_data, train_data, epochs=50, batch_size=32, validation_data=(test_data, test_data))

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
