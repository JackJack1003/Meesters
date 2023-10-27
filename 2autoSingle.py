import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pickle
import os
import pyedflib
from pathlib import Path
import gzip
import tensorflow_model_optimization as tfmot
import argparse

print('Starting 2 single')


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

# Normalize the data to values between 0 and 1
data = data / 255.0

# Split the data into training and testing sets
train_data, test_data = data[:460], data[460:]  # You can adjust the split as needed


def build_autoencoder(input_shape, latent_space):
    input_layer = layers.Input(input_shape)
    print(np.shape(input_layer))

    encoder_input = keras.Input(shape=(input_shape,))
    encoder = layers.Dense(128, activation='relu')(encoder_input)
    encoder = layers.Dense(64, activation='relu')(encoder)
    encoder = layers.Dense(32, activation='relu')(encoder)
    encoder_output = layers.Dense(latent_space, activation='relu')(encoder)

    # Decoder
    decoder_input = keras.Input(shape=(latent_space,))
    decoder = layers.Dense(32, activation='relu')(decoder_input)
    decoder = layers.Dense(64, activation='relu')(decoder)
    decoder = layers.Dense(128, activation='relu')(decoder)
    decoder_output = layers.Dense(input_layer, activation='sigmoid')(decoder)

    # Models
    encoder_model = keras.Model(encoder_input, encoder_output, name="encoder")
    decoder_model = keras.Model(decoder_input, decoder_output, name="decoder")
    autoencoder_model = keras.Model(encoder_input, decoder_model(encoder_output), name="autoencoder")

    # Compile the autoencoder
    autoencoder_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder_model.fit(train_data, train_data, epochs=50, batch_size=32, validation_data=(test_data, test_data))

    return autoencoder_model

def load_model_and_hyperparameters(file_path):
    with open(file_path, "rb") as file:
        saved_info = pickle.load(file)
        model_weights = saved_info["model_weights"]
        #hyperparameters = saved_info["hyperparameters"]
    return model_weights


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='auto_0.pkl', help='Description of my_var')
args = parser.parse_args()
file_path =  args.file  
# print(file_path)

model_weights = load_model_and_hyperparameters(file_path)

input_dim = train_data.shape[1]
latent_dim = input_dim // 2
new_autoencoder_model = build_autoencoder(input_dim, latent_dim)
new_autoencoder_model.set_weights(model_weights)

loss = new_autoencoder_model.evaluate(test_data, test_data, batch_size=10)
print('loss is ', loss)


reconstructed_data = new_autoencoder_model.predict(test_data)

single_file = str(file_path)+'_single.txt'
if not os.path.exists(single_file): 
    Path(single_file).touch()
with open(single_file, "w") as file:
    for i in range(0,5): 
        np.savetxt(file, test_data[i], fmt="%f", delimiter=", ")
        file.write('---')   
    for i in range(0,5):  
        np.savetxt(file, reconstructed_data[i], fmt="%f", delimiter=", ")
        file.write('---')   
        


# num_examples = 5
# plt.figure(figsize=(12, 6))
# for i in range(num_examples):
#     # Original data
#     plt.subplot(2, num_examples, i + 1)
#     plt.plot(test_data[i])
#     plt.title(f"Original {i + 1}")

#     # Reconstructed data
#     plt.subplot(2, num_examples, i + num_examples + 1)
#     plt.plot(reconstructed_data[i])
#     plt.title(f"Reconstructed {i + 1}")

# plt.tight_layout()
# plt.show()