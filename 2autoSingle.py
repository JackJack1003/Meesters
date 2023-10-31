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
import json

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




def load_model_and_hyperparameters(file_path):
    with open(file_path, "rb") as file:
        saved_info = pickle.load(file)
        print("Saved info keys:")
        print(saved_info.keys())
        model_weights = saved_info["model_weights"]
    file_path = file_path.replace('.pkl', '.json')
    print('Nuwe file vir model is ', file_path)
    with open(file_path, "r") as file:
        saved_info = json.load(file)
        model_architecture = saved_info["model_architecture"]
        #hyperparameters = saved_info["hyperparameters"]
    return model_weights, model_architecture


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='auto_0.pkl', help='Description of my_var')
args = parser.parse_args()
file_path =  args.file  
# print(file_path)

model_weights, model_architecture = load_model_and_hyperparameters(file_path)

print(model_architecture)
print("-------------------------------------------------------------")
new_autoencoder_model = keras.models.model_from_json(model_architecture)
new_autoencoder_model.set_weights(model_weights)
new_autoencoder_model.compile(optimizer='nadam', loss='mean_squared_error')
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