import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pickle

import pyedflib
print('Starting')

data_files = ['chb01_01.edf', 'chb01_02.edf', 'chb01_03.edf', 'chb01_04.edf', 'chb01_05.edf','chb01_06.edf','chb01_07.edf','chb01_08.edf','chb01_09.edf','chb01_10.edf']
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
train_data, test_data = data[:200], data[200:]  # You can adjust the split as needed

# Step 4: Build the autoencoder model
def build_autoencoder(input_shape, input_output_size, latent_space):
    input_layer = layers.Input(shape=input_shape)
    print(np.shape(input_layer))

    # Encoder
    x = layers.Dense(input_output_size, activation='relu')(input_layer)
    x = layers.Dense(latent_space, activation='relu')(x)

    # Decoder
    x = layers.Dense(input_output_size, activation='relu')(x)
    output_layer = layers.Dense(input_shape[0], activation='sigmoid')(x)

    model = models.Model(input_layer, output_layer)
    model.compile(optimizer='nadam', loss='mean_squared_error')

    return model

# Build the autoencoder model with the input shape (921600,)
def run_combo(epochs, batches, in_out_layer, latent_space):
    autoencoder_model = build_autoencoder(train_data.shape[1:], in_out_layer, latent_space)
    autoencoder_model.summary()

    # Store hyperparameters in a dictionary
    hyperparameters = {
        "epochs": epochs,
        "batch_size": batches,
        "input_output_size": in_out_layer,
        "latent_space": latent_space,
    }

    history = autoencoder_model.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batches,
        shuffle=True,
        validation_data=(test_data, test_data)
    )

    loss = autoencoder_model.evaluate(test_data, test_data, batch_size=batches)

    return loss, history, autoencoder_model, hyperparameters

all_epochs = [10,20,30]
all_batches = [10,20,30,40]
all_in_out = [128,256]
all_latent = [32,64]

for e in all_epochs:
    for b in all_batches:
        for i in all_in_out:
            for l in all_latent:
                loss, summary, model, hyperparameters = run_combo(e, b, i, l)
                text_to_append = 'Best loss: ' + str(loss) + ' With params: ' + str(hyperparameters)
                with open("05_Oct.txt", "a") as file:
                    file.write(text_to_append + "\n")
                with open("auto.pkl", "wb") as file:
                    saved_info = {
                        "model_weights": model.get_weights(),
                        "hyperparameters": hyperparameters,
                    }
                    pickle.dump(saved_info, file)
                    print('EEEEEEENN CYCLE KLAAR')
                    print('----------------------------------------------------------------')