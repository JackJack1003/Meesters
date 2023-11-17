import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pyedflib
import os
from pathlib import Path

# Function to load EEG data from EDF files
def load_eeg_data(data_files):
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
    return data

# Load EEG data
data_files = ['chb01_01.edf', 'chb01_02.edf', 'chb01_03.edf', 'chb01_04.edf', 'chb01_05.edf','chb01_06.edf',
              'chb01_07.edf','chb01_08.edf',
              'chb01_09.edf','chb01_10.edf', 
              'chb02_01.edf', 'chb02_02.edf', 'chb02_03.edf', 'chb02_04.edf', 'chb02_05.edf', 
              'chb02_06.edf',
            'chb02_07.edf', 'chb02_08.edf', 'chb02_09.edf', 'chb02_10.edf', 'chb02_11.edf', 
            'chb02_12.edf', 'chb02_13.edf', 'chb02_14.edf', 'chb02_15.edf'
]
EEG_DATA = load_eeg_data(data_files)

# Normalize EEG data
EEG_DATA = EEG_DATA / 255.0

# Define the autoencoder architecture
input_dim = EEG_DATA.shape[1]
latent_dim = input_dim // 2 

# Define generator model
def build_generator(latent_dim, output_dim):
    model = keras.Sequential()
    model.add(layers.Dense(128, input_dim=latent_dim, activation='relu'))
    model.add(layers.Dense(output_dim, activation='tanh'))
    return model

# Define discriminator model
def build_discriminator(input_dim):
    model = keras.Sequential()
    model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Build and compile the discriminator
discriminator = build_discriminator(input_dim=input_dim)
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002))

# Build and compile the generator
generator = build_generator(latent_dim=latent_dim, output_dim=input_dim)
discriminator.trainable = False  # Freeze the discriminator during GAN training

gan_input = keras.Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002))

# Training the GAN
batch_size = 64
epochs = 50 #10000

for epoch in range(epochs):
    for _ in range(len(EEG_DATA) // batch_size):
        # Train the discriminator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_data = generator.predict(noise)
        real_data = EEG_DATA[np.random.randint(0, EEG_DATA.shape[0], batch_size)]
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_data, labels_real)
        d_loss_fake = discriminator.train_on_batch(generated_data, labels_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        labels_gan = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, labels_gan)

    # Print progress
    print(f"Epoch: {epoch} D Loss: {d_loss} G Loss: {g_loss}")

    # Optionally, save generated EEG-like data for analysis

# Generate and save synthetic EEG data
num_samples = 100
noise = np.random.normal(0, 1, (num_samples, latent_dim))
generated_data = generator.predict(noise)
np.save('synthetic_eeg.npy', generated_data)

# # Plot generated vs. real EEG data
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(EEG_DATA[0])
# plt.title('Real EEG Data')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')

# plt.subplot(2, 1, 2)
# plt.plot(generated_data[0])
# plt.title('Generated EEG Data')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')

# plt.tight_layout()
# plt.show()

single_file = 'gan_0_1.txt'
if not os.path.exists(single_file): 
    Path(single_file).touch()
with open(single_file, "w") as file:
    for i in range(0,5): 
        np.savetxt(file, EEG_DATA[i], fmt="%f", delimiter=", ")
        file.write('---')   
    for i in range(0,5):  
        np.savetxt(file, generated_data[i], fmt="%f", delimiter=", ")
        file.write('---')  
print('DONE')
