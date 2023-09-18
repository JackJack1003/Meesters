import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import pyedflib
print('Starting')
# Replace 'your_file.edf' with the name of your uploaded EDF file
file_path = 'chb01_01.edf'

# Load the EDF file
edf_file = pyedflib.EdfReader(file_path)

# Get the list of signal labels (channel names) in the EDF file
channel_labels = edf_file.getSignalLabels()


data = []
for channel_label in channel_labels:
    channel_data = edf_file.readSignal(channel_labels.index(channel_label))
    data.append(channel_data)


edf_file.close()
print("file gelees"); 
#data het 23 arrays in dit, vir 23 channels
#data[0] het 256 meetings per sekonde vir een uur in. So elke 256 meetings = 1 sekonde


# Step 1: Install the required libraries


# Step 2: Import the necessary libraries


# Assuming 'data' is the array containing your 23 one-dimensional arrays with 921600 entries each

# Step 3: Prepare the data
data = np.array(data)

# Normalize the data to values between 0 and 1
data = data / 255.0

# Split the data into training and testing sets
train_data, test_data = data[:18], data[18:]  # You can adjust the split as needed

# Step 4: Build the autoencoder model
def build_autoencoder(input_shape):
    input_layer = layers.Input(shape=input_shape)
    print(np.shape(input_layer))

    # Encoder
    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dense(64, activation='relu')(x)

    # Decoder
    x = layers.Dense(128, activation='relu')(x)
    output_layer = layers.Dense(input_shape[0], activation='sigmoid')(x)

    model = models.Model(input_layer, output_layer)
    model.compile(optimizer='nadam', loss='mean_squared_error')

    return model

# Build the autoencoder model with the input shape (921600,)
autoencoder_model = build_autoencoder(train_data.shape[1:])
autoencoder_model.summary()

# Step 5: Train the autoencoder
epochs = 30
batch_size = 3

history = autoencoder_model.fit(
    train_data, train_data,  # Input and target are the same for an autoencoder
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(test_data, test_data)
)

# Step 6: Evaluate the autoencoder (optional)
# Evaluate the model on the test data
loss = autoencoder_model.evaluate(test_data, test_data, batch_size=batch_size)
print("Test Loss:", loss)

# Step 7: Plot the original and reconstructed data
num_examples = 5  # Number of examples to visualize

# Generate reconstructed data using the trained autoencoder
reconstructed_data = autoencoder_model.predict(test_data)

# Plot the original and reconstructed data
plt.figure(figsize=(12, 6))
for i in range(num_examples):
    # Original data
    plt.subplot(2, num_examples, i + 1)
    plt.plot(test_data[i])
    plt.title(f"Original {i + 1}")

    # Reconstructed data
    plt.subplot(2, num_examples, i + num_examples + 1)
    plt.plot(reconstructed_data[i])
    plt.title(f"Reconstructed {i + 1}")

plt.tight_layout()
plt.show()

