import numpy as np
import pyedflib
from scipy.optimize import minimize
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import mean_squared_error

# Load dataset
data_files = ['chb01_01.edf', 'chb01_02.edf', 'chb01_03.edf', 'chb01_04.edf']
data = []

for d in data_files:
    file_path = d
    # Load the EDF file
    edf_file = pyedflib.EdfReader(file_path)
    # Get the list of signal labels (channel names) in the EDF file
    channel_labels = edf_file.getSignalLabels()
    # Read the data from the EDF file
    for channel_label in channel_labels:
        channel_data = edf_file.readSignal(channel_labels.index(channel_label))
        data.append(channel_data)
    # Close the EDF file after reading the data
    edf_file.close()
    print('Done reading', d)

data = np.array(data)
data = data / 255.0

# Split the data into training and testing sets
train_data, test_data = data[:84], data[84:]

# Autoencoder optimizer function
def autoencoder_optimizer(params):
    learning_rate, hidden_units = params
    input_layer = Input(shape=(train_data.shape[1],))
    encoded = Dense(hidden_units, activation='relu')(input_layer)
    decoded = Dense(train_data.shape[1], activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(train_data, train_data, epochs=10, batch_size=32, verbose=0)
    predictions = autoencoder.predict(test_data)
    loss = mean_squared_error(test_data, predictions)
    return loss

# Initial guess for hyperparameters
initial_params = [1e-1, 10]  # Learning rate and hidden units

# Perform function optimization
result = minimize(autoencoder_optimizer, initial_params, method='Nelder-Mead')

# Get best hyperparameters and corresponding loss
best_params = result.x
best_loss = result.fun

print("Best Hyperparameters:", best_params)
print("Best Loss:", best_loss)
