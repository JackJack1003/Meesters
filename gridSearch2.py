

from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense
from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasRegressor


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

data = data / 255.0


train_data, test_data = data[:200], data[200:]


# Define a function to create the autoencoder model
def create_autoencoder(optimizer='adam', units_input=64, units_hidden=10, units_output=64):
    model = Sequential()
    model.add(Dense(units=units_input, activation='relu', input_dim=train_data.shape[1]))
    model.add(Dense(units=units_hidden, activation='relu'))
    model.add(Dense(units=units_output, activation='relu'))
    model.add(Dense(train_data.shape[1], activation='sigmoid'))
    model.compile(loss='mse', optimizer=optimizer)
    return model

# Wrap the Keras model in a scikit-learn estimator
autoencoder = KerasRegressor(build_fn=create_autoencoder, epochs=30, batch_size=64, verbose=0)

# Define the grid of batch sizes and epochs you want to search
param_grid = {
    'batch_size': [8,16,32, 64, 128],
    'epochs': [20, 30, 40]
}

# Create a GridSearchCV instance
grid_search = GridSearchCV(estimator=autoencoder, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

# Fit the grid search to your data
grid_search.fit(train_data, train_data)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(grid_search.best_params_)

# Retrieve the best model from the grid search
best_model = grid_search.best_estimator_.model

# Evaluate the best model
loss = best_model.evaluate(test_data, test_data)
print("Test Loss:", loss)
