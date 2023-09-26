import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense
from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential


import pyedflib
print('Starting')
data_files = ['chb01_01.edf', 'chb01_02.edf', 'chb01_03.edf', 'chb01_04.edf']
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


train_data, test_data = data[:84], data[84:]


def create_autoencoder(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=128, step=32), activation='relu', input_dim=train_data.shape[1]))
    model.add(Dense(units=hp.Int('units_hidden', min_value=8, max_value=16, step=2), activation='relu'))
    model.add(Dense(units=hp.Int('units_output', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dense(train_data.shape[1], activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')
    return model

# Define the tuner and search space
tuner = RandomSearch(
    create_autoencoder,
    objective='val_loss',
    max_trials=100,  # Number of trials (adjust as needed)
    executions_per_trial=1,  # Number of runs per trial
    directory='my_dir'  # Directory to store results
)

# Split your data into training and validation sets

# Perform hyperparameter tuning
tuner.search(train_data, train_data, validation_data=(test_data, test_data), epochs=30, batch_size=64)

# Print the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print(best_hyperparameters.values)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Fit the best model to your data
best_model.fit(train_data, train_data, epochs=30, batch_size=64)

# Evaluate the model
loss = best_model.evaluate(test_data, test_data)
print("Test Loss:", loss)


# # Step 4: Build the autoencoder model
# def build_autoencoder(input_shape):
#     input_layer = layers.Input(shape=input_shape)
#     print(np.shape(input_layer))

#     # Encoder
#     x = layers.Dense(128, activation='relu')(input_layer)
#     x = layers.Dense(64, activation='relu')(x)

#     # Decoder
#     x = layers.Dense(128, activation='relu')(x)
#     output_layer = layers.Dense(input_shape[0], activation='sigmoid')(x)

#     model = models.Model(input_layer, output_layer)
#     model.compile(optimizer='nadam', loss='mean_squared_error')

#     return model

# # Build the autoencoder model with the input shape (921600,)
# #autoencoder_model = build_autoencoder(train_data.shape[1:])
# autoencoder = KerasRegressor(build_fn=build_autoencoder(train_data.shape[1:]), verbose=0)

# epochs = list(range(0, 101, 10))
# batches = list(range(0, 51, 5))
# optimizer= ['adam', 'sgd', 'rmsprop']

# param_grid = {'epochs': epochs, 'batches': batches, 'optimizer': optimizer }

# grid_search = GridSearchCV(estimator=autoencoder_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
# grid_search.fit(train_data, train_data)

# print("Best Hyperparameters: ", grid_search.best_params_)
# print("Best Score: ", grid_search.best_score_)
# autoencoder_model.summary()

# Step 5: Train the autoencoder
# epochs = 30
# batch_size = 3

# history = autoencoder_model.fit(
#     train_data, train_data,  # Input and target are the same for an autoencoder
#     epochs=epochs,
#     batch_size=batch_size,
#     shuffle=True,
#     validation_data=(test_data, test_data)
# )

# Step 6: Evaluate the autoencoder (optional)
# Evaluate the model on the test data
#loss = autoencoder_model.evaluate(test_data, test_data, batch_size=batch_size)
# print("Test Loss:", loss)

# # Step 7: Plot the original and reconstructed data
# num_examples = 5  # Number of examples to visualize

# # Generate reconstructed data using the trained autoencoder
# reconstructed_data = autoencoder_model.predict(test_data)

# # Plot the original and reconstructed data
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




