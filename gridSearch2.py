

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
from sklearn.metrics import mean_squared_error
import pickle


import pyedflib
print('Starting')

# with open("best_model_weights_and_params.pkl", "wb") as file:
#     saved_info = {
#         "model_weights": [[1,2,3], [4,5,6]],
#         "hyperparameters": {"batch_size": [8,16], "epochs": [20,30]},
#     }
#     pickle.dump(saved_info, file)

# with open("best_model_weights_and_params.pkl", "rb") as file:
#     saved_info = pickle.load(file)
#     best_weights = saved_info["model_weights"]
#     print(best_weights)
#     best_hyperparameters = saved_info["hyperparameters"]
#     print(best_hyperparameters)
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

# # Create a GridSearchCV instance
# grid_search = GridSearchCV(estimator=autoencoder, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

# # Fit the grid search to your data
# grid_search.fit(train_data, train_data)



def fit_and_print_params(estimator, param_grid, scoring, cv):
    best_loss = float('inf')
    best_params = None
    for i in range(0, len(param_grid["batch_size"])-2):
        for j in range(0, len(param_grid["epochs"])-2):
            batch = []
            batch.append(param_grid["batch_size"][i])
            batch.append(param_grid["batch_size"][i+1])
            epoch = []
            epoch.append(param_grid["epochs"][j])
            epoch.append(param_grid["epochs"][j+1])
            params = {"batch_size": batch, "epochs": epoch}
            print("Evaluating parameter combination ",  params)
            grid_search = GridSearchCV(estimator=estimator, param_grid=params, scoring=scoring, cv=cv)
            grid_search.fit(train_data, train_data)
            best_model = grid_search.best_estimator_

            # Make predictions on the test data
            test_predictions = best_model.predict(test_data)

            # Calculate the evaluation metric (e.g., mean squared error in this case)
            mse = mean_squared_error(test_data, test_predictions)
            if (mse<best_loss): 
                best_loss=mse
                best_params= params
                text_to_append = 'Best loss: '+ str(mse)+ ' With params: '+str(params) 
                with open("26_Sept.txt", "a") as file:
                    # Append the text to the file
                    file.write(text_to_append + "\n")
                with open("best_model_weights_and_params.pkl", "wb") as file:
                    saved_info = {
                        "model_weights": best_model.get_weights(),
                        "hyperparameters": params,
                    }
                    pickle.dump(saved_info, file)
        

fit_and_print_params(estimator=autoencoder, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
# Print the best hyperparameters
# print("Best Hyperparameters:")
# print(grid_search.best_params_)

# Retrieve the best model from the grid search
# best_model = grid_search.best_estimator_.model

# Evaluate the best model
# loss = best_model.evaluate(test_data, test_data)
# print("Test Loss:", loss)



###############################
#Saved model code#
# def create_autoencoder(optimizer='adam', units_input=64, units_hidden=10, units_output=64, weights=None):
#     model = Sequential()
#     model.add(Dense(units=units_input, activation='relu', input_dim=train_data.shape[1]))
#     model.add(Dense(units=units_hidden, activation='relu'))
#     model.add(Dense(units=units_output, activation='relu'))
#     model.add(Dense(train_data.shape[1], activation='sigmoid'))

#     if weights is not None:
#         model.set_weights(weights)
    
#     model.compile(loss='mse', optimizer=optimizer)
#     return model

# with open("best_model_weights_and_params.pkl", "rb") as file:
#     saved_info = pickle.load(file)
#     best_weights = saved_info["model_weights"]
#     best_hyperparameters = saved_info["hyperparameters"]

# # Create the model with the saved weights and hyperparameters
# loaded_model = create_autoencoder(weights=best_weights, **best_hyperparameters)


