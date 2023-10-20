import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pickle
import os
import pyedflib
from pathlib import Path
import sys
print('Starting')


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
# print(train_data.shape[1:][0])
# sys.exit()
def getFile(_start): 
    for i in range(_start,100):
        file = "auto_"+str(i)+".pkl"
        if not os.path.exists(file): 
            Path(file).touch()
            return file
        elif os.path.getsize(file)/(1024*1024)>9: 
            file = getFile(i+1)
            return file
        else:
            return file


def add_layers(in_out, original_in_out, latent, reverse, x, input_layer): 
    print(in_out)
    if x == 0: 
        new_x =  layers.Dense(round(in_out), activation='linear')(input_layer)
    else: 
        new_x = layers.Dense(round(in_out), activation='linear')(x)
    step = (original_in_out-latent)/10
    if (reverse):
        if (in_out<original_in_out): 
            in_out += step
            in_out = round(in_out,2)
            out_x = add_layers(in_out, original_in_out, latent, reverse, new_x, input_layer)
        else: 
            return x
    else: 
        if (in_out>latent): 
            in_out -=step
            in_out = round(in_out,2)
            out_x =add_layers(in_out, original_in_out, latent, reverse, new_x,  input_layer)
        else: 
            reverse = True
            in_out += step
            out_x = add_layers(in_out, original_in_out, latent, reverse, new_x, input_layer)
    return out_x



# Step 4: Build the autoencoder model
def build_autoencoder(input_shape, input_output_size, latent_space):
    input_layer = layers.Input(shape=input_shape)
    print(np.shape(input_layer))

    # # Encoder
  
    # x = layers.Dense(input_output_size, activation='elu')(input_layer)
    # x = layers.Dense(latent_space, activation='elu')(x)

    # # Decoder
    # x = layers.Dense(input_output_size, activation='elu')(x)
    #x = add_layers(input_output_size, input_output_size, latent_space,False,0, input_layer )
    output_layer = add_layers(input_output_size, input_output_size, latent_space,False,0, input_layer )

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
all_batches = [20,30,40]
all_in_out = [train_data.shape[1:][0]/2000,256]
all_latent = [train_data.shape[1:][0]/4000,64]
count = 0
for e in all_epochs:
    for b in all_batches:
        for i in all_in_out:
            for l in all_latent:
                count +=1
                loss, summary, model, hyperparameters = run_combo(e, b, i, l)

                file = getFile(0)
                with open(file, "wb") as file:
                    saved_info = {
                        "model_weights": model.get_weights(),
                        "hyperparameters": hyperparameters,
                    }
                    pickle.dump(saved_info, file)
                    print('CYCLE KLAAR: ', count, '/48')
                    print('----------------------------------------------------------------')
                text_to_append = 'Best loss: ' + str(loss) + ' With params: ' + str(hyperparameters) + ' AND file :' +str(file)+'\n'
                with open("09_Oct.txt", "a") as file:
                    file.write(text_to_append + "\n")