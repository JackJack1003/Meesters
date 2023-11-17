import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyedflib
from pathlib import Path
import pickle
import os
import json
from keras.layers import LeakyReLU
from keras import optimizers
import math

print('Starting torch')
def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    print('MY MAX: ', max_val)
    print('...............')
    print("MY MIN VALUE ", min_val)
    print('...............')
    threshold = 1e-10
    if np.any((max_val-min_val)<threshold): 
        min_val *=0.95
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


data_files = ['chb01_01.edf', 'chb01_02.edf', 'chb01_03.edf', 'chb01_04.edf', 'chb01_05.edf','chb01_06.edf',
              'chb01_07.edf','chb01_08.edf',
              'chb01_09.edf','chb01_10.edf', 
              'chb02_01.edf'
]

#data_files = ['chb01_01.edf', 'chb01_02.edf', 'chb01_03.edf', 'chb01_04.edf']
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
data = normalize(data)


eeg_data_tensor = torch.tensor(data[:400], dtype=torch.float)

class EEGDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

eeg_dataset = EEGDataset(eeg_data_tensor)

data_loader = DataLoader(dataset=eeg_dataset, batch_size=64, shuffle=True)

class Autoencoder_EEG(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(len(data[0]), 4096),  # Adjust input size
            nn.ReLU(),
            nn.Linear(4096, 2048)

        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, len(data[0])),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder_EEG()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 5
outputs = []
for epoch in range(num_epochs):
    for eeg_batch in data_loader:
        recon = model(eeg_batch)
        loss = criterion(recon, eeg_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, eeg_batch, recon))


# Assuming you have test data in a variable called test_data

# Convert test data to a PyTorch tensor
test_data = torch.tensor(data[15:], dtype=torch.float)

# Pass the test data through the trained model's decoder
with torch.no_grad():
    reconstructed_data = model.decoder(model.encoder(test_data))


# plt.subplot(2,1,1)
# plt.plot(test_data[:500])

# plt.subplot(2,1,2)
# plt.plot(reconstructed_data[:500])
# plt.show()
single_file = '8_torch_eeg_4096.txt'
if not os.path.exists(single_file): 
    Path(single_file).touch()
with open(single_file, "w") as file:
    file.write(f'LOSS: {loss}')  
    for i in range(0,5): 
        np.savetxt(file, test_data[i], fmt="%f", delimiter=", ")
        file.write('---')   
    for i in range(0,5):  
        np.savetxt(file, reconstructed_data[i], fmt="%f", delimiter=", ")
        file.write('---')  
print('MY RECON IS: ', recon)
print('DONE')
