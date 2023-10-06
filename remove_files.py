import os 
from pathlib import Path

for i in range(0,38): 
    file = 'auto_'+str(i)+'.pkl'
    os.remove(file)