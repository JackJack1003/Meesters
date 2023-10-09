import os 
from pathlib import Path

for i in range(0,38): 
    file = 'auto_'+str(i)+'.pkl'
    if os.path.exists(file): 
        os.remove(file)