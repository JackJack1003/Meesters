import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python script.py single_file")
    sys.exit(1)

single_file = sys.argv[1]

# Initialize lists to store the data
test = []
reconstructed = []


with open(single_file, "r") as file:
    lines = file.readlines()

values = []
for l in lines: 
    if '---' in l: 
        l.replace('---', '')
        values.append(l)
        if (len(test)<6): 
            test.append(values)
        else: 
            reconstructed.append(values)
        values = []
    else: 
        values.append(l)

test = test[1:]
test = np.array(test)
reconstructed = np.array(reconstructed)
num_examples = 4
# for i in range(0, 4): 
#     to_delete = []
#     for j in range(0,len(test[i])-1): 
#         if test[i][j].dtype != np.dtype('float64'):
#             to_delete.append(j)
#     for d in to_delete: 
#         test[i] = np.delete(test[i], d)

# for i in range(0, 4): 
#     to_delete = []
#     for j in range(0,len(reconstructed[i])-1): 
#         if reconstructed[i][j].dtype != np.dtype('float64'):
#             to_delete.append(j)
#     for d in to_delete: 
#         reconstructed[i] = np.delete(reconstructed[i], d)


# print(len(test[0]))
# print(len(reconstructed[0]))
    
# min_value = test.min()
# max_value = test.max()
# normalized_test = (test[0] - min_value) / (max_value - min_value)
# min_value = reconstructed.min()
# max_value = reconstructed.max()
# normalized_reconstructed = (reconstructed - min_value) / (max_value - min_value)
x = range(0,2000)
plt.figure(figsize=(12, 6))
for i in range(num_examples):
    # Original data
    plt.subplot(2, num_examples, i + 1)
    plt.plot( test[i][:2000])
    plt.title(f"Original {i + 1}")

    # Reconstructed data
    plt.subplot(2, num_examples, i + num_examples + 1)
    plt.plot(reconstructed[i][:2000])
    plt.title(f"Reconstructed {i + 1}")
    print('Klaar met ' , i)
    

plt.tight_layout()
plt.show()
