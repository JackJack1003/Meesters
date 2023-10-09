param_grid = {
    'batch_size': [8,16,32, 64, 128],
    'epochs': [20, 30, 40]
}

for params in param_grid["batch_size"]:
    print(params[0])