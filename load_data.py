'''Loads training data, presumed to be in ./data
'''
import numpy as np

def load_data():
    x = np.genfromtxt('data/train_x.csv', delimiter=",")
    y = np.genfromtxt('data/train_y.csv', dtype=int)

    test_split = 0.2
    np.random.seed(113)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    _, num_to_index = get_labels()
    x = x[indices]
    y = y[indices]
    y = [num_to_index[yi] for yi in y.tolist()]

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])

    return (x_train, y_train), (x_test, y_test)

def get_labels():
    numbers = set()
    for i in range(0,10):
        for j in range(0,10):
            numbers.add(i+j)
            numbers.add(i*j)

    mapping = [-1]*82
    for i, j in enumerate(numbers):
       mapping[j] = i 

    return numbers, mapping
