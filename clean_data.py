'''Cleans data from the COMP 551 Kaggle dataset.
'''

import numpy as np
from load_data import load_data
from load_data import get_labels

def get_clean_data():
    x, y = load_data()
    
    # Binarize input data

    x[x > 200] = 255
    x[x < 255] = 0

    # TODO: Clean up noise

    x = x.reshape(x.shape[0], 64, 64, 1)
    test_split = 0.2
    np.random.seed(113)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    _, num_to_index, _ = get_labels()
    x = x[indices]
    y = y[indices]
    y = [num_to_index[yi] for yi in y.tolist()]

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])

    return (x_train, y_train), (x_test, y_test)
