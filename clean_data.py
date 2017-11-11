'''Cleans data from the COMP 551 Kaggle dataset.
'''

import cv2
import numpy as np
from load_data import load_data
from load_data import get_labels
import scipy.misc

def get_clean_data():
    x, y = load_data()
    x = np.uint8(x)

    # Reshape images from 1x4096 to 64x64

    x = x.reshape(-1, 64, 64)
    
    # Binarize input data

    x[x > 240] = 255
    x[x < 255] = 0

    # Clean up noise
    
    for i, img in enumerate(x):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # Minimum number of connected pixels to keep
        min_size = 20

        img2 = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        x[i] = img2.reshape(64, 64)

    x = x.reshape(-1, 64, 64, 1)

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

get_clean_data()
