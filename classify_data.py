from tensorflow.python.keras.models import load_model
from load_data import load_testing_data, get_labels
from clean_data import threshold, remove_dots
import scipy.misc, scipy.ndimage
import numpy as np
import sys

print("Loading model...")
model = load_model('data/weights.hdf5')
print("Loaded model!")

print("Loading test data...")
x_test = load_testing_data()
print("Removing background...")
x_test = threshold(x_test)
print("Removing dots...")
x_test = remove_dots(x_test)
x_test = x_test.reshape(-1, 64, 64, 1)

print("Printing data...")
predictions = model.predict(x_test)

labels, _, _ = get_labels()
labels = list(labels)

with open('data/y_test.csv', 'w+') as output:
    print("Id,Category", file=output)
    for i, prediction in enumerate(predictions):
        print(str(i + 1) + "," + str(labels[np.argmax(prediction)]), file=output)
