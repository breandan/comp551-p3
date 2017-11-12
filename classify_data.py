from tensorflow.python.keras.models import load_model
from load_data import load_testing_data, get_labels
from clean_data import threshold, remove_dots
import scipy.misc, scipy.ndimage
import numpy as np
import sys

print("Loading model...")
model = load_model('data/temp_model.hdf5')
print("Loaded model!")

print("Loading test data...")
x_test = load_testing_data()
print("Removing background...")
x_test = threshold(x_test)
print("Removing dots...")
x_test = remove_dots(x_test)
x_test = x_test.reshape(-1, 64, 64, 1)

print("Printing data...")
predictions = model.predict(x_test[0:100])

labels, _, _ = get_labels()
labels = list(labels)

to_show = int(sys.argv[1])
print(np.argmax(predictions[to_show]))
scipy.misc.imshow(scipy.ndimage.zoom(x_test[to_show].reshape(64,64), 16, order=0))
