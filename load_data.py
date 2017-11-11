'''
Loads training data, presumed to be in ./data
'''

import numpy as np
from pathlib import Path
from collections import defaultdict

def load_data():
    training_data = Path("data/training_data.npz")
    
    if training_data.is_file():
        with np.load("data/training_data.npz") as data:
            x = data['x']
            y = data['y']
    else:
        x = np.genfromtxt('data/train_x.csv', delimiter=",")
        y = np.genfromtxt('data/train_y.csv', dtype=int)
        np.savez("data/training_data.npz", x=x, y=y)

    return x, y

def get_labels():
    numbers = set()
    solutions = defaultdict(int)
    for i in range(0,10):
        for j in range(0,10):
            numbers.add(i+j)
            numbers.add(i*j)
            solutions[i+j] += 1
            solutions[i*j] += 1
    uniques = {k:v in solutions for (k, v) in solutions.items() if v == 2}

    mapping = [-1]*82
    for i, j in enumerate(numbers):
       mapping[j] = i 

    return numbers, mapping, uniques
