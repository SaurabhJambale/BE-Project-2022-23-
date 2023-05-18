import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

DIRECTORY=r"dataset\train"
CATEGORIES=["brass","cartridge_brass","copper"]

data =[]
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        label = CATEGORIES.index(category)
        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        new_arr = cv2.resize(arr, (60, 60))
        data.append([new_arr, label])

import random
random.shuffle(data)
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

import pickle
pickle.dump(X, open('features.pkl', 'wb'))
pickle.dump(y, open('labels.pkl', 'wb'))
