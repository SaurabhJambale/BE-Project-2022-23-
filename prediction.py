import cv2
import keras
import matplotlib.pyplot as plt

import numpy as np
CATEGORIES = ['brass','cartridge brass', 'copper']


def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (60, 60))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 60, 60, 1)
    return new_arr

model = keras.models.load_model(r'C:\Users\HARSHAD\Desktop\BE project 2023\BE_Project_2023\3x3x64-brass vs cartridge_brass vs copper prediction.model')


prediction = model.predict([image(r"C:\Users\HARSHAD\Desktop\BE project 2023\BE_Project_2023\dataset\test\brass\b (1501).jpg")])
print(CATEGORIES[prediction.argmax()])
print(prediction)


