from flask import Flask
from flask import *
import os
import cv2
import keras

import numpy as np


app = Flask(__name__)

UPLOAD_FOLDER=r"C:\Users\HARSHAD\Desktop\BE PROJECT NEW\static"






@app.route('/', methods = ['GET','POST'])
def upload_predict():

    if request.method == 'POST':
        image_file= request.files['image']
        if image_file:
            image_location=os.path.join(UPLOAD_FOLDER,image_file.filename)
            image_file.save(image_location)

            CATEGORIES = ['brass', 'cartridge brass', 'copper']

            def image(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(img, (60, 60))
                new_arr = np.array(new_arr)
                new_arr = new_arr.reshape(-1, 60, 60, 1)
                return new_arr

            model = keras.models.load_model(
                r'C:\Users\HARSHAD\Desktop\BE project 2023\BE_Project_2023\3x3x64-brass vs cartridge_brass vs copper prediction.model')

            prediction = model.predict([image(image_location)])
            predicted_value = CATEGORIES[prediction.argmax()]

            # Load the image
            img = cv2.imread(image_location)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Threshold the image
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Perform morphological operations to remove small holes and smooth the edges
            kernel = np.ones((3, 3), np.uint8)
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find contours of the grains
            contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate the area and perimeter of each contour
            areas = [cv2.contourArea(c) for c in contours]
            perimeters = [cv2.arcLength(c, True) for c in contours]

            # Calculate the equivalent diameter of each grain
            equivalent_diameters = [np.sqrt(4 * a / np.pi) for a in areas]

            # Calculate the average grain size and standard deviation
            avg_grain_size = np.mean(equivalent_diameters)
            std_dev = np.std(equivalent_diameters)

            # Print the results
            grain_size="Average grain size = {:.2f} micrometers".format(avg_grain_size)


            return render_template("index.html", prediction=predicted_value, image_loc=image_file.filename,grain_size=grain_size)
    return render_template("index.html",prediction=0, image_loc=None, grain_size=0)




if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')