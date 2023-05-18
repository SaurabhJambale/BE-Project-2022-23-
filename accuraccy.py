import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os

# Load the trained CNN model
model = load_model(r'C:\Users\HARSHAD\Desktop\BE PROJECT NEW\3x3x64-brass vs cartridge_brass vs copper prediction.model')

# Load and preprocess the test data
test_data = []  # List to store the preprocessed test images
test_labels = []  # List to store the corresponding test labels

# Assuming you have a directory of test images, you can loop over the images
test_image_directory = r'C:\Users\HARSHAD\Desktop\BE PROJECT NEW\dataset\test\testdata'
for image_name in os.listdir(test_image_directory):
    image_path = os.path.join(test_image_directory, image_name)
    img = image.load_img(image_path, target_size=(224, 224))  # Resize the image if necessary
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Preprocess the input image based on the requirements of your CNN model
    test_data.append(img)
    # Assuming the label is encoded in the filename or directory structure
    label = get_label_from_filename(image_name)
    test_labels.append(label)

# Convert the test data and labels to NumPy arrays
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Perform prediction on the test data
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_labels == test_labels)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))
