import cv2
import numpy as np

# Load the image
img = cv2.imread(r"C:\Users\HARSHAD\Desktop\BE PROJECT NEW\dataset\test\cartridge_brass\cb (1502).jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Perform morphological operations to remove small holes and smooth the edges
kernel = np.ones((3,3), np.uint8)
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
print("Average grain size = {:.2f} micrometers".format(avg_grain_size))
print("Standard deviation = {:.2f} micrometers".format(std_dev))
