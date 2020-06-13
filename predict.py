import argparse
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

labels = [
    '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
    'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
    ]

# Define constants
TARGET_WIDTH = 128
TARGET_HEIGHT = 128
MODEL_PATH = './my_model'

# Load the image
image = cv2.imread(args["image"])
# Preprocessing the image
image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Load the trained convolutional neural network
print("[INFO] Loading my model...")
model = load_model(MODEL_PATH, compile=False)

# Classify the input image then find the index of the class with the *largest* probability
print("[INFO] Classifying image...")
prob = model.predict(image)[0]
idx = np.argsort(prob)[-1]
print(labels[idx])