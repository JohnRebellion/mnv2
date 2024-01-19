import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import sys

import io

HAAR_ENABLED = True

# Load the saved model
model = tf.keras.models.load_model('2010SeriesPHBill.keras')

from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2

base = '/home/johnn/mnv2'
# Load the pre-trained Haar cascade for object detection
haar_object_cascade = cv2.CascadeClassifier(base + '/' + 'haar_object.xml')

from roboflow import Roboflow
rf = Roboflow(api_key="Hs8lYsJRAdg8JPBQS9ni")
project = rf.workspace("university-of-the-eat").project("capstone-kdvok")
rmodel = project.version(3).model

# Function to detect objects in an image using Haar cascade
def detect_haar_objects():
    # Read the image
    img = cv2.imread(sys.argv[1])
    
    # Convert the image to grayscale (required for object detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect objects in the image
    haar_objects = haar_object_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return (len(haar_objects) == 0)


def load_and_resize_image(
        target_size=(224, 224)):
    # Load image using PIL
    original_image = Image.open("image.jpg")

    # Resize the image to the target size
    resized_image = original_image.resize(target_size)

    return resized_image

# Function to preprocess an image
def preprocess_image():
    img = load_and_resize_image()
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions on a given image
def predict_image():
    # if HAAR_ENABLED and detect_haar_objects(): print('proceed')
    img_array = preprocess_image()
    # rs = rmodel.predict("image.jpg", confidence=40, overlap=30)
    #if len(rs) == 0:
        #return "Please try again"
    # if len(rs)>0:
    #     return "Real" #rs[0]['class'].rstrip('_')
    predictions = model.predict(img_array, verbose=0)
    
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    
    # Print the class index and corresponding label
    # print(f"Predicted class index: {predicted_class_index}")
    
    # You may need to map the class index to your class labels
    # For example, if you have a list of class labels, you can do:
    # class_labels = ["Real: 100", "Real: 1000", "Real: 20", "Real: 200", "Real: 50", "Real: 500", "Fake: 100", "Fake: 1000", "Fake: 20", "Fake: 200", "Fake: 50","Fake: 500"]  # Replace with your actual class labels
    # predicted_class_label = class_labels[predicted_class_index]
    predicted_class_label = "Real" if predicted_class_index < 6 else "Fake"
    percent = predictions[0][predicted_class_index]
    return predicted_class_label

# result = predict_image()
# print(result)
