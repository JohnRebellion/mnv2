import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import sys
# Load the saved model
model = tf.keras.models.load_model('2010SeriesPHBill.keras')

import sys
import base64
from PIL import Image
from io import BytesIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def load_and_resize_image(
        target_size=(224, 224)):
    # Load image using PIL
    original_image = Image.open(sys.argv[1])

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
    img_array = preprocess_image()
    predictions = model.predict(img_array, verbose=0)
    
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    
    # Print the class index and corresponding label
    # print(f"Predicted class index: {predicted_class_index}")
    
    # You may need to map the class index to your class labels
    # For example, if you have a list of class labels, you can do:
    class_labels = ["Real: 100", "Real: 1000", "Real: 20", "Real: 200", "Real: 50", "Real: 500", "Fake: 100", "Fake: 1000", "Fake: 20", "Fake: 200", "Fake: 50","Fake: 500"]  # Replace with your actual class labels
    predicted_class_label = class_labels[predicted_class_index]
    percent = predictions[0][predicted_class_index]
    return predicted_class_label

result = predict_image()
print(result)