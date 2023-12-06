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

def decode_data_url(data_url):
    # Extract base64-encoded part
    _, encoded_data = data_url.split(',', 1)

    # Decode base64
    decoded_data = base64.b64decode(encoded_data)

    return decoded_data

def load_and_resize_image(decoded_data, target_size=(224, 224)):
    # Load image using PIL
    original_image = Image.open(BytesIO(decoded_data))

    # Resize the image to the target size
    resized_image = original_image.resize(target_size)

    return resized_image

# Function to preprocess an image
def preprocess_image(img_path):
    decoded_data = decode_data_url(img_path)
    img = load_and_resize_image(decoded_data)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions on a given image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array, verbose=0)
    
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    
    # Print the class index and corresponding label
    # print(f"Predicted class index: {predicted_class_index}")
    
    # You may need to map the class index to your class labels
    # For example, if you have a list of class labels, you can do:
    class_labels = ["Real: 100", "Real: 1000", "Real: 20", "Real: 200", "Real: 50", "Real: 500", "Fake: 100",  "Fake: 1000", "Fake: 20", "Fake: 200", "Fake: 50", "Fake: 500"]  # Replace with your actual class labels
    predicted_class_label = class_labels[predicted_class_index]
    percent = predictions[0][predicted_class_index]
    return predicted_class_label

    # if percent >= .9:
    #     print(f"Predicted class label: {predicted_class_label} - {percent:.2%}")
    # else:
    #     print("Please try again")

# Replace 'path/to/your/new_image.jpg' with the path to the image you want to predict
# new_image_path = '50pesosfnb.jpg'
# new_image_path = 'bente.jpg'
# new_image_path = 'fake20.jpg'
new_image_path = sys.argv[1]
result = predict_image(new_image_path)
print(result)
