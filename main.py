import tensorflow as tf
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions


# Load MobileNetV2 model pre-trained on ImageNet data
# model = MobileNetV2(weights='imagenet')
# model = ResNet50V2(weights='imagenet')
model = tf.keras.models.load_model('2010SeriesPHBill.keras')

# Load and preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Make predictions on a given image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)
    
    # Print the top three predictions
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
        print(f"{i + 1}: {label} ({score:.2f})")

# Replace 'path/to/your/image.jpg' with the path to the image you want to classify
image_path = 'petmd-albino.jpg'
predict_image(image_path)
