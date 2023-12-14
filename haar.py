import cv2
import sys

base = '/home/johnn/mnv2'
# Load the pre-trained Haar cascade for object detection
haar_object_cascade = cv2.CascadeClassifier(base + '/' + 'haar_object.xml')
print(haar_object_cascade)
# Function to detect objects in an image using Haar cascade
def detect_haar_objects(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale (required for object detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect objects in the image
    haar_objects = haar_object_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    print(len(haar_objects) > 0)

# Replace 'path/to/your/image.jpg' with the path to the image you want to detect objects in
image_path = sys.argv[1]
detect_haar_objects(image_path)
