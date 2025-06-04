import numpy as np
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Path to the input image
img_path = 'football.jpg'  # Change this to your image file

# Load and preprocess the image
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
preds = model.predict(x)

# Decode and display predictions
print('Predicted:', decode_predictions(preds, top=3)[0])
