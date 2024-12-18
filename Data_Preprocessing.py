import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

image_path ='/Users/vivekrajoriya/Desktop/Dev/all_intern_proj/new_Assign/OA_train_test/TRAIN'
# Preprocess image (resize, grayscale, normalize)
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)  # Resize image to 224x224
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = image / 255.0  # Normalize pixel values to range [0, 1]
    return image

# Set up data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Example: loading image and applying augmentation
image_path = 'path_to_image.jpg'
image = preprocess_image(image_path)
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Apply data augmentation
augmented_image = datagen.flow(image)
cv2.imwrite('augmented_image.jpg', augmented_image_output * 255)  # Save the image (multiply by 255 to restore range)
