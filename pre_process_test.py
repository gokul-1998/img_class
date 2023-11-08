from PIL import Image
import os
import numpy as np


def preprocess_images_in_folder(input_directory, target_size=(224, 224)):
    images = []  # A list to collect preprocessed images

    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # Open the image using Pillow (PIL)
            image_path = os.path.join(input_directory, filename)
            image = Image.open(image_path)

            # Resize the image to the target size
            image = image.resize(target_size)

            # Normalize pixel values to the range [0, 1]
            image = np.array(image) / 255.0

            images.append(image)  # Collect the preprocessed image

    return np.array(images)  # Convert the list of images to a NumPy array

