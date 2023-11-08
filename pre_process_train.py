from PIL import Image
import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


def preprocess_images(input_directory, output_directory, target_size=(224, 224), output_format='JPEG'):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for class_folder in os.listdir(input_directory):
        class_folder_path = os.path.join(input_directory, class_folder)
        if os.path.isdir(class_folder_path):
            # Create a corresponding sub-folder in the output directory
            output_class_folder = os.path.join(output_directory, class_folder)
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)

            for filename in os.listdir(class_folder_path):
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    # Open the image using Pillow (PIL)
                    image_path = os.path.join(class_folder_path, filename)
                    image = Image.open(image_path)

                    # Resize the image to the target size
                    image = image.resize(target_size)

                    # Normalize pixel values to the range [0, 1]
                    image = np.array(image) / 255.0

                    # Save the preprocessed image to the corresponding output sub-folder
                    output_path = os.path.join(output_class_folder, filename)
                    image = Image.fromarray((image * 255).astype('uint8'))  # Convert back to uint8 before saving
                    image.save(output_path, format=output_format)
    main_data_directory = output_directory

    # Set batch size, image dimensions, and the number of classes
    batch_size = 32
    image_height, image_width = 224, 224
    num_classes = 2  # Replace with the actual number of classes

    # Create an ImageDataGenerator for data preprocessing
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2  # Split the data into training and validation
    )

    # Create data generators for training and validation data
    train_generator = datagen.flow_from_directory(
        main_data_directory,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',  # For multiclass classification
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        main_data_directory,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',  # For multiclass classification
        subset='validation'
    )


