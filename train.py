import pre_process_train as pp
from tensorflow import keras

pp.preprocess_images('train_data','processed_train_data')
# Load your preprocessed training data
train_data_directory = 'processed_train_data'

# Define the model architecture for binary classification
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output layer with one unit for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary cross-entropy for binary classification
              metrics=['accuracy'])

# Data augmentation (optional)
data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomRotation(0.2),
    keras.layers.experimental.preprocessing.RandomZoom(0.2),
])

# Create data generators for binary classification
batch_size = 32
image_height, image_width = 224, 224

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    data_augmentation,
    rescale=1.0 / 255.0  # Normalize pixel values to [0, 1]
    
)

train_generator = train_datagen.flow_from_directory(
    train_data_directory,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary',  # For binary classification
)

# Train the binary classification model
epochs = 10  # You can adjust the number of epochs
history = model.fit(train_generator, epochs=epochs)

# Optional: Save the trained model
model.save('binary_classification_model.h5')


