from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pre_process_test as pp



def load_true_labels(labels_file):
    with open(labels_file, 'r') as file:
        true_labels = [int(line.strip()) for line in file]
    return true_labels

# Load your trained model
model = keras.models.load_model('binary_classification_model.h5')

# Preprocess your test data (adjust this part to match your preprocessing)
test_data = pp.preprocess_images_in_folder('test_data')
true_labels = load_true_labels('true_labels_file.txt')

# Make predictions
predictions = model.predict(test_data)

# Threshold the predictions
predicted_classes = (predictions >= 0.5).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_classes)
precision = precision_score(true_labels, predicted_classes)
recall = recall_score(true_labels, predicted_classes)
f1 = f1_score(true_labels, predicted_classes)

# Display or save the results
print(f'Test Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
