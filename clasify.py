from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pre_process_test as pp

def classify():

    # Load your trained model
    model = keras.models.load_model('binary_classification_model.h5')

    # Preprocess your test data (adjust this part to match your preprocessing)
    test_data = pp.preprocess_images_in_folder('uploads')
    # true_labels = load_true_labels('true_labels_file.txt')

    # Make predictions
    predictions = model.predict(test_data)

    # Threshold the predictions
    predicted_classes = (predictions >= 0.5).astype(int)

    if predicted_classes[0][0]== 1:
        return "Pen"
    else:
        return "Pencil"