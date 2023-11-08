import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import clasify as c

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check and create the upload folder if it doesn't exist
def create_upload_folder():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    create_upload_folder()
    if 'image' in request.files:
        image = request.files['image']
        if image.filename != '':
            # to clear the contents in the uploads folder
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Save the uploaded image to the uploads folder
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
            return c.classify()
    return 'Image upload failed.'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
