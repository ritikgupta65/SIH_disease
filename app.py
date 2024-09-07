from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Update this line with the correct path to your model
model_path = './model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

# Load your trained model here
model = tf.keras.models.load_model(model_path)

# Dictionary mapping class indices to disease names
class_indices = {
    'Corn___Common_Rust': 0, 'Corn___Gray_Leaf_Spot': 1, 'Corn___Healthy': 2,
    'Corn___Northern_Leaf_Blight': 3, 'Potato___Early_Blight': 4, 'Potato___Healthy': 5,
    'Potato___Late_Blight': 6, 'Rice___Brown_Spot': 7, 'Rice___Healthy': 8,
    'Rice___Leaf_Blast': 9, 'Rice___Neck_Blast': 10, 'Sugarcane_Bacterial Blight': 11,
    'Sugarcane_Healthy': 12, 'Sugarcane_Red Rot': 13, 'Wheat___Brown_Rust': 14,
    'Wheat___Healthy': 15, 'Wheat___Yellow_Rust': 16
}

# Inverse dictionary to map indices to disease names
class_names = {v: k for k, v in class_indices.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']

    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Convert the uploaded file to an image
        image = Image.open(file)

        # Ensure image is in RGB mode (if your model expects RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize or preprocess the image as needed
        image = image.resize((150, 150))  # Example: resize to 150x150
        image = np.array(image) / 255.0   # Example: normalize pixel values

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Make a prediction using the loaded model
        prediction = model.predict(image)

        # Determine the predicted class index
        predicted_class_index = np.argmax(prediction[0])

        # Get the corresponding disease name
        disease_name = class_names.get(predicted_class_index, 'Unknown Class')

        # Return the prediction as a JSON response
        return jsonify({'prediction': disease_name})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
