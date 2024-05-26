from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas

# Load the model using TFSMLayer
model_path = 'model/model.frutas'  # Adjust the path if necessary
model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# Function to preprocess the image
def preprocess_image(image):
    newsize = (180, 180)
    image = image.resize(newsize)
    np_image = np.array(image)
    img_array = np.expand_dims(np_image, 0)  # Add batch dimension
    return img_array

def get_description(predicted_class, score):
    if predicted_class == 0:
        return f"This image is {100 * score:.2f}% Mango"
    elif predicted_class == 2:
        return f"This image is {100 * score:.2f}% Banano"
    elif predicted_class == 3:
        return f"This image is {100 * score:.2f}% Limon"
    elif predicted_class == 4:
        return f"This image is {100 * score:.2f}% Maracuya"
    elif predicted_class == 5:
        return f"This image is {100 * score:.2f}% Naranja"
    elif predicted_class == 6:
        return f"This image is {100 * score:.2f}% Papaya"
    else:
        return f"This image is {100 * score:.2f}% Platano"

def get_fruit_name(predicted_class):
    if predicted_class == 0:
        return f"Mango"
    elif predicted_class == 2:
        return f"Banano"
    elif predicted_class == 3:
        return f"Limon"
    elif predicted_class == 4:
        return f"Maracuya"
    elif predicted_class == 5:
        return f"Naranja"
    elif predicted_class == 6:
        return f"Papaya"
    else:
        return f"Platano"

# Endpoint to handle predictions via POST
@app.route('/model', methods=['POST'])
def predict_post():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(image)
        
        predictions = model(img_array)
        
        # Debugging: print the keys in the predictions dictionary
        print(predictions.keys())
        
        # Ensure to use the correct key for accessing predictions
        predictions = predictions[next(iter(predictions))].numpy()  # Use the first key
        
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        score = float(keras.backend.sigmoid(predictions[0][0]))
        
        description = get_description(predicted_class, score)
        fruit_name = get_fruit_name(predicted_class)
        
        return jsonify({'predicted_class': predicted_class, 'description': description, 'fruit_name': fruit_name}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


        
# Endpoint to handle predictions via GET with base64 image
@app.route('/model_base64', methods=['GET'])
def predict_get():
    image_base64 = request.args.get('image_base64')
    
    if not image_base64:
        return jsonify({'error': 'No image data in the request'}), 400
    
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        img_array = preprocess_image(image)
        
        predictions = model(img_array)
        
        # Debugging: print the keys in the predictions dictionary
        print(predictions.keys())
        
        # Ensure to use the correct key for accessing predictions
        predictions = predictions[next(iter(predictions))].numpy()  # Use the first key
        
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        score = float(keras.backend.sigmoid(predictions[0][0]))
        
        description = get_description(predicted_class, score)
        fruit_name = get_fruit_name(predicted_class)
        
        return jsonify({'predicted_class': predicted_class, 'description': description, 'fruit_name': fruit_name}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)