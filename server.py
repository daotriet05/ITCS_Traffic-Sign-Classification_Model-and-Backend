from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import torch, torchvision
import numpy as np
import os
import random
import string
from implement import inference, load_model
import traceback
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app = Flask(__name__)
CORS(app)

# Load your TensorFlow model
net = load_model('stns.pth', 'traffic_classifier.pth', 'best.pt')
print("Model loaded successfully.")

# Define the image transformation
def transform_image(image):
    image = image.convert('RGB')  # Ensure image is in RGB mode
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def save_image(image_bytes, prefix):
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    filename = f"{prefix}_{random_suffix}.jpg"
    filepath = os.path.join('static', filename)
    with open(filepath, 'wb') as f:
        f.write(image_bytes.getbuffer())
    return filepath

@app.route('/', methods=['GET'])   
def home():
    return 'Server is running.'

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        #img_t = transform_image(img)

        # Perform inference
        buffer_sem, buffer_illu, pred_class = inference(net, img)
        print(f"Inference Results: {pred_class}")

        # Save feature images
        sem_image_path = save_image(buffer_sem, 'sem_feature')
        illu_image_path = save_image(buffer_illu, 'illu_feature')
        print(f"Saved feature images: {sem_image_path}, {illu_image_path}")

        return jsonify({
            'pred_class': int(pred_class),
            'sem_image_url': f'http://21.224.75.221:5000/{sem_image_path}',
            'illu_image_url': f'http://21.224.75.221:5000/{illu_image_path}'
        })

    except Exception as e:
        print(f"Error in classify_image: {e}")
        traceback.print_exc()  # Print the full traceback for debugging
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host='0.0.0.0', port=5000, debug=True)
