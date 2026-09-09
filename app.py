import os
import pickle
import numpy as np
import tensorflow as tf
import io
import csv

# Environment fix for Keras 3 / TensorFlow 2.16+ compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import custom_object_scope

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
WORKING_DIR = './models'
MODEL_PATH = os.path.join(WORKING_DIR, 'model.h5')
TOKENIZER_PATH = os.path.join(WORKING_DIR, 'tokenizer.pkl')
CAPTIONS_PATH = os.path.join('dataset', 'captions.txt')
MAX_LENGTH = 35

print("\n--- Initializing System ---")


def load_caption_lookup(path):
    caption_lookup = {}
    try:
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                image_name = row[0].strip()
                caption = row[1].strip()
                if image_name and caption and image_name not in caption_lookup:
                    caption_lookup[image_name] = caption
        print(f"Loaded {len(caption_lookup)} captions from dataset.")
    except Exception as e:
        print("Caption lookup loading failed:", e)
    return caption_lookup

# ---------------------------------------------------
# Load Tokenizer
# ---------------------------------------------------
try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    print("❌ Tokenizer loading failed:", e)
    tokenizer = None

caption_lookup = load_caption_lookup(CAPTIONS_PATH)

# ---------------------------------------------------
# Load Caption Model (with fix for 'NotEqual' error)
# ---------------------------------------------------
try:
    # Adding custom_object_scope handles layers that Keras 3 doesn't recognize from Keras 2 files
    with custom_object_scope({'NotEqual': tf.math.not_equal}):
        caption_model = load_model(MODEL_PATH, compile=False)
    print("✅ Caption model loaded successfully.")
except Exception as e:
    print("❌ Model loading failed:", e)
    caption_model = None

# ---------------------------------------------------
# Load VGG16 Feature Extractor
# ---------------------------------------------------
try:
    vgg = VGG16(weights='imagenet')
    feature_extractor = Model(inputs=vgg.input, outputs=vgg.layers[-2].output)
    print("✅ VGG16 feature extractor ready.")
except Exception as e:
    print("❌ VGG16 loading failed:", e)
    feature_extractor = None

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([image, sequence], verbose=0)
        yhat_idx = np.argmax(yhat)
        word = idx_to_word(yhat_idx, tokenizer)

        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    
    return in_text.replace('startseq', '').strip()

# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == "admin" and password == "admin":
        return redirect(url_for('main_page'))
    return render_template('login.html', error="Invalid credentials")

@app.route('/main.html')
def main_page():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        filename = (file.filename or '').strip()

        if not filename:
            return jsonify({'error': 'No image selected'}), 400

        if filename in caption_lookup:
            return jsonify({
                'filename_match': filename,
                'caption': caption_lookup[filename],
                'source': 'dataset'
            })

        if caption_model is None or tokenizer is None or feature_extractor is None:
            return jsonify({
                'error': f"No saved caption found for '{filename}', and the AI model is not initialized."
            }), 500

        img_bytes = file.read()
        img = load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        feature = feature_extractor.predict(img, verbose=0)
        ai_caption = predict_caption(caption_model, feature, tokenizer, MAX_LENGTH)

        return jsonify({
            'filename_match': filename,
            'caption': ai_caption,
            'source': 'model'
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
