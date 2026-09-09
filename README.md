# 🖼️ Image Caption Generator

An **AI-powered Image Caption Generator** that automatically generates a meaningful natural-language description for an image using **Deep Learning**.

This project uses a **pre-trained VGG16 model** to extract visual features from images and an **LSTM-based neural network** to generate captions. The trained model is integrated into a **Flask web application**, allowing users to upload an image and receive a generated caption.

---

## 📌 Project Overview

Image Captioning is a task that combines **Computer Vision** and **Natural Language Processing (NLP)**.

The goal of this project is to take an image as input and automatically generate a textual description of its contents.

For example:

```text
Input:
    🖼️ Image of a dog playing with a ball

Output:
    "A dog is playing with a ball"
```

The project follows an **Encoder-Decoder architecture**:

* **VGG16** acts as the image feature extractor (Encoder).
* **LSTM** acts as the caption generator (Decoder).
* **Flask** provides the web interface and prediction API.

---

## ✨ Features

* 🖼️ Upload images through a web interface
* 🧠 Extract image features using pre-trained **VGG16**
* ✍️ Generate captions using an **LSTM-based Deep Learning model**
* 🔤 Caption tokenization and sequence processing
* 🌐 Flask-based web application
* 🔄 Image caption prediction through a Flask API
* 📁 Dataset caption lookup for known images
* 🤖 AI-generated captions for images not found in the dataset
* 🔐 Simple login system
* 📦 Saved model and tokenizer for inference

---

## 📊 Dataset

This project uses the **Flickr8k Dataset**.

The Flickr8k dataset contains images paired with natural-language captions and is commonly used for experimenting with image captioning systems.

### Dataset Source

**Kaggle – Flickr8k Dataset:**

https://www.kaggle.com/datasets/adityajn105/flickr8k

The dataset is used for:

* Image feature extraction
* Caption preprocessing
* Tokenization
* Training the caption-generation model
* Testing the image captioning pipeline

### Dataset Structure

The relevant dataset structure is:

```text
dataset/
│
├── Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
│
└── captions.txt
```

The Flask application loads caption information from:

```text
dataset/captions.txt
```

The application creates a lookup between image filenames and captions.

---

# 🏗️ System Architecture

```text
                         IMAGE
                           │
                           ▼
                 ┌──────────────────┐
                 │      VGG16       │
                 │ Feature Extractor│
                 └────────┬─────────┘
                          │
                          ▼
                  Image Feature Vector
                          │
                          ▼
              ┌─────────────────────────┐
              │    Caption Generator    │
              │                         │
              │  Embedding → LSTM →     │
              │       Dense Layer       │
              └───────────┬─────────────┘
                          │
                          ▼
                  Generated Caption
                          │
                          ▼
                   Flask Web App
                          │
                          ▼
                      User Output
```

---

# 🧠 How the Project Works

The complete image captioning pipeline consists of several steps.

## 1. Image Input

The user uploads an image through the Flask web application.

The `/predict` endpoint receives the uploaded image:

```text
POST /predict
```

The application first checks whether an image was actually uploaded.

---

## 2. Image Preprocessing

The uploaded image is:

1. Read by the application
2. Resized to **224 × 224**
3. Converted into an array
4. Expanded to create the required batch dimension
5. Preprocessed using VGG16 preprocessing

The application performs this preprocessing before extracting image features.

---

## 3. Feature Extraction Using VGG16

The project uses **VGG16 pretrained on ImageNet** as the image feature extractor.

Instead of using the final classification layer, the output of the layer before the final classification layer is used as the image feature representation.

```python
vgg = VGG16(weights='imagenet')

feature_extractor = Model(
    inputs=vgg.input,
    outputs=vgg.layers[-2].output
)
```

This allows the system to convert an image into a numerical feature representation that can be passed to the caption-generation model.

---

# 📝 Caption Processing

The captions associated with the Flickr8k images are processed before being used for training.

Special tokens are used to indicate the beginning and end of a caption.

### Start Token

```text
startseq
```

### End Token

```text
endseq
```

For example:

```text
A dog is running in the park
```

is converted conceptually into:

```text
startseq a dog is running in the park endseq
```

These tokens help the model understand where caption generation should start and stop.

---

# 🔤 Tokenization

Natural-language captions cannot be directly provided to a neural network.

Therefore, the words are converted into numerical representations using a tokenizer.

For example:

```text
a       → numerical ID
dog     → numerical ID
running → numerical ID
park    → numerical ID
```

The trained tokenizer is saved as:

```text
models/tokenizer.pkl
```

The Flask application loads this tokenizer during initialization.

---

# 🧩 LSTM Caption Generator

The project uses an **LSTM (Long Short-Term Memory)** network to generate captions.

LSTM is useful for sequence-generation tasks because it can learn relationships between words in a sequence.

The model receives:

### Input 1 — Image Features

The feature vector generated by VGG16.

### Input 2 — Caption Sequence

The sequence of words generated so far.

The model then predicts the probability of the **next word**.

Conceptually:

```text
Image Features
      +
Previous Words
      │
      ▼
     LSTM
      │
      ▼
Next Word
```

This process continues until the model predicts:

```text
endseq
```

or reaches the maximum caption length.

The prediction function initializes the sequence with `startseq` and repeatedly predicts the next word.

---

# 🔄 Caption Generation Process

Suppose the model receives an image of a dog.

The generation process can look like:

```text
startseq
    ↓
a
    ↓
a dog
    ↓
a dog is
    ↓
a dog is playing
    ↓
a dog is playing outside
    ↓
endseq
```

Final caption:

```text
A dog is playing outside
```

The application uses the tokenizer and trained model to convert each predicted word ID back into a word.

---

# 🌐 Flask Web Application

The trained model is integrated into a Flask web application.

The application initializes:

```text
Flask
Flask-CORS
VGG16
Caption Model
Tokenizer
Caption Dataset
```

The relevant Python libraries include:

```python
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
```

These components are initialized in `app.py`.

---

# 🔐 Login System

The application contains a basic login page before accessing the main image-captioning interface.

The current demo credentials are:

```text
Username: admin
Password: admin
```

The login route checks the submitted username and password and redirects the user to the main page when the credentials are correct.

> ⚠️ **Note:** This authentication mechanism is intended for demonstration/academic purposes and should not be used as-is in a production application.

---

# 🔍 Prediction Logic

When an image is uploaded, the application follows this process:

```text
                Upload Image
                     │
                     ▼
             Check Image Filename
                     │
              ┌──────┴──────┐
              │             │
            Found          Not Found
              │             │
              ▼             ▼
       Dataset Caption     VGG16
              │          Feature Extraction
              │             │
              │             ▼
              │       Caption Model
              │             │
              │             ▼
              │       Generated Caption
              │             │
              └──────┬──────┘
                     ▼
                JSON Response
```

If the uploaded filename exists in the caption lookup, the application can return the corresponding stored caption.

If the filename is not found, the image is processed using VGG16 and the trained caption model generates a caption.

---

# 📂 Project Structure

```text
Image-Capction-Generator/
│
├── app.py
├── Train_Model.ipynb
├── requirements.txt
│
├── dataset/
│   ├── Images/
│   └── captions.txt
│
├── models/
│   ├── model.h5
│   ├── tokenizer.pkl
│   └── features.pkl
│
├── templates/
│   ├── login.html
│   └── main.html
│
└── static/
    ├── css/
    ├── js/
    └── images/
```

> The exact structure may vary depending on which files are included in the repository.

---

# 🛠️ Technologies Used

| Technology              | Purpose                       |
| ----------------------- | ----------------------------- |
| **Python**              | Main programming language     |
| **TensorFlow**          | Deep Learning framework       |
| **Keras**               | Neural network implementation |
| **VGG16**               | Image feature extraction      |
| **LSTM**                | Caption generation            |
| **NumPy**               | Numerical operations          |
| **Flask**               | Backend web framework         |
| **Flask-CORS**          | Cross-Origin Resource Sharing |
| **HTML/CSS/JavaScript** | Frontend                      |
| **Pickle**              | Saving and loading tokenizer  |
| **Jupyter Notebook**    | Model training                |

The project's current requirements include TensorFlow, tqdm, Flask, and Flask-CORS.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/RohitDake78/Image-Capction-Generator.git
```

Move into the project directory:

```bash
cd Image-Capction-Generator
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project requires TensorFlow, Flask, Flask-CORS, and other Python dependencies listed in `requirements.txt`.

---

# ▶️ Running the Application

After installing the dependencies, run:

```bash
python app.py
```

The Flask application runs on:

```text
http://127.0.0.1:5000
```

Open the address in your browser.

You should see the login page.

---

# 🔑 Demo Login

Use the following credentials:

```text
Username: admin
Password: admin
```

After successful login, the main image captioning page will be displayed.

Upload an image to generate a caption.

---

# 🏋️ Training the Model

The project contains a Jupyter Notebook:

```text
Train_Model.ipynb
```

The notebook is used for the model-training pipeline.

The general training workflow is:

```text
Flickr8k Dataset
       │
       ▼
Load Images & Captions
       │
       ▼
Caption Preprocessing
       │
       ▼
Tokenization
       │
       ▼
VGG16 Feature Extraction
       │
       ▼
Create Training Sequences
       │
       ▼
Train LSTM Caption Model
       │
       ▼
Save Model + Tokenizer
```

After training, the saved model and tokenizer are used by `app.py` for prediction.

---

# 💾 Saved Model Files

The Flask application expects the following files:

```text
models/model.h5
models/tokenizer.pkl
```

The paths are configured in `app.py`:

```python
WORKING_DIR = './models'

MODEL_PATH = os.path.join(
    WORKING_DIR,
    'model.h5'
)

TOKENIZER_PATH = os.path.join(
    WORKING_DIR,
    'tokenizer.pkl'
)
```

The maximum caption sequence length used by the application is configured as:

```python
MAX_LENGTH = 35
```

---

# 📡 API

The application provides a prediction endpoint:

```text
POST /predict
```

### Request

Send an image using the form field:

```text
image
```

### Response

The API returns JSON containing the generated caption.

Example:

```json
{
    "filename_match": "example.jpg",
    "caption": "a dog is playing in the park",
    "source": "model"
}
```

The `source` field indicates whether the caption came from the dataset or the trained model.

---

# 🎯 Applications

Image caption generation can be useful in several areas:

### ♿ Accessibility

Automatically generated descriptions can help visually impaired users understand image content.

### 🔎 Image Search

Captions can provide additional text information for image indexing and search.

### 📱 Social Media

AI-generated descriptions can assist users in creating image descriptions.

### 🤖 AI Assistants

Image captioning can be integrated into multimodal AI assistants.

### 📚 Education

The technology can be used for educational tools that combine images and natural-language descriptions.

---

# 🚀 Future Improvements

The current project can be further improved by:

* Implementing **Attention Mechanisms**
* Experimenting with **Transformer-based image captioning**
* Using larger image-caption datasets
* Improving caption quality
* Implementing **Beam Search**
* Generating multiple caption alternatives
* Improving the frontend UI/UX
* Adding secure user authentication
* Adding caption history
* Allowing users to download generated captions
* Deploying the application to a cloud platform
* Adding model evaluation metrics such as **BLEU**

---

# ⚠️ Limitations

Some limitations of the current implementation include:

* Caption quality depends on the training dataset and trained model.
* The model may generate grammatically incorrect or incomplete captions.
* A simple username/password authentication system is currently used.
* The model may not correctly describe objects or scenes that are significantly different from the training data.
* VGG16 feature extraction and LSTM generation can require significant computational resources.

---

# 📚 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Computer Vision
* Natural Language Processing
* Deep Learning
* Convolutional Neural Networks
* VGG16
* LSTM Networks
* Image Feature Extraction
* Text Tokenization
* Sequence Generation
* Transfer Learning
* Flask Web Development
* REST API Development
* Model Deployment

---

# 👨‍💻 Author

## Rohit Dake

GitHub:
https://github.com/RohitDake78

---

# ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

# 📜 License

This project is developed for **educational and academic purposes**.

The dataset is obtained from the **Flickr8k dataset available on Kaggle**. Please refer to the dataset's original terms and licensing information before redistributing the dataset.

---

## 🙏 Acknowledgements

* **Flickr8k Dataset** for providing the image-caption data.
* **TensorFlow/Keras** for the Deep Learning framework.
* **VGG16** for pretrained image feature extraction.
* **Flask** for the web application framework.
