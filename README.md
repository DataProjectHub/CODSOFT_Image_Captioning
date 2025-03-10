# Image Captioning using Deep Learning

## Introduction
This project implements an **Image Captioning** model using **Deep Learning** techniques. The model takes an image as input and generates a meaningful caption describing the image. This was done as part of my internship in Machine Learning.

## Project Overview
The project was built from scratch, starting from dataset collection, preprocessing, feature extraction, model training, and finally testing and evaluation. The entire code was developed in VS Code and executed in Google Colab for efficient GPU usage.

## Dataset Used
The dataset used is **Flickr8k**, which contains **8,000 images** with **five captions** per image. I downloaded the Flickr8k dataset from Kaggle, a popular platform for datasets and machine learning competitions. The dataset consists of:
- Flickr8k_Dataset (Images)
- Flickr8k_text (Captions file)

## Files Created in VS Code##

During the project, I created and managed multiple files in VS Code to ensure proper structuring of the project:

train.py → Contains the model training code.
load_model_test.py → Loads the trained model and tests it on new images.
extract_features.py → Extracts image features using a pretrained CNN.
requirements.txt → Contains all dependencies required to run the project.
README.md → This file, which documents the project.
data/ → Stores the Flickr8k dataset and extracted features.
models/ → Stores the trained model files (.keras and .h5).
scripts/ → Stores helper scripts used during development.

## Steps to Build the Model
### 1. Preprocessing the Dataset
- Loaded image captions from the text file.
- Performed text preprocessing: Lowercasing, removing punctuation, and tokenizing.
- Created a vocabulary and mapped each word to an index.

### 2. Feature Extraction from Images
- Used a **Pretrained CNN (InceptionV3)** to extract image features.
- Saved the extracted features in a pickle file for faster processing.

### 3. Preparing the Data for Training
- Used Tokenizer to encode captions into sequences.
- Applied Padding and created input-output sequences.

### 4. Building the Model
The model architecture consists of:
- **CNN (InceptionV3):** Extracts image features.
- **LSTM (Long Short-Term Memory):** Generates captions based on extracted features.
- **Embedding Layer:** Converts words into dense vectors.

### 5. Training the Model
- Used categorical cross-entropy loss and Adam optimizer.
- Trained for multiple epochs, monitoring loss improvement.
- Saved the model in both '.keras' and '.h5' formats.

### 6. Testing the Model
- Loaded the trained model from saved files.
- Given an input image, generated a caption using Beam Search.
- Compared predicted captions with ground-truth captions.

## Running the Project
### Setup
1. Clone the repository:
   In powershell
   git clone https://github.com/DataProjectHub/CODSOFT_Image_Captioning.git
   cd CODSOFT_Image_Captioning
   
2. Install dependencies:
   In powershell
   pip install -r requirements.txt
   
3. Extract image features:
   In powershell
   python extract_features.py
   
4. Train the model:
   In powershell
   python train.py
   
5. Test the model:
   In powershell
   python load_model_test.py --image "sample.jpg"
   

## Challenges Faced
- Handling large dataset sizes on local machines.
- Fine-tuning the LSTM model for better caption generation.
- Experimenting with different hyperparameters for better accuracy.

## Future Enhancements
- Implement Transformer-based models like BLIP or ViT-GPT.
- Use larger datasets for more diverse captions.
- Deploy the model as a web application.

## Final Thoughts
This project was a great learning experience in Computer Vision and Natural Language Processing (NLP). It helped in understanding how to integrate CNNs and LSTMs for sequential data processing. Looking forward to improving this model further!

