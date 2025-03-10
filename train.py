import os # Handles file paths and directory creation
import pandas as pd # Reads and processes the dataset (captions file)
import nltk # Natural Language Toolkit, used for tokenizing captions
import string # Used to remove punctuation from text
import tensorflow as tf # Main framework for deep learning and neural networks
import numpy as np # Handles numerical operations
import pickle # Saves and loads serialized data (image features, tokenizer)
from tensorflow.keras.applications import ResNet50 # Pretrained ResNet50 model for image feature extraction
from tensorflow.keras.applications.resnet50 import preprocess_input # Normalizes image data for ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer # Converts text captions into sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences # Ensures all captions have the same length
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate # Layers used in the caption generation model.

# Download NLTK tokenizer 
nltk.download("punkt")

# Define the locations of images and captions, dataset paths
IMAGE_FOLDER = r"C:\Users\Dell\Desktop\CODSOFT_Image_Captioning\data\Flickr8k_Dataset\Flicker8k_Dataset"
CAPTION_FILE = r"C:\Users\Dell\Desktop\CODSOFT_Image_Captioning\data\Flickr8k_text\Flickr8k.token.txt"

# Ensure directories exist
os.makedirs("data", exist_ok=True) # Creates data/ for storing processed captions and extracted image features
os.makedirs("models", exist_ok=True) # Creates models/ for saving the trained model and tokenizer

# Load captions into a DataFrame
# Reads Flickr8k.token.txt, which contains image filenames and their captions
# Removes index numbers from captions (since each image has multiple captions)
df = pd.read_csv(CAPTION_FILE, sep="\t", header=None, names=["image", "caption"])
df["image"] = df["image"].apply(lambda x: x.split("#")[0])  

# Function to clean captions
def clean_caption(text):
    text = text.lower() # Converts text to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation)) # Removes punctuation for consistency
    words = nltk.word_tokenize(text) # Tokenizes the captions into words
    return " ".join(words)

# Saves cleaned captions into cleaned_captions.csv
df["caption"] = df["caption"].apply(clean_caption)
df.to_csv("data/cleaned_captions.csv", index=False)
print("\nCaptions preprocessed and saved!")

# Loads ResNet50, a pre-trained CNN model that extracts features from images
# Uses include_top=False to remove the final classification layer
# Uses average pooling to get a 2048-dimensional feature vector
model_resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg") 

# Extract features
image_feature_dict = {}

for img_filename in df["image"].unique():
    img_path = os.path.join(IMAGE_FOLDER, img_filename)
    
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(224, 224))# Loads each image and resizes to 224x224 pixels
        img_array = img_to_array(img) # Converts it to a NumPy array and normalizes it
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        #Extracts 2048 feature values for each image using ResNet50
        features = model_resnet.predict(img_array)
        
        image_feature_dict[img_filename] = features.flatten() # Saves the extracted features in a dictionary
    else:
        print(f"Image not found: {img_path}")

print(f"Extracted Features: {len(image_feature_dict)} images")

# Saves extracted image features into a pickle file
with open("data/image_features.pkl", "wb") as f:
    pickle.dump(image_feature_dict, f)
print(f"Image features saved! Total images: {len(image_feature_dict)}")

# Tokenizes captions converting words into numerical indices
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["caption"].tolist())

# Saves the tokenizer for later use
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Finds the longest caption length for padding sequences
max_length = max(len(seq) for seq in tokenizer.texts_to_sequences(df["caption"].tolist()))

# Prepare Training Data
X_images, X_text, Y_text = [], [], []

# Creates input-output pairs for training
for i, row in df.iterrows():
    img_name = row["image"]
    if img_name in image_feature_dict:  # Ensure image feature exists
        caption_seq = tokenizer.texts_to_sequences([row["caption"]])[0]
        # Splits captions into sequences to allow incremental learning
        for j in range(1, len(caption_seq)):
            X_images.append(image_feature_dict[img_name])
            X_text.append(caption_seq[:j])
            Y_text.append(caption_seq[j])
# Ensures all caption sequences have the same length
X_text = pad_sequences(X_text, maxlen=max_length, padding="post")
Y_text = np.array(Y_text)

print(f"\nTraining Data Prepared: {len(X_images)} samples")

# Define Vocabulary Size
VOCAB_SIZE = len(tokenizer.word_index) + 1  

# Define Model Architecture
image_input = Input(shape=(2048,), name="image_input")
caption_input = Input(shape=(max_length,), name="caption_input")
embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=256, mask_zero=True)(caption_input)
lstm_layer = LSTM(256, return_sequences=True)(embedding)
dropout = Dropout(0.2)(lstm_layer)
lstm_layer_2 = LSTM(256)(dropout)
merged = Concatenate()([image_input, lstm_layer_2])
output = Dense(VOCAB_SIZE, activation="softmax")(merged)
# Image Input= 2048 feature vector.
# Caption Input= Encoded caption sequence.
# Embedding Layer= Converts word indices into dense vectors.
# LSTM Layers= Generate captions based on previous words.
# Dropout= Reduces overfitting.
# Concatenation= Merges image features with LSTM output.
# Dense Layer= Predicts the next word.

# Define Model
model = Model(inputs=[image_input, caption_input], outputs=output)

# Used Categorical Cross-Entropy Loss for multi-class classification
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

# Print Model Summary
model.summary()

# Training the model for 10 epochs with batch size 32
EPOCHS = 10
BATCH_SIZE = 32

print("\nDebugging Shapes:")
print(f"X_images shape: {np.array(X_images).shape}")
print(f"X_text shape: {np.array(X_text).shape}")
print(f"Y_text shape: {np.array(Y_text).shape}")

model.fit(
    {"image_input": np.array(X_images), "caption_input": X_text},
    Y_text,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# Saves the model in .keras and .h5 formats for easy reloading
model.save("models/image_captioning_model.keras")  # Recommended Format
model.save("models/image_captioning_model.h5")  # .h5 for Compatibility

print("\nModel trained and saved in both `.keras` and `.h5` formats!")
