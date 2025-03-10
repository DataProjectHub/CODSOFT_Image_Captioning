
import os # Handles file and directory operations
import pickle # Saves and loads extracted image features
import numpy as np # Handles numerical operations, especially for image arrays
from tensorflow.keras.applications import ResNet50 # Imports the pre-trained ResNet50 model for feature extraction
from tensorflow.keras.applications.resnet50 import preprocess_input # Normalizes image pixel values for ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array # Loads images from disk, Converts images to NumPy arrays

# Loads ResNet50, a pre-trained CNN on the ImageNet dataset
print("Loading ResNet50 Model...")
# Uses pre-trained weights from ImageNet
# Removes the fully connected (classification) layers, so we can extract features instead of labels.
# Uses global average pooling to get a 2048-dimensional feature vector for each image.
model_resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg") 
print("Model Loaded!")

# Defines the path to the folder containing Flickr8k images.
IMAGE_FOLDER = r"C:\Users\Dell\Desktop\CODSOFT_Image_Captioning\data\Flickr8k_Dataset\Flicker8k_Dataset"

# Uses os.listdir() to list all image files inside Flickr8k_Dataset
image_filenames = os.listdir(IMAGE_FOLDER)
print(f"Found {len(image_filenames)} images in dataset.")

# Dictionary to Store Extracted Features
image_feature_dict = {}

# Loops through each image file in the dataset and creates the full image path
for idx, img_filename in enumerate(image_filenames):
    img_path = os.path.join(IMAGE_FOLDER, img_filename)
    
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(224, 224))# Loads the image and resizes it to 224x224 pixels (required for ResNet50)
        img_array = img_to_array(img) 
        img_array = np.expand_dims(img_array, axis=0)# Converts the image to a NumPy array
        img_array = preprocess_input(img_array)# Expands dimensions to match the expected input shape (1, 224, 224, 3)
        features = model_resnet.predict(img_array) # Passes the processed image through ResNet50 to extract features
        # Flattens the extracted features into a 1D vector and stores them in image_feature_dict with the image filename as the key
        image_feature_dict[img_filename] = features.flatten()
        
        # Prints a progress update every 500 images
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(image_filenames)} images...")
    else:
        print(f"Image Not Found: {img_path}")

# Save Extracted Features
os.makedirs("data", exist_ok=True)  # Ensure 'data' folder exists
# Saves image feature vectors as a pickle file (.pkl) for quick access in later stages.
with open("data/image_features.pkl", "wb") as f:
    pickle.dump(image_feature_dict, f)

print(f"\nImage Features Saved! Total Processed Images: {len(image_feature_dict)}")
