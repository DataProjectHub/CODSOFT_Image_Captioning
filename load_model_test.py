import tensorflow as tf # TensorFlow is used to load the trained deep learning model
from tensorflow.keras.utils import custom_object_scope # Ensures that any custom layers used in the model are properly recognized
from tensorflow.keras.layers import Layer # Represents a base class for all Keras layers 

# Define model paths
model_path_h5 = "models/image_captioning_model.h5" # The recommended format for TensorFlow models
model_path_keras = "models/image_captioning_model.keras" # Older format, used for compatibility

print("Loading trained model...")

try:
    # Load with custom object scope
    with custom_object_scope(custom_objects): # handle any custom layers or loss functions.
        # Prevents recompilation to speed up loading and avoid unnecessary dependencies.
        model = tf.keras.models.load_model(model_path_keras, compile=False)
    print(f"Model loaded successfully from {model_path_keras}")
except:
    try:
        # Try loading the `.h5` model if `.keras` fails
        # ensures that any custom layers or objects used in training are recognized.
        with custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path_h5, compile=False)
        print(f"Model loaded successfully from {model_path_h5}")
    except Exception as e:
        print(f"Model loading failed! Error: {e}")
