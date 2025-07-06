import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess
from tensorflow.keras.models import Model
import numpy as np
import os # Added for dummy image creation in __main__

class FeatureExtractor:
    def __init__(self, model_name='vgg16'):
        self.model_name = model_name.lower()
        self.base_model = None
        self.preprocess_input_func = None
        self.output_dim = None
        self._load_base_model()

    def _load_base_model(self):
        if self.model_name == 'vgg16':
            self.base_model = VGG16(weights='imagenet')
            self.preprocess_input_func = vgg16_preprocess
            self.output_dim = 4096 # Output of VGG16 FC2 layer
            # Create a new model that outputs features from the last pooling layer or FC2 layer
            self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('fc2').output)
        elif self.model_name == 'resnet50':
            self.base_model = ResNet50(weights='imagenet')
            self.preprocess_input_func = resnet50_preprocess
            self.output_dim = 2048 # Output of ResNet50 avg_pool layer
            # Create a new model that outputs features from the last pooling layer
            self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('avg_pool').output)
        else:
            raise ValueError("Unsupported feature model. Choose 'vgg16' or 'resnet50'.")

    def get_model(self):
        return self.model

    def get_preprocess_input_func(self):
        return self.preprocess_input_func
        
    def get_output_dim(self):
        return self.output_dim

    def extract_features(self, image_path):
        """Extracts features from a single image."""
        try:
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = self.preprocess_input_func(image)
            features = self.model.predict(image, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

if __name__ == '__main__':
    # Example usage (for testing this module independently)
    print("Running feature_extractor.py directly for testing...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dummy_image_path = os.path.join(current_dir, 'dummy_image.jpg')

    # Create a blank white image if dummy_image.jpg doesn't exist for testing
    if not os.path.exists(dummy_image_path):
        try:
            from PIL import Image
            img = Image.new('RGB', (224, 224), color='white')
            img.save(dummy_image_path)
            print(f"Created dummy image: {dummy_image_path}")
        except ImportError:
            print("Pillow (PIL) not found. Cannot create dummy image. Please install it: pip install Pillow")
            dummy_image_path = None # Set to None if cannot create or find dummy

    if dummy_image_path and os.path.exists(dummy_image_path):
        print("\nTesting VGG16 Feature Extractor:")
        vgg_extractor = FeatureExtractor(model_name='vgg16')
        features_vgg = vgg_extractor.extract_features(dummy_image_path)
        if features_vgg is not None:
            print(f"VGG16 features shape: {features_vgg.shape}")
            print(f"VGG16 output dimension: {vgg_extractor.get_output_dim()}")

        print("\nTesting ResNet50 Feature Extractor:")
        resnet_extractor = FeatureExtractor(model_name='resnet50')
        features_resnet = resnet_extractor.extract_features(dummy_image_path)
        if features_resnet is not None:
            print(f"ResNet50 features shape: {features_resnet.shape}")
            print(f"ResNet50 output dimension: {resnet_extractor.get_output_dim()}")
    else:
        print("Skipping direct feature extraction test as no dummy image is available.")