import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Assuming your other modules are in the 'src' directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_extractor import FeatureExtractor
from data_loader import DataLoader
from model_architectures import create_model

# --- Configuration ---
# Adjust these paths based on your setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
CAPTIONS_FILE = os.path.join(DATA_DIR, 'captions.txt') # Corrected from Flickr8k.token.txt based on your Get-ChildItem output

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(epochs, batch_size, feature_model_name):
    print(f"--- Training Model (Feature Model: {feature_model_name}) ---")

    # Initialize Feature Extractor
    extractor = FeatureExtractor(feature_model_name)
    feature_extractor_model = extractor.get_model()
    preprocess_input_func = extractor.get_preprocess_input_func()
    
    features_filepath = os.path.join(MODELS_DIR, f'features_{feature_model_name}.pkl')

    # Load and preprocess images if features not already extracted
    if not os.path.exists(features_filepath):
        print("Extracting image features (this may take a while)...")
        all_features = {}
        image_ids = os.listdir(IMAGES_DIR)
        for i, img_id in enumerate(image_ids):
            if i % 100 == 0:
                print(f"Processing image {i}/{len(image_ids)}")
            img_path = os.path.join(IMAGES_DIR, img_id)
            try:
                img = load_img(img_path, target_size=(224, 224))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input_func(img)
                feature = feature_extractor_model.predict(img, verbose=0)
                all_features[img_id] = feature.flatten()
            except Exception as e:
                print(f"Error processing image {img_id}: {e}")
                continue # Skip image if there's an error

        with open(features_filepath, 'wb') as f:
            pickle.dump(all_features, f)
        print(f"Image features extracted and saved to {features_filepath}")
    else:
        print(f"Loading pre-extracted image features from {features_filepath}")
        with open(features_filepath, 'rb') as f:
            all_features = pickle.load(f)

    # Load data
    data_loader = DataLoader(CAPTIONS_FILE, IMAGES_DIR)
    descriptions = data_loader.load_descriptions()
    
    # Filter descriptions to only include images for which features were extracted
    # Ensure that only images with both descriptions AND features are processed
    # Also, ensure 'startseq' and 'endseq' are correctly added by data_loader
    # It's better to clean descriptions first, then add start/end seq and filter
    
    # Clean descriptions (important: do this before adding start/end tokens for vocabulary building)
    cleaned_descriptions_raw = data_loader.clean_descriptions(descriptions)
    
    # Add startseq and endseq tokens and convert to lines for tokenizer
    all_desc_lines_with_tokens = data_loader.descriptions_to_lines(cleaned_descriptions_raw)

    # Initialize tokenizer with ALL descriptions to build full vocabulary
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc_lines_with_tokens)
    vocab_size = len(tokenizer.word_index) + 1 # +1 for out-of-vocabulary words
    print(f"Tokenizer Vocabulary Size: {vocab_size}")

    # Determine max sequence length (from descriptions with start/end tokens)
    max_length = data_loader.max_len(cleaned_descriptions_raw)
    print(f"Max Description Length: {max_length}")

    # Save tokenizer and max_length
    with open(os.path.join(MODELS_DIR, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(MODELS_DIR, 'max_length.pkl'), 'wb') as f:
        pickle.dump(max_length, f)
    print("Tokenizer and max_length saved.")

    # Now prepare filtered descriptions for the generator
    # Filter descriptions to only include images for which features were extracted
    filtered_descriptions_for_generator = {
        img_id: data_loader.descriptions_to_lines({img_id: desc_list}) # Re-wrap for descriptions_to_lines
        for img_id, desc_list in cleaned_descriptions_raw.items() if img_id in all_features
    }
    
    if not filtered_descriptions_for_generator:
        print("No matching image features found for descriptions. Training aborted.")
        return

    image_ids_list_for_generator = list(filtered_descriptions_for_generator.keys())
    
    # Generator for training data
    def data_generator(descriptions_for_gen, features, tokenizer, max_length, vocab_size, image_ids_batch):
        while True:
            for i in range(0, len(image_ids_batch), batch_size):
                batch_image_ids = image_ids_batch[i:i+batch_size]
                
                X1, X2, y = [], [], []
                for img_id in batch_image_ids:
                    # descriptions_for_gen now directly contains lines with 'startseq'/'endseq'
                    desc_lines = descriptions_for_gen[img_id] 
                    img_features = features[img_id]
                    
                    for desc_line in desc_lines: # Iterate over the already tokenized lines
                        seq = tokenizer.texts_to_sequences([desc_line])[0]
                        for i in range(1, len(seq)):
                            in_seq, out_seq = seq[:i], seq[i]
                            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                            X1.append(img_features)
                            X2.append(in_seq)
                            y.append(out_seq)
                
                if X1 and X2 and y: # Ensure batch is not empty
                    # CRITICAL FIX: Yield inputs as a tuple (X1, X2) instead of a list [X1, X2]
                    yield (np.array(X1), np.array(X2)), np.array(y) 
                
    # Create the model
    model = create_model(feature_dim=extractor.get_output_dim(), vocab_size=vocab_size, max_length=max_length)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print("Starting model training...")
    try:
        model.fit(
            data_generator(filtered_descriptions_for_generator, all_features, tokenizer, max_length, vocab_size, image_ids_list_for_generator),
            epochs=epochs,
            steps_per_epoch=len(image_ids_list_for_generator) // batch_size, # Simplified steps per epoch
            verbose=1
        )
        model.save(os.path.join(MODELS_DIR, 'image_captioning_model.h5'))
        print(f"Model trained and saved to {os.path.join(MODELS_DIR, 'image_captioning_model.h5')}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Training aborted.")


def predict_caption(image_path, feature_model_name):
    print(f"--- Predicting Caption (Feature Model: {feature_model_name}) ---")

    # Load saved model, tokenizer, and max_length paths
    model_path = os.path.join(MODELS_DIR, 'image_captioning_model.h5')
    tokenizer_path = os.path.join(MODELS_DIR, 'tokenizer.pkl')
    max_length_path = os.path.join(MODELS_DIR, 'max_length.pkl')

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}. Please train the model first.")
        return
    if not os.path.exists(max_length_path):
        print(f"Error: Max length not found at {max_length_path}. Please train the model first.")
        return

    tokenizer = None
    max_length = None
    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(max_length_path, 'rb') as f:
            max_length = pickle.load(f)
    except Exception as e:
        print(f"Error loading tokenizer or max_length: {e}")
        return

    # Initialize Feature Extractor and get its output dimension
    extractor = FeatureExtractor(feature_model_name)
    feature_extractor_model = extractor.get_model()
    preprocess_input_func = extractor.get_preprocess_input_func()
    
    # --- CRITICAL FIX for Keras 3 loading .h5 models ---
    # 1. Re-create the model architecture
    if tokenizer is None or max_length is None: # Should not happen if previous loads worked
        print("Error: Tokenizer or max_length could not be loaded, cannot recreate model.")
        return
        
    vocab_size = len(tokenizer.word_index) + 1 # +1 for out-of-vocabulary words
    model = create_model(feature_dim=extractor.get_output_dim(), vocab_size=vocab_size, max_length=max_length)
    
    # 2. Load only the weights from the .h5 file
    try:
        model.load_weights(model_path)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {e}")
        print("This might indicate an incompatibility or corruption. Training and saving again might help.")
        return
    # --- END CRITICAL FIX ---


    # Load and preprocess the input image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_input = np.expand_dims(img_array, axis=0)
        img_input = preprocess_input_func(img_input)
        image_feature = feature_extractor_model.predict(img_input, verbose=0)
    except Exception as e:
        print(f"Error processing input image {image_path}: {e}")
        return

    # Generate caption
    def generate_caption(model, tokenizer, photo_features, max_length):
        in_text = 'startseq'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([photo_features, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat, None)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text

    caption = generate_caption(model, tokenizer, image_feature, max_length)
    final_caption = caption.replace('startseq ', '').replace(' endseq', '')
    print(f"Generated Caption: {final_caption}")

    # Display image with caption
    try:
        image = mpimg.imread(image_path)
        plt.imshow(image)
        plt.title(final_caption)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {e}")
        print("Make sure matplotlib is installed and your display environment is correctly set up.")


def main():
    parser = argparse.ArgumentParser(description="Image Captioning Application")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help="Mode of operation: 'train' or 'predict'")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of epochs for training (only applicable in train mode)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training (only applicable in train mode)")
    parser.add_argument('--feature_model', type=str, default='vgg16', choices=['vgg16', 'resnet50'],
                        help="Pre-trained CNN model for feature extraction (vgg16 or resnet50)")
    parser.add_argument('--image_to_predict', type=str,
                        help="Path to the image for prediction (only applicable in predict mode)")

    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args.epochs, args.batch_size, args.feature_model)
    elif args.mode == 'predict':
        if not args.image_to_predict:
            print("Error: --image_to_predict is required for 'predict' mode.")
            parser.print_help()
            sys.exit(1)
        predict_caption(args.image_to_predict, args.feature_model)

if __name__ == "__main__":
    main()