import os
import string

class DataLoader:
    def __init__(self, captions_filepath, images_dir):
        self.captions_filepath = captions_filepath
        self.images_dir = images_dir
        self.max_description_length = 0

    def load_descriptions(self):
        """Loads descriptions from the captions file."""
        mapping = {}
        try:
            with open(self.captions_filepath, 'r') as f:
                next(f) # Skip header line 'image,caption'
                for line in f:
                    # UPDATED: Split by comma, limit to 1 split for caption to avoid splitting internal commas
                    parts = line.strip().split(',', 1) 
                    if len(parts) < 2:
                        continue # Skip malformed lines

                    image_id_full = parts[0].strip() # Image filename like 1000268201_693b08cb0e.jpg
                    
                    # UPDATED: Image ID is already the full filename, no need to strip #0 or similar
                    image_id = image_id_full 
                    
                    description = parts[1].strip()

                    # Only include descriptions for images that actually exist in the images directory
                    # This check is crucial for dataset consistency
                    if not os.path.exists(os.path.join(self.images_dir, image_id)):
                        # print(f"Warning: Image file {image_id} not found in {self.images_dir}. Skipping description.")
                        continue

                    if image_id not in mapping:
                        mapping[image_id] = []
                    mapping[image_id].append(description)
        except FileNotFoundError:
            print(f"Error: Captions file not found at {self.captions_filepath}")
            return {}
        except Exception as e:
            print(f"Error loading descriptions: {e}")
            return {}
        return mapping

    def clean_descriptions(self, descriptions):
        """Cleans the descriptions by lowercasing, removing punctuation, and words with numbers."""
        table = str.maketrans('', '', string.punctuation)
        cleaned_descriptions = {} # Create a new dict to avoid modifying in place while iterating
        for image_id, desc_list in descriptions.items():
            cleaned_list = []
            for desc in desc_list:
                # Tokenize (split into words)
                desc = desc.split()
                # Convert to lower case
                desc = [word.lower() for word in desc]
                # Remove punctuation from each word
                desc = [w.translate(table) for w in desc]
                # Remove words with numbers in them
                desc = [word for word in desc if word.isalpha()]
                # Store as string
                cleaned_list.append(' '.join(desc))
            cleaned_descriptions[image_id] = cleaned_list # Assign the cleaned list
        return cleaned_descriptions # Return the new, cleaned dictionary

    def descriptions_to_lines(self, descriptions):
        """Converts description dictionary to a list of all description strings."""
        all_desc_lines = []
        for img_id in descriptions: # Iterate through the dictionary
            for desc in descriptions[img_id]:
                # Add start and end sequence tokens
                all_desc_lines.append('startseq ' + desc + ' endseq')
        return all_desc_lines

    def to_vocabulary(self, descriptions):
        """Creates a set of all unique words (vocabulary)."""
        all_desc_lines = self.descriptions_to_lines(descriptions)
        vocabulary = set()
        for line in all_desc_lines:
            for word in line.split():
                vocabulary.add(word)
        return vocabulary

    def max_len(self, descriptions):
        """Determines the maximum description length."""
        all_desc_lines = self.descriptions_to_lines(descriptions)
        self.max_description_length = max(len(s.split()) for s in all_desc_lines) if all_desc_lines else 0
        return self.max_description_length

if __name__ == '__main__':
    # Example Usage for testing DataLoader
    print("Running data_loader.py directly for testing...")
    
    # Define dummy paths relative to the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dummy_data_dir = os.path.join(current_dir, '..', 'data_test')
    dummy_images_dir = os.path.join(dummy_data_dir, 'images')
    dummy_captions_file = os.path.join(dummy_data_dir, 'captions_test.txt')

    # Create dummy directories and files if they don't exist
    os.makedirs(dummy_images_dir, exist_ok=True)
    
    # Create a dummy captions file mimicking the user's format
    if not os.path.exists(dummy_captions_file):
        with open(dummy_captions_file, 'w') as f:
            f.write("image,caption\n") # Header
            f.write("image1_id.jpg,A man is walking.\n")
            f.write("image1_id.jpg,Another man is running.\n")
            f.write("image2_id.jpg,A dog barks at a cat.\n")
            f.write("image2_id.jpg,The cat is black.\n")
        print(f"Created dummy captions file: {dummy_captions_file}")

    # Create dummy images (can be empty files for testing purposes)
    for img_id in ['image1_id.jpg', 'image2_id.jpg']:
        with open(os.path.join(dummy_images_dir, img_id), 'w') as f:
            f.write('') # Empty file is enough for path existence check
        print(f"Created dummy image file: {img_id}")


    data_loader = DataLoader(dummy_captions_file, dummy_images_dir)

    print("\nLoading descriptions...")
    descriptions = data_loader.load_descriptions()
    print(f"Loaded {len(descriptions)} image descriptions.")
    for img_id, desc_list in list(descriptions.items())[:2]: # Print first 2
        print(f"  {img_id}: {desc_list}")

    print("\nCleaning descriptions...")
    cleaned_descriptions = data_loader.clean_descriptions(descriptions.copy()) # Pass a copy
    for img_id, desc_list in list(cleaned_descriptions.items())[:2]: # Print first 2
        print(f"  {img_id}: {desc_list}")

    print("\nCreating vocabulary...")
    vocabulary = data_loader.to_vocabulary(cleaned_descriptions)
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Vocabulary: {list(vocabulary)[:10]}...") # Print first 10

    print("\nDetermining max length...")
    max_length = data_loader.max_len(cleaned_descriptions)
    print(f"Max description length: {max_length}")
    
    # Clean up dummy files/dirs (optional)
    # import shutil
    # shutil.rmtree(dummy_data_dir)
    # print(f"\nCleaned up dummy directory: {dummy_data_dir}")