from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.utils import plot_model

def create_model(feature_dim, vocab_size, max_length):
    # Feature Extractor Model (Image Feature Input)
    inputs1 = Input(shape=(feature_dim,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence Model (Caption Input)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2) # mask_zero handles padding
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder Model (Merge Feature and Sequence Models)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Combine into a single model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    # You can optionally visualize the model architecture
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model

if __name__ == '__main__':
    # Example Usage for testing model_architectures
    print("Running model_architectures.py directly for testing...")
    
    # Define dummy parameters for model creation
    dummy_feature_dim = 4096 # Example for VGG16
    dummy_vocab_size = 5000  # Example vocabulary size
    dummy_max_length = 30    # Example max caption length

    print(f"\nCreating a dummy model with:")
    print(f"  Feature Dimension: {dummy_feature_dim}")
    print(f"  Vocabulary Size: {dummy_vocab_size}")
    print(f"  Max Length: {dummy_max_length}")

    model = create_model(dummy_feature_dim, dummy_vocab_size, dummy_max_length)
    
    print("\nModel Summary:")
    model.summary()

    # You can also try to save a plot of the model
    # try:
    #     plot_model(model, to_file='dummy_model_architecture.png', show_shapes=True, show_layer_names=True)
    #     print("\nModel architecture plot saved as dummy_model_architecture.png")
    # except ImportError:
    #     print("\nGraphviz not installed. Cannot plot model architecture. Install with 'pip install pydot graphviz' and Graphviz executable.")