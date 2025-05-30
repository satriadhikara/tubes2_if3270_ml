import tensorflow as tf
import numpy as np
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from lstm_keras_training import (
        test_sequences, test_labels, vectorizer, 
        best_result, build_lstm_model
    )
    
    best_config = best_result['config']
    
    data_to_save = {
        'test_sequences': test_sequences,
        'test_labels': test_labels,
        'best_config': best_config,
        'vocab_size': len(vectorizer.get_vocabulary()),
    }
    
    with open('lstm_saved_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print("Data successfully saved to lstm_saved_data.pkl")
    print(f"Saved test_sequences shape: {test_sequences.shape}")
    print(f"Saved test_labels shape: {test_labels.shape}")
    print(f"Saved best_config: {best_config}")
    
except Exception as e:
    print(f"Error saving data: {e}")
    print("Make sure to run this script after executing all cells in lstm_keras_training.ipynb")
