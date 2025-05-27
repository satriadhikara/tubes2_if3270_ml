"""
Helper script to save LSTM model data for reuse in the from_scratch implementation.
Run this after training the model in lstm_keras_training.ipynb to save the
necessary data that can be imported by other Python scripts.
"""

import tensorflow as tf
import numpy as np
import pickle
import os
import sys

# Add path to import from the notebook
sys.path.insert(0, os.path.dirname(__file__))

# Try to import variables from the keras training
try:
    # Get global variables from the notebook using notebook's globals
    from lstm_keras_training import (
        test_sequences, test_labels, vectorizer, 
        best_result, build_lstm_model
    )
    
    # Extract best config
    best_config = best_result['config']
    
    # Save the data
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
