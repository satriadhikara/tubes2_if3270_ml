"""
Export code to make the lstm_keras_training results importable.

This script should be run from the Jupyter notebook after training is complete.
"""

def export_notebook_data():
    """
    Export necessary data from the notebook to a module that can be imported
    """
    import pickle
    import os
    import inspect
    import sys
    
    # Get the caller's global variables
    # This assumes this is being run from within the lstm_keras_training notebook
    caller_globals = inspect.currentframe().f_back.f_globals
    
    # Get the required variables
    data_to_save = {
        'test_sequences': caller_globals.get('test_sequences'),
        'test_labels': caller_globals.get('test_labels'),
        'best_config': (
            caller_globals.get('best_result', {}).get('config') 
            if 'best_result' in caller_globals 
            else None
        ),
        'vocab_size': caller_globals.get('vocab_size'),
    }
    
    # Validate that we have the necessary data
    if not all(data_to_save.values()):
        missing = [k for k, v in data_to_save.items() if v is None]
        raise ValueError(f"Missing data: {', '.join(missing)}")
    
    # Save to file
    with open('lstm_saved_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print("Data successfully saved to lstm_saved_data.pkl")
    print("You can now run lstm_from_scratch_testing.ipynb")

# Add a cell to the end of lstm_keras_training.ipynb with:
# from export_notebook_data import export_notebook_data
# export_notebook_data()
