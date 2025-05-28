# filepath: c:\Users\User\Documents\Semester6\ML\tubes2_if3270_ml\src\lstm\from_scratch\model.py
import numpy as np
import tensorflow as tf
import h5py
import os

class LSTMModelFromScratch:    
    def __init__(self, layers):
        self.layers = layers
        
        # Calculate and display total parameters
        self.total_params = self.count_parameters()
        print(f"From-scratch model created with {self.total_params:,} total parameters")
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, x):
        # Forward pass
        logits = self.forward(x)
        
        # Convert to class predictions
        return np.argmax(logits, axis=1)
    
    def count_parameters(self):
        total_params = 0
        
        for layer in self.layers:
            # Check if the layer has weights attribute (like Embedding, LSTM, Dense)
            if hasattr(layer, 'weights'):
                total_params += np.size(layer.weights)
                if hasattr(layer, 'bias'):
                    total_params += np.size(layer.bias)
                
            # Check if it's an LSTM layer
            elif hasattr(layer, 'cell'):
                # Count LSTM parameters: kernel, recurrent, bias
                if hasattr(layer.cell, 'weights_kernel'):
                    total_params += np.size(layer.cell.weights_kernel)
                if hasattr(layer.cell, 'weights_recurrent'):
                    total_params += np.size(layer.cell.weights_recurrent)
                if hasattr(layer.cell, 'bias'):
                    total_params += np.size(layer.cell.bias)
                    
            # Check if it's a bidirectional layer
            elif hasattr(layer, 'forward_layer') and hasattr(layer, 'backward_layer'):
                # Count forward LSTM parameters
                if hasattr(layer.forward_layer.cell, 'weights_kernel'):
                    total_params += np.size(layer.forward_layer.cell.weights_kernel)
                if hasattr(layer.forward_layer.cell, 'weights_recurrent'):
                    total_params += np.size(layer.forward_layer.cell.weights_recurrent)
                if hasattr(layer.forward_layer.cell, 'bias'):
                    total_params += np.size(layer.forward_layer.cell.bias)
                
                # Count backward LSTM parameters
                if hasattr(layer.backward_layer.cell, 'weights_kernel'):
                    total_params += np.size(layer.backward_layer.cell.weights_kernel)
                if hasattr(layer.backward_layer.cell, 'weights_recurrent'):
                    total_params += np.size(layer.backward_layer.cell.weights_recurrent)
                if hasattr(layer.backward_layer.cell, 'bias'):
                    total_params += np.size(layer.backward_layer.cell.bias)
                    
        return total_params

    @classmethod
    def from_keras_model(cls, keras_model_path, vectorizer=None):
       
        from .layers import Embedding, LSTM, Bidirectional, Dropout, Dense, Softmax, Flatten
        
        # Load Keras model weights
        model = tf.keras.models.load_model(keras_model_path) if keras_model_path.endswith('.h5') else None
        
        if model is None:
            # If not a full model, try loading just the weights
            temp_model = tf.keras.models.load_model(keras_model_path.replace('.weights.h5', '.h5'))
            if temp_model is None:  
                # Create a dummy model and load weights
                # This part would need to be customized based on the actual model architecture
                pass
            
        # Extract weights from Keras model
        layers_from_scratch = []
        keras_layers = model.layers
        
        # Create corresponding from_scratch layers
        for i, layer in enumerate(keras_layers):
            if isinstance(layer, tf.keras.layers.Embedding):
                # Extract embedding weights
                weights = layer.get_weights()[0]
                layers_from_scratch.append(Embedding(weights))
                
            elif isinstance(layer, tf.keras.layers.LSTM):
                # Extract LSTM weights: kernel, recurrent, bias
                weights = layer.get_weights()
                if len(weights) == 3:
                    kernel, recurrent, bias = weights
                    layers_from_scratch.append(LSTM(kernel, recurrent, bias, layer.return_sequences))
                
            elif isinstance(layer, tf.keras.layers.Bidirectional):
                # Extract bidirectional LSTM weights
                weights = layer.get_weights()
                if len(weights) == 6:  # Two sets of weights for forward and backward
                    forward_kernel, forward_recurrent, forward_bias = weights[0:3]
                    backward_kernel, backward_recurrent, backward_bias = weights[3:6]
                    
                    forward_lstm = LSTM(forward_kernel, forward_recurrent, forward_bias, layer.layer.return_sequences)
                    backward_lstm = LSTM(backward_kernel, backward_recurrent, backward_bias, layer.layer.return_sequences)
                    
                    layers_from_scratch.append(Bidirectional(forward_lstm, backward_lstm))
                
            elif isinstance(layer, tf.keras.layers.Dropout):
                layers_from_scratch.append(Dropout(layer.rate))
                
            elif isinstance(layer, tf.keras.layers.Dense):
                # Extract dense weights and bias
                weights, bias = layer.get_weights()
                layers_from_scratch.append(Dense(weights, bias))
                
                # Add softmax activation if this is the output layer
                if i == len(keras_layers) - 1 and layer.activation.__name__ == 'softmax':
                    layers_from_scratch.append(Softmax())
                    
            elif isinstance(layer, tf.keras.layers.Flatten):
                layers_from_scratch.append(Flatten())
        
        return cls(layers_from_scratch)
        
    @classmethod
    def from_keras_weights(cls, weights_path, model_config, vectorizer=None):
        from .layers import Embedding, LSTM, Bidirectional, Dropout, Dense, Softmax
        
        # Create a Keras model with the same architecture
        temp_model = create_keras_model_from_config(model_config)
        
        # Load weights
        temp_model.load_weights(weights_path)
        
        # Now extract weights and create from_scratch layers
        layers_from_scratch = []
        
        for layer in temp_model.layers:
            # Similar logic as from_keras_model, but using the temp_model
            # This would need to be customized based on the model architecture
            pass
            
        return cls(layers_from_scratch)


def create_keras_model_from_config(config):
    lstm_layers = config.get('lstm_layers', 1)
    units_per_layer = config.get('units_per_layer', [64])
    bidirectional = config.get('bidirectional', False)
    embedding_dim = config.get('embedding_dim', 128)
    vocab_size = config.get('vocab_size', 10000)
    num_classes = config.get('num_classes', 5)
    sequence_length = config.get('sequence_length', 100)
    
    model = tf.keras.Sequential()
    
    # Add embedding layer
    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length
    ))
    
    # Add LSTM layers
    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1
        
        if bidirectional:
            model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units_per_layer[i], return_sequences=return_sequences)
            ))
        else:
            model.add(tf.keras.layers.LSTM(units_per_layer[i], return_sequences=return_sequences))
        
        # Add dropout after each LSTM layer
        model.add(tf.keras.layers.Dropout(0.3))
    
    # Add Dense output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
