import numpy as np
import tensorflow as tf
import h5py
import os

class RNNModelFromScratch:    
    def __init__(self, layers):
        """
        Create a RNN model from a list of layers
        
        Parameters:
        - layers: a list of layer objects with forward() method
        """
        self.layers = layers
        
        # Calculate and display total parameters
        self.total_params = self.count_parameters()
        print(f"From-scratch RNN model created with {self.total_params:,} total parameters")
        
    def forward(self, x):
        """
        Forward pass through the entire model
        
        Parameters:
        - x: input data
        
        Returns:
        - output after passing through all layers
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, x):
        """
        Make predictions on input data
        
        Parameters:
        - x: input data
        
        Returns:
        - predicted class indices
        """
        # Forward pass
        logits = self.forward(x)
        
        # Convert to class predictions
        return np.argmax(logits, axis=1)
    
    def count_parameters(self):
        """
        Count the total number of parameters in the model
        
        Returns:
        - total_params: int, the total number of parameters
        """
        total_params = 0
        
        for layer in self.layers:
            # Check if the layer has weights attribute (like Embedding, RNN, Dense)
            if hasattr(layer, 'weights'):
                total_params += np.size(layer.weights)
                if hasattr(layer, 'bias'):
                    total_params += np.size(layer.bias)
                
            # Check if it's a RNN layer
            elif hasattr(layer, 'cell'):
                # Count RNN parameters: kernel, recurrent, bias
                if hasattr(layer.cell, 'weights_kernel'):
                    total_params += np.size(layer.cell.weights_kernel)
                if hasattr(layer.cell, 'weights_recurrent'):
                    total_params += np.size(layer.cell.weights_recurrent)
                if hasattr(layer.cell, 'bias'):
                    total_params += np.size(layer.cell.bias)
                    
            # Check if it's a bidirectional layer
            elif hasattr(layer, 'forward_layer') and hasattr(layer, 'backward_layer'):
                # Count forward RNN parameters
                if hasattr(layer.forward_layer.cell, 'weights_kernel'):
                    total_params += np.size(layer.forward_layer.cell.weights_kernel)
                if hasattr(layer.forward_layer.cell, 'weights_recurrent'):
                    total_params += np.size(layer.forward_layer.cell.weights_recurrent)
                if hasattr(layer.forward_layer.cell, 'bias'):
                    total_params += np.size(layer.forward_layer.cell.bias)
                
                # Count backward RNN parameters
                if hasattr(layer.backward_layer.cell, 'weights_kernel'):
                    total_params += np.size(layer.backward_layer.cell.weights_kernel)
                if hasattr(layer.backward_layer.cell, 'weights_recurrent'):
                    total_params += np.size(layer.backward_layer.cell.weights_recurrent)
                if hasattr(layer.backward_layer.cell, 'bias'):
                    total_params += np.size(layer.backward_layer.cell.bias)
                    
        return total_params
    
    def summary(self):
        """
        Print a summary of the model architecture and parameters
        Similar to Keras model.summary()
        """
        print("Model: RNN From Scratch")
        print("_" * 80)
        print("{:<5} {:<20} {:<15} {:<15}".format("Layer", "Type", "Output Shape", "Param #"))
        print("=" * 80)
        
        total_params = 0
        input_shape = None
        
        for i, layer in enumerate(self.layers):
            layer_type = layer.__class__.__name__
            layer_params = 0
            output_shape = "Unknown"
            
            # Calculate parameters for this layer
            if hasattr(layer, 'weights'):
                # Embedding or Dense layer
                layer_params += np.size(layer.weights)
                if hasattr(layer, 'bias'):
                    layer_params += np.size(layer.bias)
                    
                if layer_type == 'Embedding':
                    vocab_size, embedding_dim = layer.weights.shape
                    output_shape = f"(None, None, {embedding_dim})"
                elif layer_type == 'Dense':
                    input_dim, output_dim = layer.weights.shape
                    output_shape = f"(None, {output_dim})"
                
            # RNN layers
            elif hasattr(layer, 'cell'):
                if hasattr(layer.cell, 'weights_kernel'):
                    input_dim, output_dim = layer.cell.weights_kernel.shape
                    layer_params += np.size(layer.cell.weights_kernel)
                if hasattr(layer.cell, 'weights_recurrent'):
                    layer_params += np.size(layer.cell.weights_recurrent)
                if hasattr(layer.cell, 'bias'):
                    layer_params += np.size(layer.cell.bias)
                
                if layer.return_sequences:
                    output_shape = f"(None, None, {output_dim})"
                else:
                    output_shape = f"(None, {output_dim})"
            
            # Bidirectional layers
            elif hasattr(layer, 'forward_layer'):
                # Forward RNN
                if hasattr(layer.forward_layer.cell, 'weights_kernel'):
                    input_dim, output_dim = layer.forward_layer.cell.weights_kernel.shape
                    layer_params += np.size(layer.forward_layer.cell.weights_kernel)
                if hasattr(layer.forward_layer.cell, 'weights_recurrent'):
                    layer_params += np.size(layer.forward_layer.cell.weights_recurrent)
                if hasattr(layer.forward_layer.cell, 'bias'):
                    layer_params += np.size(layer.forward_layer.cell.bias)
                
                # Backward RNN
                if hasattr(layer.backward_layer.cell, 'weights_kernel'):
                    layer_params += np.size(layer.backward_layer.cell.weights_kernel)
                if hasattr(layer.backward_layer.cell, 'weights_recurrent'):
                    layer_params += np.size(layer.backward_layer.cell.weights_recurrent)
                if hasattr(layer.backward_layer.cell, 'bias'):
                    layer_params += np.size(layer.backward_layer.cell.bias)
                
                if layer.return_sequences:
                    output_shape = f"(None, None, {output_dim*2})"  # *2 for bidirectional
                else:
                    output_shape = f"(None, {output_dim*2})"  # *2 for bidirectional
                
            print("{:<5} {:<20} {:<15} {:<15}".format(
                i, layer_type, output_shape, f"{layer_params:,}" if layer_params > 0 else "0"
            ))
            
            total_params += layer_params
            
        print("=" * 80)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {total_params:,}")
        print(f"Non-trainable params: 0")
        print("_" * 80)
        
    @classmethod
    def from_keras_model(cls, keras_model_path, vectorizer=None):
        """
        Create a RNNModelFromScratch instance from a saved Keras model
        
        Parameters:
        - keras_model_path: path to the saved Keras model weights (.h5 file)
        - vectorizer: TextVectorization layer for tokenization (optional)
        
        Returns:
        - RNNModelFromScratch instance
        """
        from .layers import Embedding, SimpleRNN, Bidirectional, Dropout, Dense, Softmax, Flatten
        
        # Load Keras model weights
        model = tf.keras.models.load_model(keras_model_path) if keras_model_path.endswith('.h5') else None
        
        if model is None:
            # If not a full model, try loading just the weights
            temp_model = tf.keras.models.load_model(keras_model_path.replace('.weights.h5', '.h5'))
            if temp_model is None:  
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
                
            elif isinstance(layer, tf.keras.layers.SimpleRNN):
                # Extract RNN weights: kernel, recurrent, bias
                weights = layer.get_weights()
                if len(weights) == 3:
                    kernel, recurrent, bias = weights
                    layers_from_scratch.append(SimpleRNN(kernel, recurrent, bias, layer.return_sequences))
                
            elif isinstance(layer, tf.keras.layers.Bidirectional):
                # Extract bidirectional RNN weights
                weights = layer.get_weights()
                if len(weights) == 6:  # Two sets of weights for forward and backward
                    forward_kernel, forward_recurrent, forward_bias = weights[0:3]
                    backward_kernel, backward_recurrent, backward_bias = weights[3:6]
                    
                    forward_rnn = SimpleRNN(forward_kernel, forward_recurrent, forward_bias, layer.layer.return_sequences)
                    backward_rnn = SimpleRNN(backward_kernel, backward_recurrent, backward_bias, layer.layer.return_sequences)
                    
                    layers_from_scratch.append(Bidirectional(forward_rnn, backward_rnn))
                
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
        """
        Alternative method to create a model from just the weights file and a config
        
        Parameters:
        - weights_path: path to Keras .weights.h5 file
        - model_config: dictionary with model configuration
        - vectorizer: TextVectorization layer (optional)
        
        Returns:
        - RNNModelFromScratch instance
        """
        from .layers import Embedding, SimpleRNN, Bidirectional, Dropout, Dense, Softmax
        
        # Create a Keras model with the same architecture
        temp_model = create_keras_model_from_config(model_config)
        
        # Load weights
        temp_model.load_weights(weights_path)
        
        # Now extract weights and create from_scratch layers
        layers_from_scratch = []
        
        for layer in temp_model.layers:
            # Similar logic as from_keras_model, but using the temp_model
            pass
            
        return cls(layers_from_scratch)


def create_keras_model_from_config(config):
    """
    Creates a Keras model based on the configuration using SimpleRNN
    
    Parameters:
    - config: dictionary with model configuration
    
    Returns:
    - Keras model
    """
    rnn_layers = config.get('rnn_layers', 1)
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
    
    # Add RNN layers
    for i in range(rnn_layers):
        return_sequences = i < rnn_layers - 1
        
        if bidirectional:
            model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.SimpleRNN(units_per_layer[i], return_sequences=return_sequences)
            ))
        else:
            model.add(tf.keras.layers.SimpleRNN(units_per_layer[i], return_sequences=return_sequences))
        
        # Add dropout after each RNN layer
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