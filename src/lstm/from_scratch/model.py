import numpy as np
import tensorflow as tf
import h5py
import os

class LSTMModelFromScratch:    
    def __init__(self, layers):
        self.layers = layers
        
        self.total_params = self.count_parameters()
        print(f"From-scratch model created with {self.total_params:,} total parameters")
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, x):
        logits = self.forward(x)

        return np.argmax(logits, axis=1)
    
    def count_parameters(self):
        total_params = 0
        
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                total_params += np.size(layer.weights)
                if hasattr(layer, 'bias'):
                    total_params += np.size(layer.bias)

            elif hasattr(layer, 'cell'):
                # Count LSTM parameters: kernel, recurrent, bias
                if hasattr(layer.cell, 'weights_kernel'):
                    total_params += np.size(layer.cell.weights_kernel)
                if hasattr(layer.cell, 'weights_recurrent'):
                    total_params += np.size(layer.cell.weights_recurrent)
                if hasattr(layer.cell, 'bias'):
                    total_params += np.size(layer.cell.bias)

            elif hasattr(layer, 'forward_layer') and hasattr(layer, 'backward_layer'):
                # Count forward LSTM parameters
                if hasattr(layer.forward_layer.cell, 'weights_kernel'):
                    total_params += np.size(layer.forward_layer.cell.weights_kernel)
                if hasattr(layer.forward_layer.cell, 'weights_recurrent'):
                    total_params += np.size(layer.forward_layer.cell.weights_recurrent)
                if hasattr(layer.forward_layer.cell, 'bias'):
                    total_params += np.size(layer.forward_layer.cell.bias)

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

        model = tf.keras.models.load_model(keras_model_path) if keras_model_path.endswith('.h5') else None
        
        if model is None:
            temp_model = tf.keras.models.load_model(keras_model_path.replace('.weights.h5', '.h5'))
            if temp_model is None:  
                pass
            
        # Extract weights from Keras model
        layers_from_scratch = []
        keras_layers = model.layers

        for i, layer in enumerate(keras_layers):
            if isinstance(layer, tf.keras.layers.Embedding):
                weights = layer.get_weights()[0]
                layers_from_scratch.append(Embedding(weights))
                
            elif isinstance(layer, tf.keras.layers.LSTM):
                weights = layer.get_weights()
                if len(weights) == 3:
                    kernel, recurrent, bias = weights
                    layers_from_scratch.append(LSTM(kernel, recurrent, bias, layer.return_sequences))
                
            elif isinstance(layer, tf.keras.layers.Bidirectional):
                weights = layer.get_weights()
                if len(weights) == 6:  
                    forward_kernel, forward_recurrent, forward_bias = weights[0:3]
                    backward_kernel, backward_recurrent, backward_bias = weights[3:6]
                    
                    forward_lstm = LSTM(forward_kernel, forward_recurrent, forward_bias, layer.layer.return_sequences)
                    backward_lstm = LSTM(backward_kernel, backward_recurrent, backward_bias, layer.layer.return_sequences)
                    
                    layers_from_scratch.append(Bidirectional(forward_lstm, backward_lstm))
                
            elif isinstance(layer, tf.keras.layers.Dropout):
                layers_from_scratch.append(Dropout(layer.rate))
                
            elif isinstance(layer, tf.keras.layers.Dense):
                weights, bias = layer.get_weights()
                layers_from_scratch.append(Dense(weights, bias))

                if i == len(keras_layers) - 1 and layer.activation.__name__ == 'softmax':
                    layers_from_scratch.append(Softmax())
                    
            elif isinstance(layer, tf.keras.layers.Flatten):
                layers_from_scratch.append(Flatten())
        
        return cls(layers_from_scratch)
        
    @classmethod
    def from_keras_weights(cls, weights_path, model_config, vectorizer=None):
        from .layers import Embedding, LSTM, Bidirectional, Dropout, Dense, Softmax

        temp_model = create_keras_model_from_config(model_config)
        
        # Load weights
        temp_model.load_weights(weights_path)

        layers_from_scratch = []
        
        for layer in temp_model.layers:
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

    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length
    ))

    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1
        
        if bidirectional:
            model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units_per_layer[i], return_sequences=return_sequences)
            ))
        else:
            model.add(tf.keras.layers.LSTM(units_per_layer[i], return_sequences=return_sequences))

        model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
