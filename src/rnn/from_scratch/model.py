import numpy as np

class RNNModelFromScratch:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x, training=False):
        output = x
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                if 'training' in layer.forward.__code__.co_varnames:
                    output = layer.forward(output, training=training)
                else:
                    output = layer.forward(output)
            else:
                output = layer(output)
        return output
    
    def predict(self, x):
        if hasattr(x, 'numpy'):
            x = x.numpy()
        
        logits = self.forward(x, training=False)
        
        predictions = np.argmax(logits, axis=-1)
        
        return predictions
    
    def predict_proba(self, x):
        if hasattr(x, 'numpy'):
            x = x.numpy()
        
        probabilities = self.forward(x, training=False)
        
        return probabilities
    
    def summary(self):
        print("RNN Model Summary")
        print("=" * 50)
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = type(layer).__name__
            
            if hasattr(layer, 'weights') and layer.weights is not None:
                if hasattr(layer, 'bias') and layer.bias is not None:
                    params = layer.weights.size + layer.bias.size
                    print(f"Layer {i+1}: {layer_name} - Params: {params:,}")
                    print(f"  Weights: {layer.weights.shape}, Bias: {layer.bias.shape}")
                else:
                    params = layer.weights.size
                    print(f"Layer {i+1}: {layer_name} - Params: {params:,}")
                    print(f"  Weights: {layer.weights.shape}")
                total_params += params
                
            elif hasattr(layer, 'kernel') and layer.kernel is not None:
                params = layer.kernel.size + layer.recurrent_kernel.size + layer.bias.size
                print(f"Layer {i+1}: {layer_name} - Params: {params:,}")
                print(f"  Kernel: {layer.kernel.shape}, Recurrent: {layer.recurrent_kernel.shape}, Bias: {layer.bias.shape}")
                total_params += params
                
            else:
                print(f"Layer {i+1}: {layer_name} - No parameters")
        
        print("=" * 50)
        print(f"Total Parameters: {total_params:,}")

def create_keras_model_from_config(config):

    import tensorflow as tf
    
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(
        input_dim=config.get('vocab_size', 10000),
        output_dim=config.get('embedding_dim', 128),
        input_length=config.get('sequence_length', 100)
    ))
    
    rnn_layers = config.get('rnn_layers', 1)
    units_per_layer = config.get('units_per_layer', [64])
    bidirectional = config.get('bidirectional', False)
    
    for i in range(rnn_layers):
        return_sequences = i < rnn_layers - 1
        
        if bidirectional:
            model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.SimpleRNN(
                    units_per_layer[i], 
                    return_sequences=return_sequences
                )
            ))
        else:
            model.add(tf.keras.layers.SimpleRNN(
                units_per_layer[i], 
                return_sequences=return_sequences
            ))
        
        model.add(tf.keras.layers.Dropout(config.get('dropout_rate', 0.3)))
 
    model.add(tf.keras.layers.Dense(
        config.get('num_classes', 3), 
        activation='softmax'
    ))
 
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model