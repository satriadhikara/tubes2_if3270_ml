import numpy as np
from .layers import Softmax 

class CNNModelFromScratch:
    def __init__(self, layers) -> None:
        """
        Initializes the CNN model with a list of layers.
        
        Input:
        layers: A list of layer instances (e.g., [Conv2D_instance, ReLU_instance, ...])
        """
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError("Layers must be a non-empty list of layer instances")
        self.layers = layers

    def forward(self, x):
        """
        Performs a forward pass through all layers.
        
        Parameters:
        x: input data (batch_size, height, width, channels) for CNN layers
        
        Returns:
        Output after passing through all layers. Shape depends on the final layer:
        - For Dense/Softmax layers: (batch_size, num_classes)
        - For CNN layers: (batch_size, height, width, channels)
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if len(x.shape) < 2:
            raise ValueError("Input must have at least 2 dimensions (batch_size, ...)")
            
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        """
        Makes predictions for the input data.
        
        Parameters:
        x: input data (batch_size, height, width, channels) for CNN layers.
        
        Returns:
        Predicted class labels (batch_size,).
        """
        output = self.forward(x)
        
        last_layer = self.layers[-1]
        if isinstance(last_layer, Softmax):
            probabilities = output
        elif hasattr(last_layer, 'activation') and last_layer.activation == 'softmax':
            probabilities = output
        else:
            softmax_layer = Softmax()
            probabilities = softmax_layer.forward(output)
            
        return np.argmax(probabilities, axis=1) 