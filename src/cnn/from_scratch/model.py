import numpy as np
from .layers import Softmax 

class CNNModelFromScratch:
    def __init__(self, layers) -> None:
        """
        layers: A list of layer instances (e.g., [Conv2D_instance, ReLU_instance, ...])
        """
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError("Layers must be a non-empty list of layer instances")
        self.layers = layers

    def forward(self, x):
        """
        Performs a forward pass through all layers.
        x: input data (batch_size, height, width, channels) for CNN layers
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
        For classification, this would be the output of the final layer.
        Returns predicted class indices.
        """
        output = self.forward(x)
        
        # Check if the last layer is a Softmax or Dense with softmax activation
        last_layer = self.layers[-1]
        if isinstance(last_layer, Softmax):
            # Output is already probabilities
            probabilities = output
        elif hasattr(last_layer, 'activation') and last_layer.activation == 'softmax':
            # Dense layer with softmax activation - output is already probabilities
            probabilities = output
        else:
            # Output is logits, need to apply softmax
            softmax_layer = Softmax()
            probabilities = softmax_layer.forward(output)
            
        return np.argmax(probabilities, axis=1) 