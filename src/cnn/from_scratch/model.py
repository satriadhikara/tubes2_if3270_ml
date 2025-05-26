import numpy as np
from .layers import Softmax 

class CNNModelFromScratch:
    def __init__(self, layers):
        """
        layers: A list of layer instances (e.g., [Conv2D_instance, ReLU_instance, ...])
        """
        self.layers = layers

    def forward(self, x):
        """
        Performs a forward pass through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        """
        For classification, this would be the output of the final Softmax layer.
        """
        logits = self.forward(x)
        if not isinstance(self.layers[-1], Softmax):
             softmax_layer = Softmax()
             probabilities = softmax_layer.forward(logits)
        else:
             probabilities = logits
        return np.argmax(probabilities, axis=1) 