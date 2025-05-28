import numpy as np

class Conv2D:
    def __init__(self, weights, biases, stride=1, padding=0, activation=None):
        """
        weights: (filter_height, filter_width, input_channels, num_filters)
        biases: (num_filters,)
        stride: int
        padding: int or 'same' - 'same' padding maintains input spatial dimensions
        activation: str - 'relu' or None
        """
        # Input validation
        if not isinstance(weights, np.ndarray) or len(weights.shape) != 4:
            raise ValueError("Weights must be a 4D numpy array with shape (filter_height, filter_width, input_channels, num_filters)")
        if not isinstance(biases, np.ndarray) or len(biases.shape) != 1:
            raise ValueError("Biases must be a 1D numpy array with shape (num_filters,)")
        if weights.shape[3] != biases.shape[0]:
            raise ValueError("Number of filters in weights must match number of biases")
        if stride <= 0:
            raise ValueError("Stride must be a positive integer")
        
        self.weights = weights
        self.biases = biases
        self.stride = stride
        self.activation = activation
        
        # Handle padding parameter
        if isinstance(padding, str) and padding.lower() == 'same':
            # Calculate padding to maintain input spatial dimensions
            self.padding = (weights.shape[0] - 1) // 2
        elif isinstance(padding, int) and padding >= 0:
            self.padding = padding
        else:
            raise ValueError("Padding must be 'same' or a non-negative integer")

        self.input_shape = None
        self.output_shape = None
        self.input_data = None

    def _apply_padding(self, x):
        if self.padding > 0:
            return np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        return x

    def forward(self, x):
        """
        x: input_data (batch_size, height, width, input_channels)
        Returns: output of the convolution operation
        """
        self.input_data = x
        n_batch, h_in, w_in, c_in = x.shape
        fh, fw, _, n_filters = self.weights.shape

        x_padded = self._apply_padding(x)
        _, h_padded, w_padded, _ = x_padded.shape

        h_out = (h_padded - fh) // self.stride + 1 
        w_out = (w_padded - fw) // self.stride + 1 

        output = np.zeros((n_batch, h_out, w_out, n_filters))

        for i in range(n_batch):
            for f_idx in range(n_filters):
                current_y = out_y = 0
                while current_y + fh <= h_padded:
                    current_x = out_x = 0
                    while current_x + fw <= w_padded: 
                        patch = x_padded[i, current_y:current_y+fh, current_x:current_x+fw, :]
                        output[i, out_y, out_x, f_idx] = np.sum(patch * self.weights[:, :, :, f_idx]) + self.biases[f_idx]
                        current_x += self.stride
                        out_x += 1
                    current_y += self.stride
                    out_y += 1
        
        if self.activation == 'relu':
            output = np.maximum(0, output)
        
        return output

class ReLU:
    def __init__(self):
        self.input_data = None

    def forward(self, x):
        self.input_data = x
        return np.maximum(0, x)

class MaxPooling2D:
    def __init__(self, pool_size=(2, 2), stride=None):
        """
        pool_size: (height, width) of the pooling window
        stride: if None, defaults to pool_size
        """
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
            
        if stride is None:
            self.stride = self.pool_size[0]
        elif isinstance(stride, int):
            self.stride = stride
        else:
            raise ValueError("Stride must be an integer or None")
            
        self.input_data = None

    def forward(self, x):
        """
        x: input_data (batch_size, height, width, channels)
        """
        self.input_data = x
        n_batch, h_in, w_in, c_in = x.shape
        ph, pw = self.pool_size

        h_out = (h_in - ph) // self.stride + 1 
        w_out = (w_in - pw) // self.stride + 1

        output = np.zeros((n_batch, h_out, w_out, c_in))

        for i in range(n_batch):
            for c in range(c_in):
                current_y = out_y = 0
                while current_y + ph <= h_in:
                    current_x = out_x = 0
                    while current_x + pw <= w_in:
                        patch = x[i, current_y:current_y+ph, current_x:current_x+pw, c]
                        output[i, out_y, out_x, c] = np.max(patch)
                        current_x += self.stride
                        out_x += 1
                    current_y += self.stride
                    out_y += 1
        return output

class AveragePooling2D:
    def __init__(self, pool_size=(2, 2), stride=None):
        """
        pool_size: (height, width) of the pooling window
        stride: if None, defaults to pool_size
        """
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
            
        if stride is None:
            self.stride = self.pool_size[0]
        elif isinstance(stride, int):
            self.stride = stride
        else:
            raise ValueError("Stride must be an integer or None")
            
        self.input_data = None

    def forward(self, x):
        self.input_data = x
        n_batch, h_in, w_in, c_in = x.shape
        ph, pw = self.pool_size

        h_out = (h_in - ph) // self.stride + 1
        w_out = (w_in - pw) // self.stride + 1

        output = np.zeros((n_batch, h_out, w_out, c_in))

        for i in range(n_batch):
            for c in range(c_in):
                current_y = out_y = 0
                while current_y + ph <= h_in:
                    current_x = out_x = 0
                    while current_x + pw <= w_in:
                        patch = x[i, current_y:current_y+ph, current_x:current_x+pw, c]
                        output[i, out_y, out_x, c] = np.mean(patch)
                        current_x += self.stride
                        out_x += 1
                    current_y += self.stride
                    out_y += 1
        return output


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

class Dense:
    def __init__(self, weights, biases, activation=None):
        """
        weights: (input_features, num_neurons)
        biases: (num_neurons,)
        activation: str - 'relu', 'softmax', or None
        """
        # Input validation
        if not isinstance(weights, np.ndarray) or len(weights.shape) != 2:
            raise ValueError("Weights must be a 2D numpy array with shape (input_features, num_neurons)")
        if not isinstance(biases, np.ndarray) or len(biases.shape) != 1:
            raise ValueError("Biases must be a 1D numpy array with shape (num_neurons,)")
        if weights.shape[1] != biases.shape[0]:
            raise ValueError("Number of neurons in weights must match number of biases")
        
        self.weights = weights
        self.biases = biases
        self.activation = activation
        self.input_data = None

    def forward(self, x):
        """
        x: input_data (batch_size, input_features)
        """
        self.input_data = x
        output = np.dot(x, self.weights) + self.biases
        
        # Apply activation if specified
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'softmax':
            exp_x = np.exp(output - np.max(output, axis=1, keepdims=True))
            output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        return output

class Softmax:
    def forward(self, x):
        """
        x: input_data (batch_size, num_classes)
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims=True) 