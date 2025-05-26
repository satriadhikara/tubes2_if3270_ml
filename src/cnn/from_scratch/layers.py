import numpy as np

class Conv2D:
    def __init__(self, weights, biases, stride=1, padding=0):
        """
        weights: (filter_height, filter_width, input_channels, num_filters)
        biases: (num_filters,)
        stride: int
        padding: int or 'same' (you'll need to implement 'same' padding logic if used)
        """
        self.weights = weights
        self.biases = biases
        self.stride = stride
        self.padding = padding 

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
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size[0]
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
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size[0]
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
    def __init__(self, weights, biases):
        """
        weights: (input_features, num_neurons)
        biases: (num_neurons,)
        """
        self.weights = weights
        self.biases = biases
        self.input_data = None

    def forward(self, x):
        """
        x: input_data (batch_size, input_features)
        """
        self.input_data = x
        return np.dot(x, self.weights) + self.biases

class Softmax:
    def forward(self, x):
        """
        x: input_data (batch_size, num_classes)
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims=True) 