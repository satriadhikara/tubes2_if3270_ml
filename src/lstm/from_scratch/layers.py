# filepath: c:\Users\User\Documents\Semester6\ML\tubes2_if3270_ml\src\lstm\from_scratch\layers.py
import numpy as np

class Embedding:
    def __init__(self, weights):
        """
        Initialize the Embedding layer with pre-trained weights
        
        Parameters:
        - weights: numpy array with shape (vocab_size, embedding_dim)
        """
        self.weights = weights
        self.vocab_size, self.embedding_dim = weights.shape

    def forward(self, x):
        """
        Forward pass for embedding layer
        
        Parameters:
        - x: input with shape (batch_size, sequence_length) containing token indices
        
        Returns:
        - embedded: output tensor with shape (batch_size, sequence_length, embedding_dim)
        """
        # Simply look up embeddings for each token index
        return self.weights[x]


class LSTMCell:
    def __init__(self, weights_kernel, weights_recurrent, bias):
        """
        Initialize LSTM cell with weights from Keras model
        
        Parameters:
        - weights_kernel: Input-to-hidden weights (input_dim, 4*hidden_dim)
        - weights_recurrent: Hidden-to-hidden weights (hidden_dim, 4*hidden_dim)
        - bias: Bias term (4*hidden_dim,)
        """
        self.weights_kernel = weights_kernel  # Input weights
        self.weights_recurrent = weights_recurrent  # Recurrent weights
        self.bias = bias
        
        # Determine hidden dimension (LSTM state size)
        self.hidden_dim = weights_recurrent.shape[0]
        
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for a single LSTM cell
        
        Parameters:
        - x: input tensor with shape (batch_size, input_dim)
        - h_prev: previous hidden state with shape (batch_size, hidden_dim)
        - c_prev: previous cell state with shape (batch_size, hidden_dim)
        
        Returns:
        - h_next: next hidden state
        - c_next: next cell state
        """
        # Concatenate input and hidden state multiplication
        z = np.dot(x, self.weights_kernel) + np.dot(h_prev, self.weights_recurrent) + self.bias
        
        # Split z into the 4 gates
        z_i, z_f, z_c, z_o = np.split(z, 4, axis=1)
        
        # Apply activations for each gate
        i = sigmoid(z_i)  # Input gate
        f = sigmoid(z_f)  # Forget gate
        c_temp = np.tanh(z_c)  # Cell gate
        o = sigmoid(z_o)  # Output gate
        
        # Update cell and hidden states
        c_next = f * c_prev + i * c_temp
        h_next = o * np.tanh(c_next)
        
        return h_next, c_next


class LSTM:
    def __init__(self, weights_kernel, weights_recurrent, bias, return_sequences=False):
        """
        Initialize LSTM layer with weights from Keras model
        
        Parameters:
        - weights_kernel: Input-to-hidden weights (input_dim, 4*hidden_dim)
        - weights_recurrent: Hidden-to-hidden weights (hidden_dim, 4*hidden_dim)
        - bias: Bias term (4*hidden_dim,)
        - return_sequences: Whether to return output for all time steps
        """
        self.cell = LSTMCell(weights_kernel, weights_recurrent, bias)
        self.return_sequences = return_sequences
        self.hidden_dim = self.cell.hidden_dim
        
    def forward(self, x):
        """
        Forward pass for LSTM layer
        
        Parameters:
        - x: input tensor with shape (batch_size, sequence_length, input_dim)
        
        Returns:
        - output: If return_sequences is True, output has shape (batch_size, sequence_length, hidden_dim)
                 Otherwise, output has shape (batch_size, hidden_dim)
        """
        batch_size, time_steps, input_dim = x.shape
        
        # Initialize cell and hidden states with zeros
        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))
        
        # Store all hidden states if return_sequences is True
        if self.return_sequences:
            h_sequence = np.zeros((batch_size, time_steps, self.hidden_dim))
        
        # Iterate through each time step
        for t in range(time_steps):
            h, c = self.cell.forward(x[:, t, :], h, c)
            
            if self.return_sequences:
                h_sequence[:, t, :] = h
        
        # Return all hidden states or only the last one
        if self.return_sequences:
            return h_sequence
        else:
            return h


class Bidirectional:
    def __init__(self, forward_lstm, backward_lstm):
        """
        Initialize Bidirectional wrapper with forward and backward LSTM layers
        
        Parameters:
        - forward_lstm: LSTM layer for forward pass
        - backward_lstm: LSTM layer for backward pass
        """
        self.forward_layer = forward_lstm  # Changed to forward_layer for consistency with parameter counting
        self.backward_layer = backward_lstm  # Changed to backward_layer for consistency with parameter counting
        self.return_sequences = forward_lstm.return_sequences
    def forward(self, x):
        """
        Forward pass for bidirectional LSTM
        
        Parameters:
        - x: input tensor with shape (batch_size, sequence_length, input_dim)
        
        Returns:
        - output: Concatenated output from forward and backward LSTMs
        """
        # Forward direction
        forward_output = self.forward_layer.forward(x)
        
        # Backward direction (reverse the time dimension)
        x_reversed = x[:, ::-1, :]
        backward_output = self.backward_layer.forward(x_reversed)
        
        if self.return_sequences:
            # Reverse the backward output back to the correct time order
            backward_output = backward_output[:, ::-1, :]
        
        # Concatenate outputs from both directions
        return np.concatenate([forward_output, backward_output], axis=-1)


class Dropout:
    def __init__(self, rate=0.0):
        """
        Initialize Dropout layer
        
        Parameters:
        - rate: dropout rate (not used in forward pass during inference)
        """
        self.rate = rate
        
    def forward(self, x):
        """
        Forward pass for Dropout layer (during inference, this is identity function)
        
        Parameters:
        - x: input tensor
        
        Returns:
        - output: same as input (during inference)
        """
        # During inference, we don't apply dropout
        return x


class Dense:
    def __init__(self, weights, bias):
        """
        Initialize Dense layer with weights from Keras model
        
        Parameters:
        - weights: weight matrix with shape (input_dim, output_dim)
        - bias: bias vector with shape (output_dim,)
        """
        self.weights = weights
        self.bias = bias
        # Note: for parameter counting, we expose the weights and bias attributes
        
    def forward(self, x):
        """
        Forward pass for Dense layer
        
        Parameters:
        - x: input tensor with shape (batch_size, input_dim)
        
        Returns:
        - output: output tensor with shape (batch_size, output_dim)
        """
        return np.dot(x, self.weights) + self.bias


class Softmax:
    def forward(self, x):
        """
        Forward pass for Softmax activation
        
        Parameters:
        - x: input logits
        
        Returns:
        - probabilities: softmax probabilities
        """
        # For numerical stability, subtract max value
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Helper activation functions
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Clip for numerical stability

def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)


class Flatten:
    def forward(self, x):
        """
        Forward pass for Flatten layer
        
        Parameters:
        - x: input tensor with shape (batch_size, ...)
        
        Returns:
        - flattened output with shape (batch_size, prod(x.shape[1:]))
        """
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
