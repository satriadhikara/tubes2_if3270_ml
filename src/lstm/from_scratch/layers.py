# filepath: c:\Users\User\Documents\Semester6\ML\tubes2_if3270_ml\src\lstm\from_scratch\layers.py
import numpy as np

class Embedding:
    def __init__(self, weights):
        self.weights = weights
        self.vocab_size, self.embedding_dim = weights.shape

    def forward(self, x):
        # Simply look up embeddings for each token index
        return self.weights[x]


class LSTMCell:
    def __init__(self, weights_kernel, weights_recurrent, bias):
        self.weights_kernel = weights_kernel  # Input weights
        self.weights_recurrent = weights_recurrent  # Recurrent weights
        self.bias = bias
        
        # Determine hidden dimension (LSTM state size)
        self.hidden_dim = weights_recurrent.shape[0]
        
    def forward(self, x, h_prev, c_prev):
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
        self.cell = LSTMCell(weights_kernel, weights_recurrent, bias)
        self.return_sequences = return_sequences
        self.hidden_dim = self.cell.hidden_dim
        
    def forward(self, x):
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
        self.forward_layer = forward_lstm  # Changed to forward_layer for consistency with parameter counting
        self.backward_layer = backward_lstm  # Changed to backward_layer for consistency with parameter counting
        self.return_sequences = forward_lstm.return_sequences
    def forward(self, x):
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
        self.rate = rate
        
    def forward(self, x):
        # During inference, we don't apply dropout
        return x


class Dense:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        # Note: for parameter counting, we expose the weights and bias attributes
        
    def forward(self, x):
        return np.dot(x, self.weights) + self.bias


class Softmax:
    def forward(self, x):
        # For numerical stability, subtract max value
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Helper activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Clip for numerical stability

def tanh(x):
    return np.tanh(x)


class Flatten:
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
