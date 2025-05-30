import numpy as np

class Embedding:
    def __init__(self, weights):
        self.weights = weights
        self.vocab_size, self.embedding_dim = weights.shape

    def forward(self, x):
        return self.weights[x]


class LSTMCell:
    def __init__(self, weights_kernel, weights_recurrent, bias):
        self.weights_kernel = weights_kernel  
        self.weights_recurrent = weights_recurrent 
        self.bias = bias
        
        self.hidden_dim = weights_recurrent.shape[0]
        
    def forward(self, x, h_prev, c_prev):
        z = np.dot(x, self.weights_kernel) + np.dot(h_prev, self.weights_recurrent) + self.bias

        z_i, z_f, z_c, z_o = np.split(z, 4, axis=1)

        i = sigmoid(z_i)  
        f = sigmoid(z_f) 
        c_temp = np.tanh(z_c)  
        o = sigmoid(z_o)  
 
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
        
        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))

        if self.return_sequences:
            h_sequence = np.zeros((batch_size, time_steps, self.hidden_dim))

        for t in range(time_steps):
            h, c = self.cell.forward(x[:, t, :], h, c)
            
            if self.return_sequences:
                h_sequence[:, t, :] = h

        if self.return_sequences:
            return h_sequence
        else:
            return h


class Bidirectional:
    def __init__(self, forward_lstm, backward_lstm):
        self.forward_layer = forward_lstm 
        self.backward_layer = backward_lstm 
        self.return_sequences = forward_lstm.return_sequences
    def forward(self, x):
        forward_output = self.forward_layer.forward(x)
        
        x_reversed = x[:, ::-1, :]
        backward_output = self.backward_layer.forward(x_reversed)
        
        if self.return_sequences:
            backward_output = backward_output[:, ::-1, :]

        return np.concatenate([forward_output, backward_output], axis=-1)


class Dropout:
    def __init__(self, rate=0.0):
        self.rate = rate
        
    def forward(self, x):
        return x


class Dense:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def forward(self, x):
        return np.dot(x, self.weights) + self.bias


class Softmax:
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -15, 15))) 

def tanh(x):
    return np.tanh(x)


class Flatten:
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
