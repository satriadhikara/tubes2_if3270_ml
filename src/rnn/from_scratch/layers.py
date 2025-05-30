import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def tanh(x):
    return np.tanh(x)

class Embedding:
    def __init__(self, weights):
        self.weights = weights
        self.vocab_size, self.embedding_dim = weights.shape
    
    def forward(self, x):
        return self.weights[x]

class SimpleRNN:
    def __init__(self, kernel, recurrent_kernel, bias, return_sequences=False):
        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.bias = bias
        self.return_sequences = return_sequences
        self.units = kernel.shape[1]
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
    
        h = np.zeros((batch_size, self.units))
        
        outputs = []
        
        for t in range(seq_len):
            h = tanh(np.dot(x[:, t, :], self.kernel) + np.dot(h, self.recurrent_kernel) + self.bias)
            outputs.append(h)
        
        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return outputs[-1]

class RNNCell:
    def __init__(self, kernel, recurrent_kernel, bias):
        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.bias = bias
        self.units = kernel.shape[1]
    
    def forward(self, x, h_prev):
        return tanh(np.dot(x, self.kernel) + np.dot(h_prev, self.recurrent_kernel) + self.bias)

class Bidirectional:
    def __init__(self, forward_rnn, backward_rnn):
        self.forward_rnn = forward_rnn
        self.backward_rnn = backward_rnn
        self.return_sequences = forward_rnn.return_sequences
    
    def forward(self, x):
        forward_output = self.forward_rnn.forward(x)

        x_reversed = x[:, ::-1, :]
        backward_output = self.backward_rnn.forward(x_reversed)
        
        if self.return_sequences:
            backward_output = backward_output[:, ::-1, :]
            return np.concatenate([forward_output, backward_output], axis=-1)
        else:
            return np.concatenate([forward_output, backward_output], axis=-1)

class Dropout:
    def __init__(self, rate):
        self.rate = rate
    
    def forward(self, x, training=False):
        if training and self.rate > 0:
            mask = np.random.binomial(1, 1 - self.rate, x.shape) / (1 - self.rate)
            return x * mask
        else:
            return x

class Dense:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

class Softmax:
    def __init__(self):
        pass
    
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class Flatten:
    def __init__(self):
        pass
    
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)