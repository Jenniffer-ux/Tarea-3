import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation='relu'):
        self.layers = layers
        self.activation_name = activation
        self.weights = []
        self.biases = []
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.layers)-1):
            fan_in = self.layers[i]
            fan_out = self.layers[i+1]
            limit = np.sqrt(2 / fan_in) if self.activation_name == 'relu' else np.sqrt(1 / fan_in)
            w = np.random.randn(fan_in, fan_out) * limit
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def activation(self, x):
        if self.activation_name == 'relu':
            return np.maximum(0, x)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == 'tanh':
            return np.tanh(x)

    def activation_derivative(self, x):
        if self.activation_name == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_name == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(x)**2

    def forward(self, X):
        activations = [X]
        inputs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w) + b
            inputs.append(z)
            a = self.activation(z)
            activations.append(a)
        return activations, inputs

    def backward(self, X, y, activations, inputs, learning_rate):
        m = y.shape[0]
        delta = activations[-1] - y
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, delta) / m
            dB = np.sum(delta, axis=0, keepdims=True) / m
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(inputs[i-1])
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * dB

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            activations, inputs = self.forward(X)
            self.backward(X, y, activations, inputs, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
