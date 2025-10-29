import numpy as np

class NeuralNetwork:
    def __init__(self, input_size:int , hidden_size:int , output_size:int , learning_rate:int = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_hidden_input = np.random.randn(self.input_size , self.hidden_size)
        self.bias_hidden = np.zeros((1,self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size , self.output_size)
        self.bias_hidden_output = np.zeros((1 , self.output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, X:int):
        return 1 / (1 + np.exp(-X))
    
    def derevative_sigmoid(self , X:int):
        return X*(X - 1)

    def forward(self , inputs):
        self.hidden_input = np.dot(inputs, self.weights_hidden_input) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output , self.weights_hidden_output) + self.bias_hidden_output
        return self.sigmoid(self.output)
    
    def backward(self ,inputs , label , output):
        error_output = label - output
        delta_output = error_output * self.derevative_sigmoid(output)
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.derevative_sigmoid(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * self.learning_rate 
        self.bias_hidden_output += np.sum(delta_output , axis=0 , keepdims=True) * self.learning_rate
        self.weights_hidden_input += inputs.T.dot(delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden , axis=0 , keepdims=True) * self.learning_rate
    
    def train(self , X , y , epochs):
        for _ in range(epochs):
            output= self.forward(X)
            self.backward(X , y , output)



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
y = np.array([[0], [1], [1], [0]]) 
# Define neural network parameters 
input_size = 2 
hidden_size = 4 
output_size = 1 
learning_rate = 0.1 
epochs = 10000 
# Create and train neural network 
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate) 
nn.train(X, y, epochs) 
# Test the trained neural network 
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
predicted_output = nn.forward(test_input) 
print("Predicted output:") 
print(predicted_output)