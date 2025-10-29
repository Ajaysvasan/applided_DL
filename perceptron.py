import numpy as np

class Perceptron:
    def __init__(self,input_size:int , learning_rate:int = 0.01 , epochs:int = 1000):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size+1)

    def activation_function(self , x):
        return 1 if x>=0 else 0
    
    def predict(self, inputs):
        summation = np.dot(inputs , self.weights[1:]) + self.weights[0]
        return self.activation_function(summation)
    
    def train(self,train_inputs , labels):
        for _ in range(self.epochs):
            for input , label in zip(train_inputs , labels):
                prediction = self.predict(input)
                self.weights[1:]+= self.learning_rate * (label - prediction) * input
                self.weights[0] += self.learning_rate * (label - prediction)


train_inputs = np.array([[0,0 ] , [0 , 1] , [1, 0] ,[1,1]])
labels = np.array([0,0,0,1])
perceptron = Perceptron(input_size=2)
perceptron.train(train_inputs , labels)
test_input = np.array([1,1])
print(f"The predicted output is {perceptron.predict(test_input)}")