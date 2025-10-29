import numpy as np 
from keras.models import Sequential
from keras.layers import Dense

class DeepNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, task):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.task = task
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        
        # Add first hidden layer with input size
        model.add(Dense(self.hidden_layers[0], input_dim=self.input_size, activation='relu'))

        # Add remaining hidden layers
        for layer in self.hidden_layers[1:]:
            model.add(Dense(layer, activation='relu'))
        
        # Output Layer
        if self.task == 'classification':
            model.add(Dense(self.output_size, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif self.task == 'regression':
            model.add(Dense(self.output_size, activation='linear'))
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        else:
            raise ValueError("Invalid task. Use 'classification' or 'regression'.")

        return model
    
    def train(self, inputs, labels, batch_size, epochs=1000):
        if self.task == 'classification':
            labels = np.eye(self.output_size)[labels]  # One-hot encoding
            self.model.fit(inputs, labels, epochs=epochs, batch_size=batch_size, verbose=0)
        elif self.task == 'regression':
            self.model.fit(inputs, labels, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, inputs):
        return self.model.predict(inputs)

# Data
X_classification = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
y_classification = np.array([0, 1, 1, 0])

X_regression = np.array([[1], [2], [3], [4], [5]]) 
y_regression = np.array([2, 4, 6, 8, 10]) 

# Hyperparameters
input_size_classification = X_classification.shape[1] 
input_size_regression = X_regression.shape[1] 
hidden_layers = [4, 3]  
output_size_classification = len(np.unique(y_classification)) 
output_size_regression = 1 
epochs = 100 
batch_size = 1 

# Classification Model
dnn_classification = DeepNeuralNetwork(input_size_classification, hidden_layers, 
                                       output_size_classification, 'classification') 
dnn_classification.train(X_classification, y_classification, batch_size, epochs) 

# Regression Model
dnn_regression = DeepNeuralNetwork(input_size_regression, hidden_layers, 
                                   output_size_regression, 'regression') 
dnn_regression.train(X_regression, y_regression, batch_size, epochs) 

# Predictions
test_input_classification = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
predicted_output_classification = dnn_classification.predict(test_input_classification) 
print("Predicted output for classification task:") 
print(predicted_output_classification) 

test_input_regression = np.array([[6], [7], [8]]) 
predicted_output_regression = dnn_regression.predict(test_input_regression) 
print("Predicted output for regression task:") 
print(predicted_output_regression)
