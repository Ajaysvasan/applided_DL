import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

class TextCNN:
    def __init__(self, maxFeatures, maxLength, embeddedDim, filters, kernelSize, outputDim):
        self.maxFeatures = maxFeatures
        self.maxLength = maxLength
        self.embeddedDim = embeddedDim
        self.filters = filters
        self.kernelSize = kernelSize
        self.outputDim = outputDim
        self.model = self.model_build()
        
    def model_build(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.maxFeatures, output_dim=self.embeddedDim, input_length=self.maxLength))
        model.add(Conv1D(filters=self.filters, kernel_size=self.kernelSize, activation="relu"))
        model.add(GlobalMaxPooling1D())
        
        # Output layer for binary classification
        model.add(Dense(self.outputDim, activation='sigmoid'))
        
        # Correct loss for binary classification
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        return model
    
    def train(self, xTrain, yTrain, xVal, yVal, batch_size, epochs=10):
        self.model.fit(xTrain, yTrain, validation_data=(xVal, yVal), epochs=epochs, batch_size=batch_size, verbose=1)
     
    def evaluate(self, xTest, yTest):
        return self.model.evaluate(xTest, yTest, verbose=1)


# =========================
# Example usage with IMDb dataset
# =========================

max_features = 5000   # Vocabulary size
maxlen = 100          # Max length of sequence
embedding_dim = 50
filters = 64
kernel_size = 3
output_dim = 1        # Binary classification
epochs = 5
batch_size = 32

# Load IMDb dataset
print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to the same length
print("Padding sequences...")
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Create and train TextCNN model
text_cnn = TextCNN(max_features, maxlen, embedding_dim, filters, kernel_size, output_dim)
print("Training model...")
text_cnn.train(X_train, y_train, X_test, y_test, batch_size, epochs)

# Evaluate the trained model
print("Evaluating model...")
loss, accuracy = text_cnn.evaluate(X_test, y_test)
print("✅ Test Loss:", loss)
print("✅ Test Accuracy:", accuracy)
