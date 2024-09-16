import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import joblib
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (each pixel value is between 0 and 1)
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# One-hot encode the labels (for 10 output classes)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Neural Network class definition
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, Z):
        # ReLU activation function
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        # Softmax function for output layer
        expZ = np.exp(Z - np.max(Z))  # Stability improvement
        return expZ / expZ.sum(axis=1, keepdims=True)
    
    def forward(self, X):
        # Forward pass: input -> hidden -> output
        Z1 = X.dot(self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = self.softmax(Z2)
        return A1, A2
    
    def compute_loss(self, A2, Y):
        # Cross-entropy loss
        m = Y.shape[0]
        log_probs = -np.log(A2[range(m), np.argmax(Y, axis=1)])
        loss = np.sum(log_probs) / m
        return loss

    def backward(self, X, Y, A1, A2, learning_rate=0.01):
        # Backpropagation
        m = X.shape[0]

        dZ2 = A2 - Y
        dW2 = A1.T.dot(dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * (A1 > 0)
        dW1 = X.T.dot(dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

def train(model, X_train, Y_train, epochs=10, batch_size=64, learning_rate=0.01):
    loss_history = []
    accuracy_history = []
    
    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        Y_train_shuffled = Y_train[permutation]

        epoch_loss = 0
        epoch_accuracy = 0
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            Y_batch = Y_train_shuffled[i:i+batch_size]

            # Forward pass
            A1, A2 = model.forward(X_batch)

            # Compute loss
            loss = model.compute_loss(A2, Y_batch)
            epoch_loss += loss

            # Backward pass
            model.backward(X_batch, Y_batch, A1, A2, learning_rate)

            # Calculate batch accuracy
            batch_predictions = np.argmax(A2, axis=1)
            batch_labels = np.argmax(Y_batch, axis=1)
            batch_accuracy = np.mean(batch_predictions == batch_labels)
            epoch_accuracy += batch_accuracy
        
        epoch_loss /= (X_train.shape[0] // batch_size)
        epoch_accuracy /= (X_train.shape[0] // batch_size)

        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    return loss_history, accuracy_history

def evaluate(model, X_test, Y_test):
    _, A2 = model.forward(X_test)
    predictions = np.argmax(A2, axis=1)
    labels = np.argmax(Y_test, axis=1)
    accuracy = np.mean(predictions == labels)
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# Function to save the model using joblib
def save_model(model, filename):
    model_dict = {
        'W1': model.W1,
        'b1': model.b1,
        'W2': model.W2,
        'b2': model.b2
    }
    joblib.dump(model_dict, filename)
    print(f"Model saved to {filename}")

# Function to load the model using joblib
def load_model(filename):
    model_dict = joblib.load(filename)
    return model_dict

# Function to plot loss and accuracy over epochs
def plot_metrics(loss_history, accuracy_history):
    epochs = range(1, len(loss_history) + 1)
    
    # Plot Loss
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, 'r', label='Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_history, 'b', label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

# Instantiate the neural network
input_size = 784  # 28x28 pixels flattened
hidden_size = 128
output_size = 10  # 10 digits (0–9)

model = NeuralNetwork(input_size, hidden_size, output_size)

# Train the model
loss_history, accuracy_history = train(model, x_train, y_train, epochs=10, batch_size=64, learning_rate=0.01)

# Evaluate the model
evaluate(model, x_test, y_test)

# Save the trained model
save_model(model, 'mnist_neural_network.pkl')

# Plot loss and accuracy
plot_metrics(loss_history, accuracy_history)