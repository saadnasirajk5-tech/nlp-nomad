import numpy as np   # Import numpy library for numerical operations

# It's the XOR problem   # Define input data for XOR logic gate
X = np.array([[0,0],  # Input pair 1
              [0,1],  # Input pair 2
              [1,0],  # Input pair 3
              [1,1]])  # Input pair 4

y = np.array([[0],[1],[1],[0]])  # Expected output for XOR inputs

np.random.seed(42)  # Set random seed for reproducibility

input_size = 2  # Number of input neurons
hidden_size = 4  # Number of neurons in hidden layer
output_size = 1  # Number of output neurons
learning_rate = 0.1  # Learning rate for weight updates

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)  # Weights from input to hidden layer
b1 = np.zeros((1, hidden_size))  # Biases for hidden layer

W2 = np.random.randn(hidden_size, output_size)  # Weights from hidden to output layer
b2 = np.zeros((1, output_size))  # Biases for output layer

# Activation function: sigmoid
def sigmoid(x):  # Sigmoid activation function
    return 1 / (1 + np.exp(-x))  # Calculate sigmoid

# Derivative of sigmoid function
def sigmoid_derivative(x):  # Derivative for backpropagation
    return x * (1 - x)  # Derivative formula

print("Start training")  # Indicate training start

for epoch in range(10000):  # Training loop for 10000 iterations
    # --- FORWARD PASS ---
    # Calculate hidden layer input
    z1 = np.dot(X, W1) + b1  # Weighted sum plus bias
    a1 = sigmoid(z1)  # Activation output of hidden layer
    # Calculate output layer input
    z2 = np.dot(a1, W2) + b2  # Weighted sum plus bias
    output = sigmoid(z2)  # Activation output of output layer

    # --- BACKPROPAGATION ---
    # Calculate error at output
    error = y - output  # Difference between expected and predicted
    d_output = error * sigmoid_derivative(output)  # Delta for output layer

    # Calculate error for hidden layer
    error_hidden_layer = d_output.dot(W2.T)  # Propagate error back
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(a1)  # Delta for hidden layer

    # --- OPTIMIZATION ---
    # Update weights and biases for output layer
    W2 += a1.T.dot(d_output) * learning_rate  # Adjust weights
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate  # Adjust biases

    # Update weights and biases for hidden layer
    W1 += X.T.dot(d_hidden_layer) * learning_rate  # Adjust weights
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate  # Adjust biases

    if epoch % 1000 == 0:  # Every 1000 epochs
        loss = np.mean(np.square(y - output))  # Calculate mean squared error
        print(f"Epoch {epoch}: Loss {loss:.5f}")  # Print loss

# TESTING
print("\nFinal Predictions:")  # Print final prediction message
print(output.round())  # Print rounded output predictions
