import numpy as np   

# It's the XOR problem   
X = np.array([[0,0], 
              [0,1],
              [1,0],
              [1,1]])   
y = np.array([[0],[1],[1],[0]])

np.random.seed(42) 

input_size = 2 
hidden_size = 4 
output_size = 1 
learning_rate = 0.1 

# Weights and Biases 
W1 = np.random.randn(input_size, hidden_size) 
b1 = np.zeros((1, hidden_size)) 

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation Function 
def sigmoid(x): 
    return 1/ (1 + np.exp(-x)) 

def sigmoid_derivative(x): 
    return x * (1-x) 

print("Start training")

for epoch in range(10000): 
    # --- FORWARD PASS ---
    # Hidden layer 
    z1 = np.dot(X, W1) + b1   
    a1 = sigmoid(z1)   
    # Output layer 
    z2 = np.dot(a1, W2) + b2 
    output = sigmoid(z2)   
    
    # --- BACKPROPAGATION ---
    # 1. Output Layer Error
    error = y - output 
    d_output = error * sigmoid_derivative(output)
    
    # 2. Hidden Layer Error
    # We take error from output and bring it back through W2
    error_hidden_layer = d_output.dot(W2.T)    
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(a1) 
    
    # --- OPTIMIZATION ---
    
    # Updating Output Layer (W2, b2) using d_output
    # We need a1.T here because a1 was the input to this layer
    W2 += a1.T.dot(d_output) * learning_rate 
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    # Updating Hidden Layer (W1, b1) using d_hidden_layer
    # We need X.T here because X was the input to this layer
    W1 += X.T.dot(d_hidden_layer) * learning_rate
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.square(y - output)) 
        print(f"Epoch {epoch}: Loss {loss:.5f}")

# TESTING
print("\nFinal Predictions:")
print(output.round())


