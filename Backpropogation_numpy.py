import numpy as np  
# It's the XOR problem  
# DataSetup 
"""X is input data and y is true labels"""
X= np.array([[0,0], 
             [0,1],
             [1,0],
             [1,1]])  
y = np.array([[0],[1],[1],[0]])
np.random.seed(42) # For reproducibility 
# Architecture is 2 input neurons 4 hidden neurons and 1 output neuron 
input_size = 2 
hidden_size = 4 
output_size = 1 
learning_rate = 0.1 

# Weights and Biases 
W1 = np.random.randn(input_size, hidden_size) 
b1 = np.zeros((1, hidden_size)) 

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation Fucntion 
def sigmoid(x): 
    return 1/ (1 + np.exp(-x)) 
def sigmoid_derivative(x): 
    return x * (1-x) 
# Training Loop 
print("Start training")

for epoch in range(10000): 
    # forward propogation 
    # hidden layer 
    z1 = np.dot(X, W1) + b1  
    a1 = sigmoid(z1)  
    # Output layer 
    z2 = np.dot(a1, W2) + b2 
    output = sigmoid(z2)  
    
    #### BackPropogation  
    """We need to contribute how much weight 
       contributed to error so we move backward from output 
       to input 
    """
    # Calculating Error at output 
    error = y - output 
    d_output = error * sigmoid_derivative(output)
    """Calculating gradient for hidden layer. How much hidden 
      layer contributed to output error? So we will take error from 
      future and bring it back through W2 
    """
    error_hidden_layer = d_output.dot(W2.T)   
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(a1) 
    
    # Optimization 
    """Updating weights based on gradients we just calcualted"""
    # Updating W2 and b1 
    W2 += a1.dot(d_hidden_layer) * learning_rate 
    b1 += np.sum(d_hidden_layer,axis=0, keepdims=True) * learning_rate
    if epoch % 1000 == 0:
        loss = np.mean(np.square(y - output)) # Simple MSE for display
        print(f"Epoch {epoch}: Loss {loss:.5f}")

# 4. TESTING
print("\nFinal Predictions:")
print(output.round())
print("(Should be [[0], [1], [1], [0]])")
    
    








