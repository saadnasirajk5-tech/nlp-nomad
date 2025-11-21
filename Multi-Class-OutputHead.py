
### COMBINING SOFTMAX WITH CATEGORICAL CROSS ENTROPY ### 
"""When we combine Softmax with categorical cross entropy 
   the math simplifies massively and the gradient(error signal is clean)
"""
# We are gonna implement softmax , cross-entropy-loss and softmax cross entropy backward 

import numpy as np 
def softmax(z): 
    # Exponentiating the scores  
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    # Normalizing 
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
    return probabilities 

def cross_entropy_loss(P, Y):
    N = Y.shape[0] # Batch size 
    # Now to prevent log(0) errros we are clipping P  
    P_clipped = np.clip(P,1e-12, 1.0 - 1e-12) 
    loss = -np.sum(Y * np.log(P_clipped)) / N       
    return loss       

# Combined softmax cross entropy backward  
def softmax_cross_entropy_backward(P, Y): 
    """
    The combined backward pass: Softmax + Cross-Entropy Loss.
    This is the "cheat code" of Deep Learning math.
    
    P: Predicted probabilities (Softmax output) (N, C)
    Y: True one-hot labels (N, C)
    
    The derivative of CCE loss w.r.t the pre-activation scores (logits)
    simplifies to the error in probabilities""" 

    N = Y.shape[0]    
    dZ = P - Y     
    # Normalize by batch size N 
    dZ /= N       
    return dZ         

# --- 3. DEMO AND INTEGRATION ---

# Dummy Data Setup: 4 samples, 3 classes (C1, C2, C3)
# Z = Raw Logits (e.g., output from MyLinear layer)
Z = np.array([
    [1.0, 2.0, 3.0],    # Sample 1
    [-1.0, 0.5, 1.5],   # Sample 2
    [5.0, 1.0, 0.0],    # Sample 3
    [0.0, 0.0, 0.0]     # Sample 4
])

# True Labels (Y) - One-Hot Encoded
# Sample 1 is C3, Sample 2 is C2, Sample 3 is C1, Sample 4 is C1
Y = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0]
])

# FORWARD PASS
P = softmax(Z)
print("Predicted Probabilities (P):\n", P.round(3))

loss = cross_entropy_loss(P, Y)
print(f"\nCalculated Cross-Entropy Loss: {loss:.4f}")

# BACKWARD PASS (THE CRITICAL STEP)
dZ = softmax_cross_entropy_backward(P, Y)
print("\nGradient dZ (P - Y / N):\n", dZ.round(4))

# This dZ is the gradient you would pass directly to the backward() 
# function of the preceding MyLinear layer (e.g., layer2.backward(dZ))










