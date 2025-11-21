
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
    N = Y.shape(0) # Batch size 
    # Now to prevent log(0) errros we are clipping P  
    P_clipped = np.clip(P,1e-12, 1.0 - 1e-12) 
    loss = -np.sum(Y * np.log(P_clipped)) / N       
    return loss       
















