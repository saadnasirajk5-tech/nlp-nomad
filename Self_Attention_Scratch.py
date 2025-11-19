import numpy as np  
# Creating Self attention from scratch  
def softmax(x):
    e_x = np.exp(x - np.max(x,axis=-1,keepdims=True)) 
    # Dividing by sum of exponentials to normalize    
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


"""Initializing the weights for Query, key and value matrices. 
       In real they are learned during training but here we are
       initializing randomly to show the mechanics 
    """
class SelfAttentionLayer: 
    def __init__(self, embed_dim, d_model): # embed_dim is size of input vector for each word and d_model is sizof output internal vector 
        np.random.seed(42) 
        # Transforming input into key 
        self.W_q = np.random.rand(embed_dim, d_model) 
        # Key 
        self.W_k = np.random.rand(embed_dim, d_model) 
        # Value 
        self.W_v = np.random.rand(embed_dim, d_model) 
    def forward(self, inputs): 
        """passing the input tokens through attention mechanisms""" 
        # Creating Q,K,V matrices 
        #Xing input matrices by our weight matrices 
        queries = np.dot(inputs, self.W_q) 
        keys = np.dot(inputs, self.W_k) 
        values = np.dot(inputs, self.W_v)     
        print(f"Queries Shape{queries.shape}")   
        
        """ Calculating the attention scores so we can 
        see how much each word relates to other word 
        """ 
        # Taking the transpose of keys to align dimensions for matrix multiplication 
        scores = np.dot(queries, keys.T) 
        print(scores) 
        # If score is high its means they are highly related
        ## Scaling the scores  
        # Getting the dimensionality of the query vector 
        d_k = queries.shape[1] 
        scaled_scores = scores / np.sqrt(d_k) 
        # Appling softmax  
        attention_weights = softmax(scaled_scores) 
        print("Attention Weights") 
        print(np.round(attention_weights, 2)) 
        
        # weighted sum of values 
        """Multiplying the probability map with actual content"""
        output = np.dot(attention_weights, values) 
        return attention_weights, values     
    
    
########## RUNNING THE STIMULATION ##########
if __name__ == "__main__": 
    input_sentence_embeddings = np.array([
         [1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0], 
         [1.0, 1.0, 0.0, 0.0]  
    ])

print(f"Input Shape: {input_sentence_embeddings.shape}") 
attention_layer = SelfAttentionLayer(embed_dim=4, d_model=3) 
# Running the pass 
contextualized_embeddings, weights = attention_layer.forward(input_sentence_embeddings)
print(contextualized_embeddings) 
