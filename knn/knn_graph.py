import numpy as np




class knn_graph():
    
    def __init__(self, k):
        self.k = k
    
    def fit(self, x):
        
        self.x = x
        
    def predict(self):
        n_samples, n_features = self.x.shape
        
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                sum_ = 0
                for f in range(n_features):
                    sum_ += (self.x[i, f] - self.x[j, f]) ** 2
                
                distance_matrix[i, j] = np.sqrt(sum_)
                
                
        for i in range(n_samples):
            sorted_ = np.sort(distance_matrix[i], axis=0)
            
                
        return distance_matrix
    
    