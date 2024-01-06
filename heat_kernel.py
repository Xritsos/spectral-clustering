import numpy as np


def rbf(x1, x2, t):
    
    n = x1.shape[0]
    m = x2.shape[0]
    
    xx1 = np.dot(np.sum(np.power(x1, 2), 1).reshape(n, 1), np.ones((1, m)))
    xx2 = np.dot(np.sum(np.power(x2, 2), 1).reshape(m, 1), np.ones((1, n))) 
    
    result = np.exp(- (xx1 + xx2.T - 2 * np.dot(x1, x2.T)) / t) 
    
    return result
