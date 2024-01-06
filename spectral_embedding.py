import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh, eig
import networkx as nx

import warnings

from heat_kernel import rbf

class laplacian_eigenmaps():
    
    def __init__(self, n_dimensions):
        self.k = None
        self.adj_matrix = None
        self.eigen_values = None
        self.eigen_vectors = None
        self.n_zero_values = 1
        self.n_dimensions = n_dimensions
        
    def adjacency_matrix(self, data, neighbors):
        self.k = neighbors
        # graph
        adj_matrix_obj = kneighbors_graph(data, n_neighbors=self.k, 
                                          metric='euclidean',
                                          mode='connectivity', 
                                          include_self=True)
        # graph to numpy
        self.adj_matrix = adj_matrix_obj.toarray()
        
        # self.adj_matrix *= rbf(data, data, t=20)
        
        # check for symmetry
        is_symmetric = np.allclose(self.adj_matrix, self.adj_matrix.T, 
                                   rtol=1e-05, atol=1e-08)
        
        if not is_symmetric:
            self.adj_matrix = (self.adj_matrix + self.adj_matrix.T) / 2
            
        # check for fully connected graph
        G = nx.from_numpy_array(self.adj_matrix)
        self.n_zero_values = nx.number_connected_components(G)
        
        print()
        print(f"Connected Components of Graph: {self.n_zero_values}")
        
        if self.n_zero_values != 1:
            warnings.warn("Graph is not fully connected !")
            
        return self.adj_matrix
            
    def eigen_maps(self):
        # compute degree matrix
        D = self.adj_matrix.sum(axis=1)
        D = np.diag(D)
        
        Laplacian = D - self.adj_matrix
        
        # test Laplacian
        is_sum_zero = True
        
        sum_ = []
        # check rows
        for i in range(Laplacian.shape[0]):
            sum_.append(np.sum(Laplacian[i, :]))
        
        if np.sum(sum_) != 0:
            is_sum_zero = False
    
        sum_ = []
        # check cols
        for i in range(Laplacian.shape[1]):
            sum_.append(np.sum(Laplacian[:, i]))
            
        if np.sum(sum_) != 0:
            is_sum_zero = False
        
        # if not is_sum_zero:
        #     raise ValueError("Laplacian Matrix has non zero sums !")
        
        # compute eigenvectors and eigenvalues
        self.eigen_values, self.eigen_vectors = eigh(Laplacian, D)
        
        # keep only the non-zeros
        self.eigen_values = self.eigen_values[:]
        self.eigen_vectors = self.eigen_vectors[:, :self.n_dimensions]
        
        if self.n_dimensions > self.eigen_vectors.shape[1]:
            raise ValueError("Number of dimensions cannot be greater than initial !")
        
        return self.eigen_values, self.eigen_vectors
        