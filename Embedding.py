import warnings

import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh
import networkx as nx

from heat_kernel import rbf


class Spectral_Embedding():
    
    def __init__(self, n_dimensions, neighbors, rbf_on=False, t=10):
        self.k = None
        self.adj_matrix = None
        self.eigen_values = None
        self.eigen_vectors = None
        self.n_zero_values = 1
        self.n_dimensions = n_dimensions
        self.neighbors = neighbors
        self.rbf_on = rbf_on
        
    def transform(self, data):
        # graph
        adj_matrix_obj = kneighbors_graph(data, n_neighbors=self.neighbors, 
                                        metric='euclidean',
                                        mode='connectivity', 
                                        include_self=True)
        # graph to numpy
        self.adj_matrix = adj_matrix_obj.toarray()
            
        # check for symmetry
        is_symmetric = np.allclose(self.adj_matrix, self.adj_matrix.T, 
                                   rtol=1e-05, atol=1e-08)
        
        if not is_symmetric:
            self.adj_matrix = (self.adj_matrix + self.adj_matrix.T) / 2
            
        if self.rbf_on:
            self.adj_matrix *= rbf(data, data, t)
            
        # check for fully connected graph
        G = nx.from_numpy_array(self.adj_matrix)
        self.n_zero_values = nx.number_connected_components(G)
        
        print()
        print(f"Connected Components of Graph: {self.n_zero_values}")
        
        if self.n_zero_values != 1:
            warnings.warn("Graph is not fully connected !")
            
        # compute degree matrix
        D = self.adj_matrix.sum(axis=1)
        D = np.diag(D)
        
        Laplacian = D - self.adj_matrix
        
        # if solve the generalized problem -> L = normalized (random walk)
        # if solve simple eigenvalue problem -> L = simple
        # compute eigenvectors and eigenvalues (solution of Lrw)
        self.eigen_values, self.eigen_vectors = eigh(Laplacian, D)
        
        # keep only the non-zeros
        self.eigen_values = self.eigen_values[self.n_zero_values:]
        self.eigen_vectors = self.eigen_vectors[:, self.n_zero_values:]
        
        if self.n_dimensions > self.eigen_vectors.shape[1]:
            raise ValueError(f"Number of dimensions cannot be greater than {self.eigen_vectors.shape[1]} !")
        
        # keep only the specified dimensions
        self.eigen_vectors = self.eigen_vectors[:, :self.n_dimensions]
        
        return self.eigen_values, self.eigen_vectors
        