import numpy as np
from sklearn.cluster import KMeans

import sys
sys.path.append('./')

from modules.Embedding import Spectral_Embedding



class Spectral_Clustering():
    
    def __init__(self, n_neighbors=5, rbf_on=False, n_clusters=5, 
                 n_dimensions=2, t=5):
        
        self.neighbors = n_neighbors
        self.rbf = rbf_on
        self.t = t
        self.n_clusters = n_clusters
        self.n_dim = n_dimensions
        self.kmeans = None
        self.centroids = None
        self.labels = None
        self.embedded_data = None
        
    def embedd(self, data):
        emb = Spectral_Embedding(n_dimensions=self.n_dim, rbf_on=self.rbf, 
                                 neighbors=self.neighbors, t=self.t)
        
        self.eigen_values, self.embedded_data = emb.transform(data=data)
        
    def cluster(self, data):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=7, n_init=10)
        self.kmeans.fit(data)
        
        self.labels = self.kmeans.predict(data)
        self.centroids = self.kmeans.cluster_centers_
        
        return self.labels
        
    def predict(self, data):
        
       return self.kmeans.predict(data)
    