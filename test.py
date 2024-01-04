import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh
from numpy import linalg
from matplotlib import pyplot as plt
import warnings

import networkx as nx


def rbf(x1, x2, gamma):
    
    n = x1.shape[0]
    m = x2.shape[0]
    
    xx1 = np.dot(np.sum(np.power(x1, 2), 1).reshape(n, 1), np.ones((1, m)))
    xx2 = np.dot(np.sum(np.power(x2, 2), 1).reshape(m, 1), np.ones((1, n))) 
    
    result = np.exp(-(xx1 + xx2.T - 2 * np.dot(x1, x2.T)) * gamma) 
    
    return result


if __name__ == "__main__":
    neighbors = 9
    
    df = pd.read_csv('./cereal.csv')

    x1 = df['calories'].to_numpy()
    x2 = df['fat'].to_numpy()
    
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)
    
    x = np.concatenate((x1, x2), axis=1)
    
    print(x.shape)
    
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    graph = kneighbors_graph(x, n_neighbors=neighbors, metric='euclidean', 
                             mode='connectivity', include_self='auto')
    
    graph = graph.toarray()
    
    # graph = rbf(x, x, gamma=0.00000004)
    
    G = nx.from_numpy_array(graph)
    cc = nx.number_connected_components(G)
    
    if cc != 1:
        warnings.warn("Graph is not fully connected, Laplacian Eigenmaps may not work as expected")
    
    
    degree = np.identity(graph.shape[0]) * neighbors
    
    I = np.identity(graph.shape[0])
    
    laplacian_rw = I - np.dot(linalg.inv(degree), graph)
    
    eigen_values, eigen_vectors = linalg.eigh(laplacian_rw)
    idxs = np.argsort(eigen_values)
        
    eigen_values = eigen_values[idxs]
    eigen_vectors = eigen_vectors[idxs]
    
    print(eigen_values)
    
    # laplacian = degree - graph
    
   
    # eigen_values, eigen_vectors = eigh(laplacian, degree)
    # eigen_values = eigen_values[1:]
    # eigen_vectors = eigen_vectors[1:]
    
    # print(eigen_values)
    # m = 2
    
    # eigen_values = eigen_values[:2]
    # eigen_vectors = eigen_vectors[:2]
    
    # x_new = np.dot(x, eigen_vectors)
    
    
    # print(x)
    # print(x_new)
    
    
    # fig, ax = plt.subplots(figsize=(8, 8))
    # cax = ax.matshow(graph, cmap='viridis')

    # # Display numerical values in each cell
    # for i in range(graph.shape[0]):
    #     for j in range(graph.shape[1]):
    #         ax.text(j, i, f'{graph[i, j]}', ha='center', va='center', color='white', fontsize=8)

    # # Add labels and show the plot
    # plt.title('Adjacency Matrix')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Sample Index')
    # plt.colorbar(cax, label='Distance')
    # plt.show()
    