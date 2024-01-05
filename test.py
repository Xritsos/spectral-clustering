import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh, eig
from numpy import linalg
from matplotlib import pyplot as plt
import warnings
from scipy.sparse.linalg import eigsh
import networkx as nx

import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def rbf(x1, x2, gamma):
    
    n = x1.shape[0]
    m = x2.shape[0]
    
    xx1 = np.dot(np.sum(np.power(x1, 2), 1).reshape(n, 1), np.ones((1, m)))
    xx2 = np.dot(np.sum(np.power(x2, 2), 1).reshape(m, 1), np.ones((1, n))) 
    
    result = np.exp(-(xx1 + xx2.T - 2 * np.dot(x1, x2.T)) * gamma) 
    
    return result


def read_data():
    # batch = unpickle('./cifar/data_batch_1')
    
    # data = batch[b'data'][:1000]
    
    df = pd.read_csv('./cereal.csv')
    #print(df.columns)

    x1 = df['calories'].to_numpy()
    x2 = df['fat'].to_numpy()
    
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)
    
    x = np.concatenate((x1, x2), axis=1)
    
    return x


def calculate_eigenmaps(neighbors, data):
    
    adj_matrix_obj = kneighbors_graph(data, n_neighbors=neighbors, metric='euclidean', 
                                      mode='connectivity', include_self=False)
    
    adj_matrix = adj_matrix_obj.toarray()
    adj_matrix += adj_matrix.T
    adj_matrix = adj_matrix / 2
    
    print("Adjacency Matrix")
    print(adj_matrix)
    print()
    
    # weights = rbf(x, x, gamma=2)
    
    # check for fully connected graph
    G = nx.from_numpy_array(adj_matrix)
    cc = nx.number_connected_components(G)
    print()
    print(f"Connected Components: {cc}")
    
    if cc != 1:
        warnings.warn("Graph is not fully connected, Laplacian Eigenmaps may not work as expected")
    
    d = adj_matrix.sum(axis=1)
    D = np.diag(d)
    print("Degree Matrix")
    print(D)
    
    laplacian =  D - adj_matrix
    
    # # degree = np.identity(adj_matrix.shape[0]) * neighbors
    
    I = np.ones((adj_matrix.shape[0], adj_matrix.shape[0]))
    
    D = linalg.inv(D)
    D = np.sqrt(D)
    
    # normalized Laplacian
    laplacian_rw = np.matmul(np.matmul(D, laplacian), D)
    
    
    sum_ = []
    for i in range(laplacian_rw.shape[0]):
        sum_.append(np.sum(laplacian_rw[i, :]))
    
    print("Laplacian row sum")
    print(sorted(sum_))
    print()
    
    sum_ = []
    for i in range(laplacian_rw.shape[1]):
        sum_.append(np.sum(laplacian_rw[:, i]))
    
    print("Laplacian col sum")
    print(sorted(sum_))
    print()
    
    # print("Laplacian Normalized")
    # print(laplacian_rw)
    # print()
    # print("Laplacian")
    # print(laplacian)
    
    eigen_values, eigen_vectors = eigh(laplacian_rw)
    # eigen_values, eigen_vectors = eigh(laplacian, D)
    
    # # # # sort values and vectors
    # # # idxs = np.argsort(eigen_values)
        
    # # # eigen_values = eigen_values[idxs]
    # # # eigen_vectors = eigen_vectors[idxs]
    
    return eigen_values, eigen_vectors



if __name__ == "__main__":
    x = read_data()
    
    print(x.shape)
    # neighbors = int(np.ceil(np.sqrt(x.shape[0])))
    neighbors = 3
    print()
    print(f"Number of Neighbors: {neighbors}")
    
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    eigen_values, eigen_vectors = calculate_eigenmaps(neighbors, x)
    # calculate_eigenmaps(neighbors, x)
    
    print(eigen_values)
    