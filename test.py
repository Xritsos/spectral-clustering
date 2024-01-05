import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.datasets import make_swiss_roll
import pickle
from sklearn.cluster import KMeans

from spectral_embedding import laplacian_eigenmaps
from rbf_kernel import rbf


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_data():
    # batch = unpickle('./cifar/data_batch_1')
    
    # data = batch[b'data'][:1000]
    
    df = pd.read_csv('./cereal.csv')
    #print(df.columns)

    x1 = df['calories'].to_numpy()
    x2 = df['fat'].to_numpy()
    x3 = df['weight'].to_numpy()
    
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)
    x3 = np.expand_dims(x3, axis=1)
    
    x = np.concatenate((x1, x2, x3), axis=1)
    
    return x


if __name__ == "__main__":
    neighbors = 11
    print()
    print(f"Number of Neighbors: {neighbors}")
    
    # x = read_data()
   
    # fig = plt.figure()
    
    # plt.scatter(x[:, 0], x[:, 1])
    
    # plt.show() 
    
    x, d_color = make_swiss_roll(500, random_state=0)
    
    # thresholds = np.percentile(d_color, [25, 50, 75])

    # d_color = np.digitize(d_color, bins=thresholds)
    
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=d_color)

    plt.show()
    
    spectral_emb = laplacian_eigenmaps(n_dimensions=2)
    ad_matrix = spectral_emb.adjacency_matrix(x, neighbors)
    eigen_values, eigen_vectors = spectral_emb.eigen_maps()
    
    fig = plt.figure()

    plt.plot(eigen_values, marker='o')
    
    plt.show()
    
    
    scaler = MinMaxScaler()
    eigen_vectors = scaler.fit_transform(eigen_vectors)
    
    kmeans = KMeans(n_clusters=7)
    tr = kmeans.fit_predict(eigen_vectors)    
    
    fig, axs = plt.subplots(1, 2)
    
    axs[0].scatter(eigen_vectors[:, 0], eigen_vectors[:, 1], c=d_color[spectral_emb.n_zero_values:])
    axs[1].scatter(eigen_vectors[:, 0], eigen_vectors[:, 1], c=tr)
    # plt.xlim([-0.2, 0])
    # plt.ylim([-0.15, 0.15])
    plt.show() 
    
    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")

    # ax.scatter3D(eigen_vectors[:, 0], eigen_vectors[:, 1], eigen_vectors[:, 2], 
    #              c=d_color[spectral_emb.n_zero_values:])

    # plt.show()
    