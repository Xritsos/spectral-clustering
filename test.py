import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.datasets import make_swiss_roll
import pickle
from sklearn.cluster import KMeans
from PIL import Image

from spectral_embedding import laplacian_eigenmaps
from heat_kernel import rbf
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA

import struct


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
    
def read_data():
    
    with open('./mnist.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        x = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        x = x.reshape((size, nrows, ncols))
        
    with open('./mnist_labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        y = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        y = y.reshape((size,))
        
    # batch = unpickle('./cifar/data_batch_1')
    
    # x = batch[b'data']
    # y = np.asarray(batch[b'labels'])
    
    x1 = x[y==0][:50]
    x2 = x[y==1][:50]
    x3 = x[y==2][:50]
    x4 = x[y==3][:50]
    x5 = x[y==4][:50]
    x6 = x[y==5][:50]
    x7 = x[y==6][:50]
    x8 = x[y==7][:50]
    x9 = x[y==8][:50]
    x10 = x[y==9][:50]
    
    # image = x1[0]
    
    # print(image.shape)
    
    # # r = image[:1024].reshape((32, 32))
    # # g = image[1024:2048].reshape((32, 32))
    # # b = image[2048:].reshape((32, 32))
    
    # # image = np.dstack((r, g, b))
    
    # fig = plt.figure()
    
    # plt.imshow(image)
    
    # plt.show()
    
    x = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10))
    
    y1 = y[y==0][:50]
    y2 = y[y==1][:50]
    y3 = y[y==2][:50]
    y4 = y[y==3][:50]
    y5 = y[y==4][:50]
    y6 = y[y==5][:50]
    y7 = y[y==6][:50]
    y8 = y[y==7][:50]
    y9 = y[y==8][:50]
    y10 = y[y==9][:50]
    
    y = np.concatenate((y1, y2, y3, y4, y5, y6, y7, y8, y9, y10))
    
    # im_frame = Image.open('./River/River_3.jpg')
    # x = np.array(im_frame)

    # df = pd.read_csv('./cereal.csv')
    # print(df.columns)

    # x1 = df['calories'].to_numpy()
    # x2 = df['fat'].to_numpy()
    # x3 = df['weight'].to_numpy()
    
    # x1 = np.expand_dims(x1, axis=1)
    # x2 = np.expand_dims(x2, axis=1)
    # x3 = np.expand_dims(x3, axis=1)
    
    # x = np.concatenate((x1, x2, x3), axis=1)
    
    return x, y


if __name__ == "__main__":
    
    x, y = read_data()
    
    x = x.reshape((x.shape[0], 28**2))
    
    # x, y = make_swiss_roll(500, random_state=3)
    # thresholds = np.percentile(y, [25, 50, 75])

    # y = np.digitize(y, bins=thresholds)

    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(x)
    
    neighbors_1 = 5
    spectral_emb_1 = laplacian_eigenmaps(n_dimensions=2)
    ad_matrix_1 = spectral_emb_1.adjacency_matrix(input_data, neighbors_1, 
                                                  rbf_on=True, t=200)
    
    eigen_values_1, eigen_vectors_1 = spectral_emb_1.eigen_maps()
    
    print(eigen_values_1[0])
    
    print(f"Eigen_vectors 1: {eigen_vectors_1.shape}")
    
    scaler = MinMaxScaler()
    eigen_vectors_1 = scaler.fit_transform(eigen_vectors_1)
    
    kmeans_spectral = KMeans(n_clusters=10, random_state=7, n_init=10)
    spectral_labels = kmeans_spectral.fit_predict(eigen_vectors_1)
    
    kmeans = KMeans(n_clusters=10, random_state=7, n_init=10)
    means_labels = kmeans.fit_predict(input_data)
    
    
    fig, axs = plt.subplots(1, 3)
    
    scatter1 = axs[0].scatter(eigen_vectors_1[:, 0], eigen_vectors_1[:, 1], c=y, cmap='tab20')
    scatter2 = axs[1].scatter(eigen_vectors_1[:, 0], eigen_vectors_1[:, 1], c=spectral_labels, cmap='tab10')
    scatter3 = axs[2].scatter(eigen_vectors_1[:, 0], eigen_vectors_1[:, 1], c=means_labels, cmap='tab10')
    
    axs[0].set_title("Data after Eigenmaps to 2D")
    axs[1].set_title("Spectral Clustering Result")
    axs[2].set_title("KMeans Result")
    
    axs[0].set_facecolor('black')
    axs[1].set_facecolor('black')
    axs[2].set_facecolor('black')
    
    legend1 = axs[0].legend(*scatter1.legend_elements())
    axs[0].add_artist(legend1)
    
    # legend2 = axs[1].legend(*scatter2.legend_elements())
    # axs[1].add_artist(legend2)
    
    # legend3 = axs[2].legend(*scatter3.legend_elements())
    # axs[2].add_artist(legend3)
    
    plt.show()
    