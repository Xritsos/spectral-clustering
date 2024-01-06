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
    
    x1 = x[y==0][:100]
    x2 = x[y==1][:100]
    x3 = x[y==2][:100]
    x4 = x[y==3][:100]
    x5 = x[y==4][:100]
    x6 = x[y==5][:100]
    x7 = x[y==6][:100]
    x8 = x[y==7][:100]
    x9 = x[y==8][:100]
    x10 = x[y==9][:100]
    
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
    
    y1 = y[y==0][:100]
    y2 = y[y==1][:100]
    y3 = y[y==2][:100]
    y4 = y[y==3][:100]
    y5 = y[y==4][:100]
    y6 = y[y==5][:100]
    y7 = y[y==6][:100]
    y8 = y[y==7][:100]
    y9 = y[y==8][:100]
    y10 = y[y==9][:100]
    
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
    
    x = x.reshape((1000, 28**2))
    
    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(x)
    
    pca = PCA(n_components=2)
    init_data_vis = pca.fit_transform(input_data)
    
    neighbors = 5
    spectral_emb = laplacian_eigenmaps(n_dimensions=2)
    ad_matrix = spectral_emb.adjacency_matrix(input_data, neighbors)
    
    eigen_values, eigen_vectors = spectral_emb.eigen_maps()
    
    # scaler = MinMaxScaler()
    # eigen_vectors = scaler.fit_transform(eigen_vectors)
    
    kmeans_spectral = KMeans(n_clusters=10, random_state=0, n_init=10)
    spectral_labels = kmeans_spectral.fit_predict(eigen_vectors)
    
    fig, axs = plt.subplots(1, 2)
    
    axs[0].scatter(init_data_vis[:, 0], init_data_vis[:, 1], c=y)
    axs[1].scatter(init_data_vis[:, 0], init_data_vis[:, 1], c=spectral_labels)
    
    axs[0].set_title("Initial Data")
    axs[1].set_title("Spectral Clustering")
    
    axs[0].set_facecolor('black')
    axs[1].set_facecolor('black')
    
    plt.show()
    
    # kmeans_spectral = KMeans(n_clusters=3, random_state=0, n_init=10)
    # spectral_labels = kmeans_spectral.fit_predict(eigen_vectors)
    
    # spectral_labels = spectral_labels.reshape((image.shape[:2]))
    
    # kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    # labels = kmeans.fit_predict(input_data)
    
    # labels = labels.reshape((image.shape[:2]))
    
    
    # # spectral_labels[spectral_labels!=4] = 0
    # # labels[labels!=1] = 0
    
    
    # fig, axs = plt.subplots(1, 3)
    
    # axs[0].imshow(image)
    # axs[1].imshow(spectral_labels)
    # axs[2].imshow(labels)
    
    # axs[0].set_title("RGB Image")
    # axs[1].set_title("Spectral Clustering")
    # axs[2].set_title("KMeans")
    
    # plt.show()
    
    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")

    # ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=d_color)

    # plt.show()
    
    
    
    # fig = plt.figure()

    # plt.plot(eigen_values, marker='o')
    
    # plt.show()
    
    # fig, axs = plt.subplots(1, 2)
    
    # axs[0].scatter(eigen_vectors[:, 0], eigen_vectors[:, 1], c=d_color[spectral_emb.n_zero_values:])
    # axs[1].scatter(sk_emb[:, 0], sk_emb[:, 1], c=d_color)
    # # plt.xlim([-0.2, 0])
    # # plt.ylim([-0.15, 0.15])
    # plt.show() 
  
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1, projection ="3d")
    # ax.scatter3D(eigen_vectors[:, 0], eigen_vectors[:, 1], eigen_vectors[:, 2], 
    #              c=d_color[spectral_emb.n_zero_values:])
    
    # ax = fig.add_subplot(1, 2, 2, projection ="3d")    
    # ax.scatter3D(sk_emb[:, 0], sk_emb[:, 1], sk_emb[:, 2], 
    #              c=d_color)

    # plt.show()
    