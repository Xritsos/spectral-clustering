import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import calinski_harabasz_score, accuracy_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import sys
sys.path.append('./')
import time

from modules.Clustering import Spectral_Clustering
from load_mnist import load


def split(x, y, train_size):
    y0_train = y[y==0][:train_size]
    y1_train = y[y==1][:train_size]
    y2_train = y[y==2][:train_size]
    y3_train = y[y==3][:train_size]
    y4_train = y[y==4][:train_size]
    y5_train = y[y==5][:train_size]
    y6_train = y[y==6][:train_size]
    y7_train = y[y==7][:train_size]
    y8_train = y[y==8][:train_size]
    y9_train = y[y==9][:train_size]
    
    y_train = np.concatenate((y0_train, y1_train, y2_train, y3_train, y4_train, 
                              y5_train, y6_train, y7_train, y8_train, y9_train))
    
    x0_train = x[y==0][:train_size]
    x1_train = x[y==1][:train_size]
    x2_train = x[y==2][:train_size]
    x3_train = x[y==3][:train_size]
    x4_train = x[y==4][:train_size]
    x5_train = x[y==5][:train_size]
    x6_train = x[y==6][:train_size]
    x7_train = x[y==7][:train_size]
    x8_train = x[y==8][:train_size]
    x9_train = x[y==9][:train_size]
    
    x_train = np.concatenate((x0_train, x1_train, x2_train, x3_train, x4_train, 
                              x5_train, x6_train, x7_train, x8_train, x9_train))
    
    y0_test = y[y==0][train_size:]
    y1_test = y[y==1][train_size:]
    y2_test = y[y==2][train_size:]
    y3_test = y[y==3][train_size:]
    y4_test = y[y==4][train_size:]
    y5_test = y[y==5][train_size:]
    y6_test = y[y==6][train_size:]
    y7_test = y[y==7][train_size:]
    y8_test = y[y==8][train_size:]
    y9_test = y[y==9][train_size:]
    
    y_test = np.concatenate((y0_test, y1_test, y2_test, y3_test, y4_test, 
                             y5_test, y6_test, y7_test, y8_test, y9_test))
    
    x0_test = x[y==0][train_size:]
    x1_test = x[y==1][train_size:]
    x2_test = x[y==2][train_size:]
    x3_test = x[y==3][train_size:]
    x4_test = x[y==4][train_size:]
    x5_test = x[y==5][train_size:]
    x6_test = x[y==6][train_size:]
    x7_test = x[y==7][train_size:]
    x8_test = x[y==8][train_size:]
    x9_test = x[y==9][train_size:]
    
    x_test = np.concatenate((x0_test, x1_test, x2_test, x3_test, x4_test, 
                             x5_test, x6_test, x7_test, x8_test, x9_test))

    return x_train, x_test, y_train, y_test


def read_scores():
    # load best - worst scores' parameters
    df_on = pd.read_csv('./MNIST/heat_on.csv')
    df_off = pd.read_csv('./MNIST/heat_off.csv')
    
    # max calinski
    cal_on_max = df_on.iloc[df_on['Calinski'].idxmax()]
    cal_off_max = df_off.iloc[df_off['Calinski'].idxmax()]
    
    # min calinski
    cal_on_min = df_on.iloc[df_on['Calinski'].idxmin()]
    cal_off_min = df_off.iloc[df_off['Calinski'].idxmin()]
    
    # print(f"Max Calinski Score Heat on: {cal_on_max}")
    # print('===============================================')
    # print()
    # print(f"Max Calinski Score Heat off: {cal_off_max}")
    # print('===============================================')
    # print()
    # print(f"Min Calinski Score Heat on: {cal_on_min}")
    # print('===============================================')
    # print()
    # print(f"Min Calinski Score Heat off: {cal_off_min}")
    
    # max silhoutte
    sil_on_max = df_on.iloc[df_on['Silhouette'].idxmax()]
    sil_off_max = df_off.iloc[df_off['Silhouette'].idxmax()]
    
    # min calinski
    sil_on_min = df_on.iloc[df_on['Silhouette'].idxmin()]
    sil_off_min = df_off.iloc[df_off['Silhouette'].idxmin()]
    
    print(f"Max Silhouette Score Heat on: {sil_on_max}")
    print('===============================================')
    print()
    print(f"Max Silhouette Score Heat off: {sil_off_max}")
    print('===============================================')
    print()
    print(f"Min Silhouette Score Heat on: {sil_on_min}")
    print('===============================================')
    print()
    print(f"Min Silhouette Score Heat off: {sil_off_min}")
    
def run_tests():
    # load MNIST with 100 samples per digit
    x, y = load(150)
    
    # reshape data
    x = x.reshape((x.shape[0], -1))
    
    # scale data
    x = x / 255
    
    # start = time.time()
    
    # create 2D data for visualization
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', 
                perplexity=50, random_state=42)
    
    x_2d = tsne.fit_transform(x)
    
    # tsne_time = round(time.time() - start, 1)
    
    # neighbors = [8, 10, 15, 20, 40, 50]
    # n_dimensions = [2, 3, 5, 8, 10, 15, 20]
    # tau = [5, 10, 20, 40, 100]
    
    # neigh_list = []
    # n_dim_list = []
    # tau_list = []
    # calinski_list = []
    # silh_list = []
    # clustering_time = []
    # for n in neighbors:
    #     for dim in n_dimensions:
    #         for t in tau:
    
    # spectral clustering
    # start = time.time()
    sp_clustering = Spectral_Clustering(n_neighbors=15, rbf_on=True, n_clusters=10, 
                                        n_dimensions=2, t=5)
    
    sp_clustering.embedd(x_2d)
    
    x_embedded = sp_clustering.embedded_data
    
    # split to train - test
    x_train, x_test, y_train, y_test = split(x_embedded, y, train_size=100)
    
    # scale results 
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    cluster_train_labels = sp_clustering.cluster(x_train)
    
    # clustering_time.append(round(time.time() - start, 1))
    
    cluster_test_labels = sp_clustering.predict(x_test)
    
    calinski = calinski_harabasz_score(x_train, cluster_train_labels)
    calinski = round(calinski)
    silh = silhouette_score(x_train, cluster_train_labels, random_state=2)
    silh = round(silh, 3)
    
    print(f"Calinski: {calinski}")
    print(f"Silhouette: {silh}")
    
    # neigh_list.append(n)
    # n_dim_list.append(dim)
    # tau_list.append(t)
    # calinski_list.append(calinski)
    # silh_list.append(silh)

    # df = pd.DataFrame(columns=['Neighbors', 'Dimensions', 'Calinski', 'Tau',
    #                     'Silhouette', 'Runtime_tSNE', 'Runtime_clustering'])
                        
    # df['Neighbors'] = neigh_list
    # df['Dimensions'] = n_dim_list
    # df['Tau'] = tau_list
    # df['Calinski'] = calinski_list
    # df['Silhouette'] = silh_list
    # df['Runtime_tSNE'] = tsne_time * len(silh_list)
    # df['Runtime_clustering'] = clustering_time

    # df.to_csv('./MNIST/heat_on.csv')
    
    # keep only training samples for ploting
    x0 = x_2d[y==0][:100]
    x1 = x_2d[y==1][:100]
    x2 = x_2d[y==2][:100]
    x3 = x_2d[y==3][:100]
    x4 = x_2d[y==4][:100]
    x5 = x_2d[y==5][:100]
    x6 = x_2d[y==6][:100]
    x7 = x_2d[y==7][:100]
    x8 = x_2d[y==8][:100]
    x9 = x_2d[y==9][:100]
    
    x_2d_train = np.concatenate((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9))
    
    # creat mapping for train set
    # clabels = cluster_train_labels * -1
    # clabels[clabels==0] = 0
    # clabels[clabels==-8] = 1
    # clabels[clabels==-7] = 2
    # clabels[clabels==-6] = 3
    # clabels[clabels==-2] = 4
    # clabels[clabels==-5] = 5
    # clabels[clabels==-3] = 6
    # # clabels[clabels==-3] = 7
    # clabels[clabels==-9] = 6
    # clabels[clabels==-4] = 2
    # clabels[clabels==-1] = 8
    
    # # create mapping for test set
    # test_clabels = cluster_test_labels * -1
    # test_clabels[test_clabels==0] = 0
    # test_clabels[test_clabels==-8] = 1
    # test_clabels[test_clabels==-7] = 2
    # test_clabels[test_clabels==-6] = 3
    # test_clabels[test_clabels==-2] = 4
    # test_clabels[test_clabels==-5] = 5
    # test_clabels[test_clabels==-3] = 6
    # test_clabels[test_clabels==-9] = 6
    # test_clabels[test_clabels==-4] = 2
    # test_clabels[test_clabels==-1] = 8
    # # test_clabels[test_clabels==-8] = 5
    
    # # some digits cannot be represented by clusters and we remove them
    # y_test[y_test==9] = 4
    # y_test[y_test==7] = 4
    
    # accuracy = accuracy_score(y_test, test_clabels)
    # print()
    # print('===================')
    # print(f"Accuracy Score: {accuracy * 100}")
    # print('===================')
    # print()    
    
    fig, axs = plt.subplots(1, 2)
        
    scatter_1 = axs[0].scatter(x_2d_train[:, 0], x_2d_train[:, 1], c=y_train, cmap='tab10')
    scatter_2 = axs[1].scatter(x_2d_train[:, 0], x_2d_train[:, 1], c=cluster_train_labels, 
                               cmap='tab10')
     
    axs[0].set_facecolor('black')
    axs[1].set_facecolor('black')
    
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    
    axs[1].set_yticks([])
    axs[1].set_xticks([])

    axs[0].set_title("2D Embedded Data")
    axs[1].set_title("Clustered 2D Data")
    
    legend1 = axs[0].legend(*scatter_1.legend_elements())
    axs[0].add_artist(legend1)
    
    legend2 = axs[1].legend(*scatter_2.legend_elements())
    axs[1].add_artist(legend2)
    
    # plt.suptitle(f'Calinski: {clustering_score} - Silh: {clustering_score_2}')
    
    plt.show()


if __name__ == "__main__":
    
    # read_scores()
    
    run_tests()      
    

    
    