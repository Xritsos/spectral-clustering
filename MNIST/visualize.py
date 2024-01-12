import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import sys

sys.path.append('./')
from load_mnist import load
from modules.Embedding import Spectral_Embedding

def plot():
    
    x, y = load(100)
    
    x = x.reshape((x.shape[0], -1))
    
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    # spectral embedding
    sp_emb = Spectral_Embedding(n_dimensions=2, neighbors=40, rbf_on=True, t=100)
    values, vectors = sp_emb.transform(x)
    
    vectors = scaler.fit_transform(vectors)
    
    # t-SNE embedding
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', 
                perplexity=10, random_state=42)
    
    x_tsne = tsne.fit_transform(x)
    
    x_tsne = scaler.fit_transform(x_tsne)
    
    fig, axs = plt.subplots(1, 2)
    
    scatter1 = axs[0].scatter(vectors[:, 0], vectors[:, 1], c=y)
    scatter2 = axs[1].scatter(x_tsne[:, 0], x_tsne[:, 1], c=y)
    
    axs[0].set_facecolor('black')
    axs[1].set_facecolor('black')
    
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    
    axs[0].set_title("Spectrtal Embedding 2D")
    axs[1].set_title("t-SNE 2D")
    
    legend1 = axs[0].legend(*scatter1.legend_elements())
    axs[0].add_artist(legend1)
    
    legend2 = axs[1].legend(*scatter2.legend_elements())
    axs[1].add_artist(legend2)
    
    plt.show()



if __name__ == "__main__":
    
    plot()