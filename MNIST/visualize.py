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
    
    neigs = [4, 8, 10, 15, 20, 40]
    taus = [10, 20, 40, 100]
    
    for neig in neigs:
        # spectral embedding
        sp_emb = Spectral_Embedding(n_dimensions=2, neighbors=neig, 
                                    rbf_on=False, t=tau)
        
        values, vectors = sp_emb.transform(x)
        
        vectors = scaler.fit_transform(vectors)
        
        # Spectral Embedding plot
        fig_1, ax = plt.subplots(1)
        
        scatter_1 = ax.scatter(vectors[:, 0], vectors[:, 1], c=y)
        
        ax.set_facecolor('black')
        
        ax.set_yticks([])
        ax.set_xticks([])

        ax.set_title(f'Neighbors: {neig}')
        
        legend1 = ax.legend(*scatter_1.legend_elements())
        ax.add_artist(legend1)
        
        plt.suptitle("Spectrtal Embedding 2D")

        plt.savefig(f'./MNIST/visualization_plots/spectral/{neig}')
            
        plt.close()
    
    
    perps = [5, 10, 20, 30, 40, 50]
    i = 1
    for perp in perps:
        # t-SNE embedding
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', 
                    perplexity=perp, random_state=42)
        
        x_tsne = tsne.fit_transform(x)
        
        x_tsne = scaler.fit_transform(x_tsne)
        
        # t-SNE plot
        fig_2, ax = plt.subplots(1)
        
        scatter_2 = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y)
        
        ax.set_facecolor('black')
        
        ax.set_yticks([])
        ax.set_xticks([])

        ax.set_title(f'Perplexity {perp}')
        
        legend2 = ax.legend(*scatter_2.legend_elements())
        ax.add_artist(legend2)
        
        plt.suptitle("t-SNE 2D")
        
        plt.savefig(f'./MNIST/visualization_plots/tsne/{i}')
        
        i += 1


if __name__ == "__main__":
    
    plot()