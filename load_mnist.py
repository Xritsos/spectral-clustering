import numpy as np
import struct


def load(n_samples):
    
    with open('./mnist_labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        y = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        y = y.reshape((size,))
    
    with open('./mnist.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        x = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        x = x.reshape((size, nrows, ncols))
        
    x1 = x[y==0][:n_samples]
    x2 = x[y==1][:n_samples]
    x3 = x[y==2][:n_samples]
    x4 = x[y==3][:n_samples]
    x5 = x[y==4][:n_samples]
    x6 = x[y==5][:n_samples]
    x7 = x[y==6][:n_samples]
    x8 = x[y==7][:n_samples]
    x9 = x[y==8][:n_samples]
    x10 = x[y==9][:n_samples]
    
    x = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10))
    
    y1 = y[y==0][:n_samples]
    y2 = y[y==1][:n_samples]
    y3 = y[y==2][:n_samples]
    y4 = y[y==3][:n_samples]
    y5 = y[y==4][:n_samples]
    y6 = y[y==5][:n_samples]
    y7 = y[y==6][:n_samples]
    y8 = y[y==7][:n_samples]
    y9 = y[y==8][:n_samples]
    y10 = y[y==9][:n_samples]
    
    y = np.concatenate((y1, y2, y3, y4, y5, y6, y7, y8, y9, y10))
    
    return x, y

