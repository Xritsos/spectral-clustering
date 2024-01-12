import numpy as np
import glob
from PIL import Image
from matplotlib import pyplot as plt



def load(n_images):
    
    images = np.zeros((10, 64, 64, 3), dtype=np.int64)
    for i, image in enumerate(glob.glob('./River/images/*')):
        im_frame = Image.open(image)
        x = np.array(im_frame)
        
        images[i, :, :, :] = x
        
    return images[:n_images, :, :, :]
