import numpy as np
import matplotlib.pyplot as plt

def show(img_tensor, title=None, save_fp=None):
    plt.imshow(np.transpose(img_tensor, (1, 2, 0)))
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if save_fp is not None:
        plt.savefig(save_fp)
    plt.show()
