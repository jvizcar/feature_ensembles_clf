"""functions used in this project"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from skimage.feature import hog
import cv2



LABEL_MAP = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def load_batch(file_path, plot_sample=False):
    """Load a batch of CIFAR-10 data in friendly format. 
    
    :param file_path : str
        the path to the data file to load
    :param plot_sample : bool (default: False)
        plot a random sample of the images
    
    :return X : ndarray
        images in numpy form, to get any image i: X[i, :, :, :]
    :return Y : dataframe
        corresponding int and string labels for the images in X, where row i matches images X[i, :, :, :]
    """
    with open(file_path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    # reshape data to [image index, height, width, channel]
    # to get any image i: X[i, :, :, :]
    X = data[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

    # return corresponding label dataframe with int and string labels
    y_data = {'label': data[b'labels']}
    y_data['label_name'] = [LABEL_MAP[label] for label in y_data['label']]
    Y = pd.DataFrame(data=y_data, columns=['label', 'label_name'])
    
    if plot_sample:
        # randomly select 9 images to plot with their labels
        plot_samples(X, Y)
    return X, Y


def plot_samples(X, Y):
    """Randomly plot CIFAR sample images given formatted numpy image data (X) and its label dataframe (Y). 
    See load_batch(..) above for information about these inputs.
        """
        # randomly select 9 images to plot with their labels
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(5,6))
    ax = ax.ravel()
    for i, idx in enumerate(random.sample(range(0, len(Y)), 9)):
        ax[i].imshow(X[idx, :, :, :], interpolation='bicubic')
        ax[i].set_title(Y.loc[idx, 'label_name'], fontsize=12)
        ax[i].axis('off')
    plt.show()

    
def extract_features(image, feature_type, limit=None):
    """Extract image features from provided RGB image.
    
    :param image : ndarray
        RGB image data
    :param str : feature_type
        type of features to extract
    :param limit : int (default: None)
        limit of image features to return, if None then return as many features as possible
    
    :return features : list
        the list of image descriptors
    """    
    # extract feature
    if feature_type == 'hog':
        # convert image to grayscale
        gray_im = cv2.cvtColor(image.copy() ,cv2.COLOR_RGB2GRAY)
        features = hog(gray_im, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[2, 2],
                     block_norm='L2-Hys', visualize=False, transform_sqrt=True)
        features = list(features)
    else:
        raise Exception('Feature type {} is not a valid descriptor to extract'.format(feature_type))
        
    if limit is not None:
        features = features[:limit]
    return features
