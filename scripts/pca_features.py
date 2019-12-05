"""Combine the three feature sets of interest and run a PCA to reduce the dimensionality. Save the new features"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
import imageio
from skimage.feature import hog
from os.path import join as oj
from tqdm import tqdm
np.random.seed(64)


def run(data_dir, n_components=500):
    """Combine all the features and normalize the features"""
    # load vgg16 features and associated filepaths
    train_vgg16 = np.load('data/vgg16_train.npy')
    test_vgg16 = np.load('data/vgg16_test.npy')
    train_filenames = np.load('data/vgg16_train_filenames.npy')
    test_filenames = np.load('data/vgg16_test_filenames.npy')
    
    # loop through each of the images in train and test and recreate the hog and flatten features
    # to append at the end.
    train_fts = []
    test_fts = []
    
    for filename in tqdm(list(train_filenames)):
        im_path = oj(data_dir, 'train', filename)
        # read the image
        im =imageio.imread(im_path)
        # convert image to grayscale
        gray_im = cv2.cvtColor(im.copy(), cv2.COLOR_RGB2GRAY)
        features = hog(gray_im, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[2, 2],
                       block_norm='L2-Hys', visualize=False, transform_sqrt=True)
        
        # append flatten features
        features = list(features) + list(im.flatten() / 255.)
        train_fts.append(features)
    
    # horizontal stack the vgg16 features to the hog and flatten features
    train_fts = np.hstack((
        np.array(train_fts), train_vgg16[:, :-1]
    ))
    
    # repeat for testing
    for filename in tqdm(list(test_filenames)):
        im_path = oj(data_dir, 'test', filename)
        # read the image
        im =imageio.imread(im_path)
        # convert image to grayscale
        gray_im = cv2.cvtColor(im.copy(), cv2.COLOR_RGB2GRAY)
        features = hog(gray_im, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[2, 2],
                       block_norm='L2-Hys', visualize=False, transform_sqrt=True)
        
        features = list(features) + list(im.flatten() / 255.)
        test_fts.append(features)
    
    test_fts = np.hstack((
        np.array(test_fts), test_vgg16[:, :-1]
    ))
    
    # standard scale the features
    scaler = StandardScaler()
    scaler.fit(train_fts)
    train_fts = scaler.transform(train_fts)
    test_fts = scaler.transform(test_fts)
    
    # run PCA
    pca = PCA(n_components=n_components)
    pca.fit(train_fts)
    train_fts = pca.transform(train_fts)
    test_fts = pca.transform(test_fts)
    
    # add the labels
    x_train = np.hstack((train_fts, train_vgg16[:,-1].reshape((-1,1))))
    x_test = np.hstack((test_fts, test_vgg16[:,-1].reshape((-1,1))))
    
    # save the outputs
    _ = np.save(oj(data_dir, 'pca_train'), x_train)
    _ = np.save(oj(data_dir, 'pca_test'), x_test)