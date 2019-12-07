"""Combine the three feature sets of interest and run a PCA to reduce the dimensionality. Save the new features"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
from skimage.feature import hog
np.random.seed(64)


def composite_features(feature_list, save_path, zscore=True, pca=True, n_components=1000):
    """Combine a set of features and add PCA and Zscore (by choice)"""
    # post and fence problem, initiate the array with first features
    x_train, y_train, x_test, y_test = np.load(feature_list[0], allow_pickle=True)
    train_data = x_train
    test_data = x_test
    
    for feature_path in feature_list[1:]:
        x_train, y_train, x_test, y_test = np.load(feature_path, allow_pickle=True)
        
        # horizontal stack the features
        train_data = np.hstack((train_data, x_train))
        test_data = np.hstack((test_data, x_test))
        
    if zscore:
        # standard scale the features
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    
    if pca:
        # run PCA
        _pca = PCA(n_components=n_components)
        _pca.fit(train_data)
        train_data = _pca.transform(train_data)
        test_data = _pca.transform(test_data)
    
    data = train_data, y_train, test_data, y_test
    _ = np.save(save_path, data)
