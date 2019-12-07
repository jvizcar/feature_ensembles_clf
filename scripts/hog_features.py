# extract training and testing hog features
import cv2
from skimage.feature import hog
import numpy as np
from tensorflow.keras.datasets import cifar10


def hog_features(save_path):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_temp = []
    test_temp = []

    for i in range(x_train.shape[0]):
        # convert image to grayscale
        gray_im = cv2.cvtColor(x_train[i].copy(), cv2.COLOR_RGB2GRAY)
        features = hog(gray_im, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[2, 2],
                       block_norm='L2-Hys', visualize=False, transform_sqrt=True)

        train_temp.append(features)
        
    for i in range(x_test.shape[0]):
        # convert image to grayscale
        gray_im = cv2.cvtColor(x_test[i].copy(), cv2.COLOR_RGB2GRAY)
        features = hog(gray_im, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[2, 2],
                       block_norm='L2-Hys', visualize=False, transform_sqrt=True)

        test_temp.append(features)

    data = np.array(train_temp), y_train, np.array(test_temp), y_test

    # save the outputs
    _ = np.save(save_path, data)
