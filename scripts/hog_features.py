# extract training and testing hog features
import os
from os.path import join as oj
import cv2
import imageio
from skimage.feature import hog
import numpy as np


def run(data_dir, labels):
    x_train = []
    x_test = []

    # loop through all the training files
    train_dir = oj(data_dir, 'train')
    for fld in os.listdir(train_dir):
        class_dir = oj(train_dir, fld)
        for filename in os.listdir(class_dir):
            im_path = oj(class_dir, filename)
            # read the image
            im =imageio.imread(im_path)

            # convert image to grayscale
            gray_im = cv2.cvtColor(im.copy(), cv2.COLOR_RGB2GRAY)
            features = hog(gray_im, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[2, 2],
                           block_norm='L2-Hys', visualize=False, transform_sqrt=True)
            features = list(features) + [labels.index(fld)]

            x_train.append(features)

    # repeat for testing data
    test_dir = oj(data_dir, 'test')
    for fld in os.listdir(test_dir):
        class_dir = oj(test_dir, fld)
        for filename in os.listdir(class_dir):
            im_path = oj(class_dir, filename)
            # read the image
            im = imageio.imread(im_path)

            # convert image to grayscale
            gray_im = cv2.cvtColor(im.copy(), cv2.COLOR_RGB2GRAY)

            # extract hog features
            features = hog(gray_im, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[2, 2],
                           block_norm='L2-Hys', visualize=False, transform_sqrt=True)

            # append the labels at the end
            features = list(features) + [labels.index(fld)]

            x_test.append(features)

    # save the outputs
    _ = np.save(oj(data_dir, 'HOG_train'), np.array(x_train))
    _ = np.save(oj(data_dir, 'HOG_test'), np.array(x_test))
