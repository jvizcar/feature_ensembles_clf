import numpy as np
import imageio
import os
from os.path import join as oj


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
            im = imageio.imread(im_path)
            features = im.flatten() / 255.

            # flatten the features
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

            features = im.flatten() / 255.

            # append the labels at the end
            features = list(features) + [labels.index(fld)]

            x_test.append(features)

    # save the outputs
    _ = np.save(oj(data_dir, 'flatten_train'), np.array(x_train))
    _ = np.save(oj(data_dir, 'flatten_test'), np.array(x_test))

