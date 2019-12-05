"""Run main script
"""
import argparse
from scripts import load_cifar10, train_vgg16, hog_features, tests, flatten_features
import cv2
from matplotlib import pyplot as plt
import numpy as np

# train and test dir will be saved in DATA_DIR - if already there then it will not reload
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
DATA_DIR = 'data/'
TESTS = {'hog': False, 'vgg16': False, 'flatten': True}

if __name__ == '__main__':
    # load CIFAR10 data
    # load_cifar10.run(DATA_DIR, LABELS)

    # train vgg16 model
    # train_vgg16.run(data_dir=DATA_DIR)

    # extract histogram of oriented gradient features
    # hog_features.run(DATA_DIR, LABELS)

    # extract flatten pixel features
    # flatten_features.run(DATA_DIR, LABELS)

    # test hog features
    tests.run(DATA_DIR, TESTS)




