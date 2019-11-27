"""Run main script
"""
import argparse
from scripts import load_cifar_data


if __name__ == '__main__':
    # load data -- saves I_train.npy, I_test.npy, Y_train.npy, and Y_test.npy to data dir
    load_cifar_data.run()

    # run preprocess
    # run model training
    # run model testing
    # run plotting results