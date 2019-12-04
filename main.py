"""Run main script
"""
import argparse
from scripts import load_cifar10, train_vgg16

# train and test dir will be saved in DATA_DIR - if already there then it will not reload
DATA_DIR = 'data/'


if __name__ == '__main__':
    # load CIFAR10 data
    # load_cifar10.run(save_dir=DATA_DIR)

    # train vgg16 model
    train_vgg16.run(data_dir=DATA_DIR)


    # model_train.run()

    # run model testing

    # run plotting results