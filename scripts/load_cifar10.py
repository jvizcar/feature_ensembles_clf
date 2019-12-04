"""Read the raw CIFAR-10 data and save to file.
Data is saved as data/train/class1/ and data/test/class2 as pngs
"""
import numpy as np
import pickle
import os
from os.path import join as oj
import imageio

# CIFAR10 labels to map int labels (list index) to string labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def run(save_dir):
    train_dir = oj(save_dir, 'train')
    if not os.path.isdir(train_dir):
        # load and format the raw training data
        im_train = np.empty((0, 32, 32, 3), dtype=np.uint8)
        y_train = []

        # concatenate information from the traning batch files
        for i in range(1, 6):
            with open('data/data_batch_{}'.format(i), 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                # create the training_im_data array for batch
                batch_imgs = data[b'data'].reshape(data[b'data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1)

                # concatenate the batch to training array
                im_train = np.concatenate((im_train, batch_imgs))
                y_train.extend(list(data[b'labels']))

        # loop through the images and labels and save
        label_counts = {label: 0 for label in labels}  # getting counts in this fashion avoids repeating
        for i in range(len(y_train)):
            class_name = labels[y_train[i]]

            # create class dir if needed
            im_dir = oj(train_dir, class_name)
            os.makedirs(im_dir, exist_ok=True)

            # image naming convention ex: train_cat1.png, train_cat2.png, ..., train_catn.png
            n = str(label_counts[class_name] + 1)
            label_counts[class_name] += 1
            im_path = oj(im_dir, 'train_' + class_name + n + '.png')

            # save image as png
            imageio.imwrite(im_path, im_train[i], format='PNG-PIL')
    else:
        print('train dir already present, not re-creating files')

    # repeat for testing dataset
    test_dir = oj(save_dir, 'test')
    if not os.path.isdir(test_dir):
        with open('data/test_batch', 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            # create the training_im_data array for batch
            im_test = data[b'data'].reshape(data[b'data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
            y_test = list(data[b'labels'])

        # loop through the images and labels and save
        label_counts = {label: 0 for label in labels}  # getting counts in this fashion avoids repeating
        for i in range(len(y_test)):
            class_name = labels[y_test[i]]

            # create class dir if needed
            im_dir = oj(test_dir, class_name)
            os.makedirs(im_dir, exist_ok=True)

            # image naming convention ex: train_cat1.png, train_cat2.png, ..., train_catn.png
            n = str(label_counts[class_name] + 1)
            label_counts[class_name] += 1
            im_path = oj(im_dir, 'test_' + class_name + n + '.png')

            # save image as png
            imageio.imwrite(im_path, im_test[i], format='PNG-PIL')
    else:
        print('test dir already present, not re-creating files')



