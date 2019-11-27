"""Read the raw CIFAR-10 data and format into training and testing numpy arrays for the images and the labels.
4 output files are created in run(): I_train.npy, I_test.npy, Y_train.npy, and Y_test.npy"""
import numpy as np
import pickle


def run():
    # load and format the raw training data
    I_train = np.empty((0, 32, 32, 3), dtype=np.uint8)
    Y_train = []

    for i in range(1, 6):
        with open('data/data_batch_{}'.format(i), 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            # create the training_im_data array for batch
            batch_imgs = data[b'data'].reshape(data[b'data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1)

            # concatenate the batch to training array
            I_train = np.concatenate((I_train, batch_imgs))
            Y_train.extend(list(data[b'labels']))

    # load and format the testing data
    I_test = np.empty((0, 32, 32, 3), dtype=np.uint8)
    Y_test = []

    with open('data/test_batch', 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        # create the training_im_data array for batch
        batch_imgs = data[b'data'].reshape(data[b'data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1)

        # concatenate the batch to training array
        I_test = np.concatenate((I_test, batch_imgs))
        Y_test.extend(list(data[b'labels']))

    # save the training and testing image and label arrays to file
    _ = np.save('data/I_train', I_train)
    _ = np.save('data/I_test', I_test)
    _ = np.save('data/Y_train', np.array(Y_train))
    _ = np.save('data/Y_test', np.array(I_test))

