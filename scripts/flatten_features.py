import numpy as np
from tensorflow.keras.datasets import cifar10


def flatten_features(save_path):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_temp = []
    test_temp = []

    for i in range(x_train.shape[0]):
        features = x_train[i].flatten() / 255.
        train_temp.append(features)
        
    for i in range(x_test.shape[0]):
        features = x_test[i].flatten() / 255.
        test_temp.append(features)

    data = np.array(train_temp), y_train, np.array(test_temp), y_test

    # save the outputs
    _ = np.save(save_path, data)