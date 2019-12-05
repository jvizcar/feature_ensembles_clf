# run testing using different feature sets and approaches
from sklearn.neural_network import MLPClassifier
import numpy as np
from os.path import join as oj


def run(data_dir, tests):
    if tests['hog']:
        # run MLP via sklearn on the hog features
        # load the data
        train_data = np.load(oj(data_dir, 'HOG_train.npy'))
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        test_data = np.load(oj(data_dir, 'HOG_test.npy'))
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=64)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print(score)

    if tests['flatten']:
        # run MLP via sklearn on the hog features
        # load the data
        train_data = np.load(oj(data_dir, 'flatten_train.npy'))
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        test_data = np.load(oj(data_dir, 'flatten_test.npy'))
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=64)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print(score)