import numpy as np
from os.path import join as oj
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout


def FCNN(input_features):
    """Defines a fully connected network with two hidden layers of 100 and 100 neurons respectively with dropout"""
    model = Sequential()
    model.add(Dense(300, input_dim=input_features, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
    

def run(data_dir, labels, feature_type):
    """Run a simple fully connected neural network on the feature set provided"""    
    train_data = np.load(oj(data_dir, '{}_train.npy'.format(feature_type)))
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    test_data = np.load(oj(data_dir, '{}_test.npy'.format(feature_type)))
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    
    # get model
    model = FCNN(x_train.shape[1])
    model.fit(x=x_train, y=y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test), shuffle=True)
    
    
    