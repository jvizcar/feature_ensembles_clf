import numpy as np
from os.path import join as oj
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def FCNN(dim):
    """Defines a fully connected network with two hidden layers of 100 and 100 neurons respectively with dropout"""
    model = Sequential()
    
    # add input layer
    model.add(Dense(units=300, input_dim=dim, activation='relu'))
    
    # add dropout
    model.add(Dropout(0.5))
    
    # add a hidden layer
    model.add(Dense(units=100, input_dim=300, activation='relu'))
        
    # add the output layer
    model.add(Dense(units=10, input_dim=100,activation='softmax'))
    
    return model
    

def run(data_dir, labels, feature_type):
    """Run a simple fully connected neural network on the feature set provided"""    
    train_data = np.load(oj(data_dir, '{}_train.npy'.format(feature_type)))
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    
    # one-hot encode the results
    y_train = keras.utils.to_categorical(y_train)
    
    test_data = np.load(oj(data_dir, '{}_test.npy'.format(feature_type)))
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    y_test = keras.utils.to_categorical(y_test)
    
    # get model
    model = FCNN(x_train.shape[1])
    sgd = optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # create callbacks
    checkpoint = ModelCheckpoint("models/FCNN_{}.h5".format(feature_type),
                                 monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    
    # early stopping
    early = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
    
    history = model.fit(x_train, y_train, epochs=150, batch_size=128, validation_data=(x_test, y_test),
             callbacks=[checkpoint, early])
    return history
    
    
    