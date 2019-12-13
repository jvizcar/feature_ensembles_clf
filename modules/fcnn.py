import numpy as np
from os.path import join as oj
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model


class fcnn:
    def __init__(self, features_path, model_path, train=True, maxepoches=250, learning_rate=0.001):
        """trainable options: all, none, five (last five layers only)"""
        self.num_classes = 10
        self.batch_size = 64
        self.learning_rate = learning_rate
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.maxepoches = maxepoches
        # location to load  (if train = False) or save model (if train = True)
        self.model_path = model_path
        
        # load features
        x_train, y_train, x_test, y_test = np.load(features_path, allow_pickle=True)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # if train is false then model will be loaded from model_path
        if train:
            self.model = self.build_model()
            self.model = self.train(self.model)
        else:
            self.model = load_model(self.model_path)
            
    def build_model(self):
        """Defines a fully connected network with two hidden layers of 100 and 100 neurons respectively with dropout"""
        model = Sequential()

        # add input layer
        model.add(Dense(units=300, input_dim=self.x_train.shape[1], activation='relu'))

        # add dropout
        model.add(Dropout(0.5))

        # add a hidden layer
        model.add(Dense(units=100, input_dim=300, activation='relu'))

        # add the output layer
        model.add(Dense(units=self.num_classes, input_dim=100,activation='softmax'))

        return model
    
    def predict(self, batch_size=50):
        """Predict the class probabilities for the testing data."""
        x = self.x_test        
        predict = self.model.predict(x, batch_size)
        return predict
    
    def train(self, model):
        #training parameters
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        
        # The data, shuffled and split between train and test sets:
        x_train, y_train, x_test, y_test = self.x_train, self.y_train, self.x_test, self.y_test
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        # compile model
        sgd = optimizers.SGD(lr=0.001, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        # early stopping
        early = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='auto')
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=1, 
                                save_best_only=True, save_weights_only=False, mode='auto')
        
        historytemp = model.fit(x_train, y_train, epochs=self.maxepoches, batch_size=batch_size, validation_data=(x_test, y_test),
             callbacks=[checkpoint, early])
        historytemp = historytemp.__dict__
        del historytemp['model']
        _ = np.save(self.model_path.replace('.h5', '.npy'), historytemp) 
        return model