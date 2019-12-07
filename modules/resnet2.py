from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
import os
from os.path import join as oj
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras.regularizers import l2
from modules import utils
from tensorflow.keras.models import Sequential

class resnet2:
    def __init__(self, train=True, maxepoches=250, model_path='models/resnet2.h5'):
        """trainable options: all, none, five (last five layers only)"""
        self.num_classes = 10
        self.x_shape = [75,75,3]
        self.maxepoches = maxepoches
        self.weight_decay = 0.0005
        self.model_path = model_path
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.batch_size = 64
        self.learning_rate = 0.001
        
        # load the cifar10 data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # resize the images
        x_shape = self.x_shape
        train_resized = np.zeros((x_train.shape[0], x_shape[0], x_shape[1], x_shape[2]), dtype=np.uint8)
        for i in range(x_train.shape[0]):
            train_resized[i] = (resize(x_train[i], (x_shape[0], x_shape[1])) * 255).astype(np.uint8)
        self.x_train = train_resized
        
        test_resized = np.zeros((x_test.shape[0], x_shape[0], x_shape[1], x_shape[2]), dtype=np.uint8)
        for i in range(x_test.shape[0]):
            test_resized[i] = (resize(x_test[i], (x_shape[0], x_shape[1])) * 255).astype(np.uint8)
        self.x_test = test_resized
        
        self.y_train = y_train
        self.y_test = y_test
        
        if train:
            self.model = self.build_model()
            self.model = self.train(self.model)
        else:
            self.model = load_model(self.model_path)

    def build_model(self):
        # build a vgg16 model, custom by out team
        model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.x_shape)
        
        # CIFAR 10 has 10 classes - original VGG16 with imagenet has 1000 classes
        # to tailor the features to vote for 10 classes instead add a couple of dense layers with dropout and a softmax
        for layer in model.layers:
            layer.trainable = True
        
        for i in range(5):  # pop the last 5 layers
            model.layers.pop()
        
        # set up the output of the model
        output = model.output
        output = GlobalAveragePooling2D(name="avg_pool")(output)
        output = Dense(1024, activation='relu', kernel_regularizer=l2(l=0.001))(output)
        predictions = Dense(10, activation='softmax')(output)
        
        model = Model(inputs=model.input, outputs=predictions)
        return model
    
    def train(self, model):
        #training parameters
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = (self.x_train, self.y_train), (self.x_test, self.y_test)
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
                
        data_gen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,  # horizontal flip is the only custom augmentation
            preprocessing_function=preprocess_input
        ) 
        
        test_gen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=preprocess_input            
        )
        
        data_gen.fit(x_train)
        test_gen.fit(x_test)
        
        # compile the model
        sgd = optimizers.SGD(lr=learning_rate, momentum=0.9)
        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    
        # early stopping
        early = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='auto')
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=1, 
                                save_best_only=True, save_weights_only=False, mode='auto')
        
        historytemp = model.fit_generator(
            data_gen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=self.maxepoches, 
            validation_data=test_gen.flow(x_test, y_test, shuffle=False),
            callbacks=[checkpoint, early],
            validation_steps=x_test.shape[0] // batch_size)
        historytemp = historytemp.__dict__
        del historytemp['model']
        _ = np.save(self.model_path.replace('.h5', '.npy'), historytemp) 
        return model
    
    def predict(self, batch_size=32):
        """Predict the class probabilities for the testing data."""
        x = self.x_test
        data_gen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=preprocess_input
        )
        data_gen.fit(x)
        
        predict = self.model.predict(data_gen.flow(x, batch_size=batch_size, shuffle=False))
        return predict
    
    def extract_features(self, x, batch_size=64):
        # concatenate all the data to extract features
        
        # modify the model by removing the softmax layers
        new_model = Model(inputs=self.model.get_input_at(0), outputs=self.model.layers[-2].get_output_at(0)) 
            
        sgd = optimizers.SGD(lr=0.001, momentum=0.9)
        new_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        data_gen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=preprocess_input
        )
        data_gen.fit(x)
        
        features = new_model.predict(data_gen.flow(x, batch_size=batch_size, shuffle=False))
        
        # stack the labels as a column
        return features