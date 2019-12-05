"""Extract VGG16 features from last dense layer (not softmax layer)"""
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from os.path import join as oj
import numpy as np


def run(data_dir, labels, **kwargs):
    # load the model
    model = load_model(oj('models', kwargs['model_name']))

    
    # create new model with all but the last three layers (NOTE that this only works with models
    # built in the Github paper fashion, since the last three layers are removed: softmax, dropout, and 
    # batch normalization)
    new_model = Sequential()
    for layer in model.layers[:-3]: # just exclude last layer from copying
        new_model.add(layer)   
    
    sgd = optimizers.SGD(lr=0.001, momentum=0.9)
    new_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # create a data generator for the training and testing data to quickly generate the features via prediction
    train_dgen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
    train_gen = train_dgen.flow_from_directory(
        oj(data_dir, "train"),
        target_size=kwargs['image_shape'][:-1],
        batch_size=1,
        shuffle=False,
        class_mode="categorical",
        classes=labels
    )
    predict = new_model.predict_generator(train_gen, steps=train_gen.samples, verbose=1)
    classes = np.reshape(train_gen.classes, (-1, 1))
    train_data = np.hstack((predict, classes))
    _ = np.save(oj(data_dir, 'vgg16_train'), train_data)
    _ = np.save(oj(data_dir, 'vgg16_train_filenames'), train_gen.filenames)    

    # repeat for testing data
    test_dgen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
    test_gen = test_dgen.flow_from_directory(
        oj(data_dir, "test"),
        target_size=kwargs['image_shape'][:-1],
        batch_size=1,
        shuffle=False,
        class_mode="categorical",
        classes=labels
    )
    predict = new_model.predict_generator(test_gen, steps=test_gen.samples, verbose=1)
    classes = np.reshape(test_gen.classes, (-1, 1))
    test_data = np.hstack((predict, classes))
    _ = np.save(oj(data_dir, 'vgg16_test'), test_data)
    _ = np.save(oj(data_dir, 'vgg16_test_filenames'), test_gen.filenames)
