"""Save features from Inception ResNet V2 on the CIFAR dataset without training and using imagenet weights"""
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from os.path import join as oj
import numpy as np


def run(data_dir, labels, input_shape=(75, 75)):
    """Input shape is the resize shape of the CIFAR data, with a minimum (and default) of 75 by 75.
    """
    # load Inception-ResNet-v2 model, with imagenet weights
    # remove the top dense layers, and specify the input shape of image
    # Note: CIFAR image size is too small to use (32 by 32), smallest size allowed is (75 by 75)
    model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
    x = model.output
    predictions = Flatten()(x)
    model = Model(inputs=model.input, outputs=predictions)
    
    # add and optimizer and compile the image
    sgd = optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # create data generators
    train_dgen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
    train_gen = train_dgen.flow_from_directory(
        oj(data_dir, "train"),
        target_size=input_shape,
        batch_size=1,
        shuffle=False,
        class_mode="categorical",
        classes=labels
    )
    
    test_dgen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
    test_gen = test_dgen.flow_from_directory(
        oj(data_dir, "test"),
        target_size=input_shape,
        batch_size=1,
        shuffle=False,
        class_mode="categorical",
        classes= labels
    )

    # predict to extract features
    predict = model.predict_generator(train_gen, steps=train_gen.samples, verbose=1)
    classes = np.reshape(train_gen.classes, (-1, 1))
    train_data = np.hstack((predict, classes))  # concatenate the classes as a last column
    
    predict = model.predict_generator(test_gen, steps=test_gen.samples, verbose=1)
    classes = np.reshape(test_gen.classes, (-1, 1))
    test_data = np.hstack((predict, classes))  # concatenate the classes as a last column
    
    
    # save the features data and the filename list to associate rows to images
    _ = np.save(oj(data_dir, 'resnet2_train'), train_data)
    _ = np.save(oj(data_dir, 'resnet2_train_filenames'), train_gen.filenames)

    _ = np.save(oj(data_dir, 'resnet2_test'), test_data)
    _ = np.save(oj(data_dir, 'resnet2_test_filenames'), test_gen.filenames)
    