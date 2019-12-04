"""Basic implementation of VGG16 in Keras - using transfer learning.
"""
# from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from os.path import join as oj

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(image_shape):
    """Load pretrained vgg16 model with imagenet weights but without the dense layers. Give"""
    # load the VGG16 model from Keras directly
    model = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)

    # imagenet classes are similar to CIFAR10 - by transfer learning logic freeze all layers of original model
    # default is for all of them to be trainable
    for layer in model.layers:
        layer.trainable = False

    # CIFAR 10 has 10 classes - original VGG16 with imagenet has 1000 classes
    # to tailor the features to vote for 10 classes instead add a couple of dense layers with dropout and a softmax
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(10, activation="softmax")(x)

    # create final model
    model_final = Model(inputs=model.input, outputs=predictions)

    return model_final


def generators(data_dir, **kwargs):
    """Handle the creation of flow from directory data generators. Returns both the train and test generators, in our
    implementation of we sample the training dir for both the training and validation dataset.

    Data augmentation and some preprocesssing is encoded.

    Note that datgenerators allow you to select the augmentation methods, preprocessing functions from pre-trained
    models, and (for training) define validation split factor.

    From datagen you can use flow from directory to create generators of the data gens and specify which directory
    to take images from, batch size to use when training, plus more.
    """
    # train and validation are sampled from same data generator - signifiy by using subset parameter
    train_val_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,  # horizontal flip is the only custom augmentation
        preprocessing_function=preprocess_input,  # vgg16 default preprocessing - subtracting the mean color
        validation_split=0.2
    )

    train_gen = train_val_datagen.flow_from_directory(
        oj(data_dir, "train"),
        batch_size=kwargs['batch_size'],
        class_mode="categorical",
        classes=CLASSES,
        target_size=kwargs['image_shape'][:-1],
        subset="training"  # signifiying that this is training gen
    )

    val_gen = train_val_datagen.flow_from_directory(
        oj(data_dir, 'train'),
        batch_size=kwargs['batch_size'],
        class_mode="categorical",
        classes=CLASSES,
        target_size=kwargs['image_shape'][:-1],
        subset="validation"  # signifiying that this is training gen
    )

    # testing data gen is the same as training but with no validation split param
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,  # horizontal flip is the only custom augmentation
        preprocessing_function=preprocess_input,  # vgg16 default preprocessing - subtracting the mean color
    )

    test_gen = test_datagen.flow_from_directory(
        oj(data_dir, 'test'),
        class_mode='categorical',
        classes=CLASSES,
        target_size=kwargs['image_shape'][:-1]
    )

    return train_gen, val_gen, test_gen


def run(data_dir):
    # VGG16 was trained on images of size (256, 256, 3) - Keras allows you to pass a different input image shape
    # so long as you specify it. CIFAR10 images are (32, 32, 3)

    # hyperparameters are passed in as key-word arguments
    kwargs = dict(
        batch_size=16, epochs=5, learning_rate=0.0001, momentum=0.9, image_shape=(32, 32, 3)
    )

    # load the VGG16 model from Keras directly
    model = load_model(image_shape=kwargs['image_shape'])

    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=kwargs['learning_rate'],
                  momentum=kwargs['momentum']), metrics=["accuracy"])

    # create data flow from directoy generators
    train_gen, val_gen, test_gen = generators(data_dir, **kwargs)

    # create checkpoints for saving best model
    os.makedirs('models/', exist_ok=True)
    checkpoint = ModelCheckpoint("models/vgg16_test.h5", monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', save_freq=1)

    # below is example of early stopping, but for now ignore
    # early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    # train model
    model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.samples // kwargs['batch_size'],
        validation_data=val_gen,
        epochs=kwargs['epochs'],
        validation_steps=val_gen.samples // kwargs['batch_size'],
        # callbacks=[checkpoint]
    )





