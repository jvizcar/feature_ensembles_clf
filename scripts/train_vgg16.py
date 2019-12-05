"""Basic implementation of VGG16 in Keras - using transfer learning.
"""
# from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
import tensorflow.keras as keras
import os
from os.path import join as oj

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(**kwargs):
    """Load pretrained vgg16 model with imagenet weights but without the dense layers. Give"""
    weight_decay = 0.0005
    
    # load the VGG16 model from Keras directly
    model = VGG16(include_top=False, weights='imagenet', input_shape=kwargs['image_shape'])

    # imagenet classes are similar to CIFAR10 - by transfer learning logic freeze all layers of original model
    # default is for all of them to be trainable
    if kwargs['allTrainable']:
        for layer in model.layers:
            layer.trainable = True
    elif kwargs['all_frozen']:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:-5]:
            layer.trainable = False

    # CIFAR 10 has 10 classes - original VGG16 with imagenet has 1000 classes
    # to tailor the features to vote for 10 classes instead add a couple of dense layers with dropout and a softmax
    x = model.output
    x = Flatten()(x)
    
    # paper layers add the last layers provided in this example: https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py
    if kwargs['paper_layers']:
        x = Dense(512, kernel_regularizer=regularizers.l2(weight_decay), activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(10, activation='softmax')(x)
    else:
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
    if kwargs['extra_aug']:
        train_val_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,  # horizontal flip is the only custom augmentation
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            preprocessing_function=preprocess_input,  # vgg16 default preprocessing - subtracting the mean color
        )
    else:
        train_val_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,  # horizontal flip is the only custom augmentation
            preprocessing_function=preprocess_input,  # vgg16 default preprocessing - subtracting the mean color
        )

    train_gen = train_val_datagen.flow_from_directory(
        oj(data_dir, "train"),
        batch_size=kwargs['batch_size'],
        class_mode="categorical",
        classes=CLASSES,
        target_size=kwargs['image_shape'][:-1],
        seed=64,
    )

    # validation data gen is the same as training but with no validation split param
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_input,  # vgg16 default preprocessing - subtracting the mean color
    )

    val_gen = val_datagen.flow_from_directory(
        oj(data_dir, 'test'),
        class_mode='categorical',
        classes=CLASSES,
        target_size=kwargs['image_shape'][:-1]
    )

    return train_gen, val_gen


def run(data_dir, **kwargs):
    # VGG16 was trained on images of size (256, 256, 3) - Keras allows you to pass a different input image shape
    # so long as you specify it. CIFAR10 images are (32, 32, 3)
    # create data flow from directoy generators
    train_gen, val_gen = generators(data_dir, **kwargs)
    
    # create checkpoints for saving best model
    os.makedirs('models/', exist_ok=True)
    checkpoint = ModelCheckpoint("models/vgg16_lr_{}_mom_{}_xaug_{}_gitLayers_{}_allFrozen_{}_decay_{}_allTrain_{}.h5".format(
                                     kwargs['learning_rate'], kwargs['momentum'], kwargs['extra_aug'], kwargs['paper_layers'],
                                     kwargs['all_frozen'], kwargs['decay'], kwargs['allTrainable']
                                 ),
                                 monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)


    # load the VGG16 model from Keras directly
    model = load_model(**kwargs)
    
    
    if kwargs['decay']:
        # for use in learning rate decay
        lr_drop = kwargs['lr_drop']
        learning_rate = kwargs['learning_rate']
        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        
        sgd = optimizers.SGD(lr=kwargs['learning_rate'], decay=kwargs['lr_decay'], momentum=kwargs['momentum'], nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        # train model
        model.fit_generator(
            train_gen,
            steps_per_epoch=train_gen.samples // kwargs['batch_size'],
            validation_data=val_gen,
            epochs=kwargs['epochs'],
            validation_steps=val_gen.samples // kwargs['batch_size'],
            callbacks=[checkpoint, reduce_lr],
            shuffle=True
        )
    else:
        # training the model without learning rate decay
        sgd = optimizers.SGD(lr=kwargs['learning_rate'], momentum=kwargs['momentum'])
        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

        # train model
        model.fit_generator(
            train_gen,
            steps_per_epoch=train_gen.samples // kwargs['batch_size'],
            validation_data=val_gen,
            epochs=kwargs['epochs'],
            validation_steps=val_gen.samples // kwargs['batch_size'],
            callbacks=[checkpoint],
            shuffle=True
        )
        
    return model





