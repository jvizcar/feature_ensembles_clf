"""Basic implementation for testing inception - using transfer learning.
"""
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.application import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
import os
from os.path import join as oj

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(**kwargs):
    """Load pretrained vgg16 model with imagenet weights but without the dense layers. Give"""
    weight_decay = 0.0005
    
    # load the VGG16 model from Keras directly
    model = InceptionResNetV2(include_top=False, weights='imagenet') #, input_shape=kwargs['image_shape'])

    # CIFAR 10 has 10 classes - original VGG16 with imagenet has 1000 classes
    # to tailor the features to vote for 10 classes instead add a couple of dense layers with dropout and a softmax
    
#     for layer in model.layers[:-5]:
#             layer.trainable = False
            
#     x = model.output
#     x = Flatten()(x)
#     x = Dense(512, kernel_regularizer=regularizers.l2(weight_decay), activation="relu")(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     predictions = Dense(10, activation='softmax')(x)
    
    for i in range(5):  # pop the last 5 layers
        model.layers.pop()
        
    # set up the output of the model
    output = model.output
    output = GlobalAveragePooling2D(name="avg_pool")(output)
    output = Dense(1024, activation='relu', kernel_regularizer=l2(l=kwargs['learning_rate']))(output)
    predictions = Dense(10, activation='softmax')(output)

    # create final model
    model = Model(inputs=model.input, outputs=predictions)

    return model


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
        target_size=(299, 299),
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
        target_size=(299, 299)
    )

    return train_gen, val_gen


def run(data_dir, **kwargs):
    # VGG16 was trained on images of size (256, 256, 3) - Keras allows you to pass a different input image shape
    # so long as you specify it. CIFAR10 images are (32, 32, 3)
    # create data flow from directoy generators
    train_gen, val_gen = generators(data_dir, **kwargs)
    
    # create checkpoints for saving best model
    os.makedirs('models/', exist_ok=True)
    checkpoint = ModelCheckpoint("models/inception_lr_{}_mom_{}_xaug_{}_gitLayers_{}_allFrozen_{}_decay_{}_allTrain_{}.h5".format(
                                     kwargs['learning_rate'], kwargs['momentum'], kwargs['extra_aug'], kwargs['paper_layers'],
                                     kwargs['all_frozen'], kwargs['decay'], kwargs['allTrainable']
                                 ),
                                 monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    
    # early stopping
    early = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')

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
            callbacks=[checkpoint, early],
            shuffle=True
        )
        
    return model





