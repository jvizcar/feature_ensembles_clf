"""predict using a trained VGG16 model on the testing dataset. 

You can use this script to add other analysis additions, such as plotting confusion matrices.
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os.path import join as oj
import numpy as np


def run(data_dir, labels, **kwargs):
    """Test a given model on the testing data."""
    # load a model
    model = load_model(oj('models', kwargs['model_name']))
    
    # create a test data loader
    pred_dgen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
    pred_gen = pred_dgen.flow_from_directory(
        oj(data_dir, "test"),
        target_size=kwargs['image_shape'][:-1],
        batch_size=1,
        shuffle=False,
        class_mode="categorical",
        classes=labels
    )
    predict = model.predict_generator(pred_gen, steps=pred_gen.samples, verbose=1)
    corrects = 0

    for i, c in enumerate(pred_gen.classes):
        pred = predict[i]
        predicted_label = np.argmax(pred)
        if c == predicted_label:
            corrects += 1

    acc = corrects / len(pred_gen.classes)
    print('Testing accuracy: {:1f}'.format(acc * 100))