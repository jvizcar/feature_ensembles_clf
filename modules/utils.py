"""Set of useful functions used in project"""
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set(color_codes=True)
from modules import resnet2, vgg16, cifar_vgg, utils, fcnn
from scripts import hog_features, flatten_features, composite_features
import matplotlib.pyplot as plt

LABELS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


def calculate_accuracy(probabilities, true_y):
    """Probabilities contain an array of (n x m) with n being samples and m being class. It is an output from predict method
    from model classes. True y is an 1d array containing the int labels. This function returns the accuracy of the model given
    by correct classification / total test samples.
    """
    correct_count = 0
    n = len(true_y)
    
    for i in range(n):
        if true_y[i] == np.argmax(probabilities[i]):
            correct_count += 1
    accuracy = correct_count / n
    return accuracy


def save_model_features(model, save_path):
    """Given a loaded model, invoke the extract features functions and save it to numpy array in (x_train, y_train, x_test,
    y_test) format"""
    train_data = model.extract_features(model.x_train)
    test_data = model.extract_features(model.x_test)
    
    data = train_data, model.y_train, test_data, model.y_test
    _ = np.save(save_path, data)
    

def plot_confusion_matrix(model, labels=LABELS, title='', save_path=None):
    """y_true shoud be a list of int labels, while y_pred should be the same"""
    pred_y = model.predict()
    y_true =  model.y_test

    temp = []
    for i in range(pred_y.shape[0]):
        temp.append(np.argmax(pred_y[i]))
    y_pred = temp
    
    fig = plt.figure(figsize=(17, 17))
    ax = fig.add_subplot(111)
    mat = confusion_matrix(y_true=y_true, y_pred=y_pred).astype(float)
    sns.heatmap(mat.T, square=True, annot=True, fmt='0.0f',
                xticklabels=LABELS, # Labels
                yticklabels=LABELS, # Labels
                vmin = 0, cmap='coolwarm', cbar=False,
                ax = ax
               )
    plt.yticks(rotation=45, fontsize=24)
    plt.xticks(rotation=45, fontsize=24)
    plt.xlabel('True', fontsize=24)
    plt.ylabel('Predicted', fontsize=24)
    plt.title(title, fontsize=24)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()