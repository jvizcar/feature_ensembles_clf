"""Set of useful functions used in project"""
import numpy as np


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