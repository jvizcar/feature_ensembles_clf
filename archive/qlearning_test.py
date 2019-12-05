"""simple script for testing q-learning for classification on two clases"""
from modules.qlearning_clf import QLearningClassifier
import pickle

if __name__ == '__main__':
    # read the data
    with open('CIFAR10_Data/train_2class.pkl', 'r') as fp:
        df = pickle.load(fp)

    with open('CIFAR10_Data/test_2class.pkl', 'r') as fp:
        df_test = pickle.load(fp)

    # train with 10 epochs
    game = QLearningClassifier(df)
    game.train(epochs=10)

    # test
    test_acc = game.test(df_test)
    print('Testing accuracy = {}'.format(test_acc * 100))