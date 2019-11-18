# CS557-Final_Project
Using reinforcement learning in classification of the CIFAR dataset.

Information about the datasets used can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

___

**Instructions**

1. clone the repo
2. download the [CIFARD-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
3. unzip/extract data into a directory (rename directory to CIFAR10_Data)
4. move CIFAR10_Data dir inside repo
5. run dataset.ipynb to create train.csv, test.csv, train.npy, test.npy files for easier handling of CIRAD10 dataset
    * train.npy and test.npy contain the image data in RGB format
    * train.csv and test.csv contain the corresponding labels for each image in .npy files as well as batch number for the training data
