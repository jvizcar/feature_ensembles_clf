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
6. run image_features.ipynb to extract HOG features (possibly other features in the future) and add them to the dataframe. Output of this is pickle files of the data that contain the features as well as labels

___

**Data info**

Both the training and testing datasets have a set of files associated with them. 

1) train/test.npy contains the image data (RGB data) for all the images in each dataset. If X contains training data and you want image indexed n then you can obtain it via X[n, :, :, :]. 

2) train/test.csv contains the corresponding class labels in int and string format. For training dataset it also contains the batch each image is associated from (dataset contains 5 batches of 10k images each). 

3) train/test.pkl contains the same information as the .csv files plus the feature columns.
