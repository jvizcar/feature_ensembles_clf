"""functions used in this project"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from skimage.feature import hog
import cv2



LABEL_MAP = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def load_batch(file_path, plot_sample=False):
    """Load a batch of CIFAR-10 data in friendly format. 
    
    :param file_path : str
        the path to the data file to load
    :param plot_sample : bool (default: False)
        plot a random sample of the images
    
    :return X : ndarray
        images in numpy form, to get any image i: X[i, :, :, :]
    :return Y : dataframe
        corresponding int and string labels for the images in X, where row i matches images X[i, :, :, :]
    """
    with open(file_path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    # reshape data to [image index, height, width, channel]
    # to get any image i: X[i, :, :, :]
    X = data[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

    # return corresponding label dataframe with int and string labels
    y_data = {'label': data[b'labels']}
    y_data['label_name'] = [LABEL_MAP[label] for label in y_data['label']]
    Y = pd.DataFrame(data=y_data, columns=['label', 'label_name'])
    
    if plot_sample:
        # randomly select 9 images to plot with their labels
        plot_samples(X, Y)
    return X, Y


def plot_samples(X, Y):
    """Randomly plot CIFAR sample images given formatted numpy image data (X) and its label dataframe (Y). 
    See load_batch(..) above for information about these inputs.
        """
        # randomly select 9 images to plot with their labels
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(5,6))
    ax = ax.ravel()
    for i, idx in enumerate(random.sample(range(0, len(Y)), 9)):
        ax[i].imshow(X[idx, :, :, :], interpolation='bicubic')
        ax[i].set_title(Y.loc[idx, 'label_name'], fontsize=12)
        ax[i].axis('off')
    plt.show()

    
def extract_features(image, feature_type, limit=None):
    """Extract image features from provided RGB image.
    
    :param image : ndarray
        RGB image data
    :param str : feature_type
        type of features to extract
    :param limit : int (default: None)
        limit of image features to return, if None then return as many features as possible
    
    :return features : list
        the list of image descriptors
    """    
    # extract feature
    if feature_type == 'hog':
        # convert image to grayscale
        gray_im = cv2.cvtColor(image.copy() ,cv2.COLOR_RGB2GRAY)
        features = hog(gray_im, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[2, 2],
                     block_norm='L2-Hys', visualize=False, transform_sqrt=True)
        features = list(features)
    else:
        raise Exception('Feature type {} is not a valid descriptor to extract'.format(feature_type))
        
    if limit is not None:
        features = features[:limit]
    return features


class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0: return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = self.items()
        compare = lambda x, y:  sign(y[1] - x[1])
        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y ):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x,y = y,x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in y.items():
            self[key] += value

    def __add__( self, y ):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__( self, y ):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend