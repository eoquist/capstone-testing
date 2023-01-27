# using this site as a reference/guide
# https://nextjournal.com/gkoehler/digit-recognition-with-keras

# imports for array-handling and plotting
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential, load_model
from keras.datasets import mnist
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('agg')

# let's keep our keras backend tensorflow quiet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# for testing on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# keras imports for the dataset and building our neural network

# handy function that splits MNIST data into training and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# Let's inspect a few examples.  The MNIST dataset contains only grayscale images.
# For more advanced image datasets, we'll have the three color channels (RGB).
fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
fig
