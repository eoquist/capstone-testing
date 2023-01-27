# using this site as a reference/guide
# https://nextjournal.com/gkoehler/digit-recognition-with-keras

###############################
#   PREPARING THE DATASET    #
##############################

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

# In order to train our neural network to classify images we first have to unroll the height \timeswidth
# pixel format into one big vector - the input vector. So its length must be 28 \cdot 28 = 784.
# But let's graph the distribution of our pixel values.
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(X_train[0], cmap='gray', interpolation='none')
plt.title("Digit: {}".format(y_train[0]))
plt.xticks([])
plt.yticks([])
plt.subplot(2, 1, 2)
plt.hist(X_train[0].reshape(784))
plt.title("Pixel Value Distribution")
fig

# As expected, the pixel values range from 0 to 255:
# the background majority close to 0, and those close to 255 representing the digit.

# Normalizing the input data helps to speed up the training. Also, it reduces the chance of getting stuck
# in local optima, since we're using stochastic gradient descent to find the optimal weights for the network.

# Let's reshape our inputs to a single vector vector and normalize the pixel values to lie between 0 and 1.
# let's print the shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)
# X_train shape (60000, 28, 28)
# y_train shape (60000,)
# X_test shape (10000, 28, 28)
# y_test shape (10000,)

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)
# Train matrix shape (60000, 784)
# Test matrix shape (10000, 784)

# So far the truth (Y in machine learning lingo) we'll use for training still holds integer values from 0 to 9.
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]))

# Let's encode our categories - digits from 0 to 9 - using one-hot encoding. The result is a vector with a length equal
# to the number of categories. The vector is all zeroes except in the position for the respective category. Thus a '5' will
# be represented by [0,0,0,0,1,0,0,0,0].

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
# Shape before one-hot encoding:  (60000,)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
# Shape after one-hot encoding:  (60000, 10)


###########################
#   BUILDING THE NETWORK  #
###########################
# see site for better understanding
# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

#########################################
#   COMPILING AND TRAINING THE MODEL    #
########################################
# compiling the sequential model
# !!! look into Keras metrics
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')

# training the model and saving metrics in history
history = model.fit(X_train, Y_train,
                    batch_size=128, epochs=20,
                    verbose=2,
                    validation_data=(X_test, Y_test))

# saving the model
save_dir = "/results/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# plotting the metrics
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

fig

#########################################
#   EVALUATE THE MODEL'S PERFORMANCE    #
########################################
mnist_model = load_model()
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])
# Test Loss 0.07407343140110847
# Test Accuracy 0.9822

# load the model and create predictions on the test set
mnist_model = load_model()
predicted_classes = mnist_model.predict_classes(X_test)

# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices), " classified correctly")
print(len(incorrect_indices), " classified incorrectly")

# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7, 14)

figure_evaluation = plt.figure()

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6, 3, i+1)
    plt.imshow(X_test[correct].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title(
        "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                          y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6, 3, i+10)
    plt.imshow(X_test[incorrect].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title(
        "Predicted {}, Truth: {}".format(predicted_classes[incorrect],
                                         y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

figure_evaluation
