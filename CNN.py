import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from PIL import Image, ImageOps
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import regularizers
import cv2
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from Data_Util import loadDataCNN


def trainCNN(size=200, n=500, epochs=5, verbose=True, name="Model", plot=False):
    print("Image Size: {size}x{size}, N: {n}, Epochs: {e}, GS: {bool}".format(size=size, n=n, e=epochs, bool=gs))

    X_train = X_train / 255.0
    X_test = X_test / 255.0
    batch_size = 16

    model = tf.keras.models.Sequential([
        # since Conv2D is the first layer of the neural network, we should also specify the size of the input
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(size, size, 3)),
        # apply pooling
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        # and repeat the process
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                               kernel_regularizer=regularizers.l2(l=0.1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # flatten the result to feed it to the dense layer
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        # and define 512 neurons for processing the output coming by the previous layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test Score: ", score[0])
    print("Test accuracy: ", score[1])
    y_pred = model.predict(X_test)

    if plot:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(name)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    model.save('CNN.h5')
    print('Model Saved!')


if __name__ == '__main__':
    load = False
    filename = 'SVM.sav'
    macOS = False

    if macOS:
        directories = ["/Users/arshbanerjee/Downloads/archive/StableDiff/StableDiff/StableDiff/",
                       "/Users/arshbanerjee/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m"
                       "-laion4.75+/laion400m-laion4.75+/"]
    else:
        directories = ["C:/Users/arsh0/Downloads/archive/Real_combined/",
                       "C:/Users/arsh0/Downloads/archive/AI_Combined/"]

    X_train, x_test, y_train, y_test = loadDataCNN(directories, n=7000)
    #  trainCNN(size=200, n=1500, epochs=15, macOs=False, name="CNN with image size 200", plot=True)
    if load:
        pass
    else:
        trainCNN(size=64, n=7500, epochs=10, name="CNN (200x200)", plot=False)
