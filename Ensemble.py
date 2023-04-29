import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from Data_Util import loadHistogram, loadRandCrop, get_random_crop
import imageio.v3 as iio
from PIL import Image


def predict(directory):
    Hist = []
    image_data = []
    size = 200
    colors = ("red", "green", "blue")
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        print(f)
        # checking if it is a file
        if os.path.isfile(f):
            image = iio.imread(uri=f)

            # Skip over corrupted images/incorrect shapes
            if len(np.array(image).shape) != 3:
                continue
            if np.array(image).shape[2] == 4:
                continue

            singleImage = []
            for channel_id, color in enumerate(colors):
                histogram, bin_edges = np.histogram(
                    image[:, :, channel_id], bins=256, range=(0, 256)
                )
                singleImage.append(histogram)

            Hist.append(singleImage)

            im = Image.open(f)
            im = np.array(im)
            im = get_random_crop(im, size, size)
            image_data.append(im)

    Hist = np.array(Hist)
    image_data = np.array(image_data)

    CNN = tf.keras.models.load_model('Models/CNN.h5')
    CNN2 = tf.keras.models.load_model('Models/CNN2.h5')
    y_pred_CNN2 = CNN2.predict(Hist)
    y_pred_CNN = CNN.predict(image_data)

    print(y_pred_CNN2)
    print(y_pred_CNN)

    y_pred = []

    for i in range(len(y_pred_CNN2)):
        if y_pred_CNN2[i] == y_pred_CNN[i]:
            y_pred.append(y_pred_CNN2[i])
        else:
            if y_pred_CNN[i] == 0:
                y_pred.append(y_pred_CNN[i])
            elif y_pred_CNN[i] == 1 and y_pred_CNN2[i] > 0:
                y_pred.append(y_pred_CNN[i])
            else:
                y_pred.append(y_pred_CNN2[i])

    y_pred = np.rint(np.array(y_pred).flatten())
    print(y_pred)


if __name__ == '__main__':
    # To Predict a single image place in a new folder and put the path below
    # predict("Path/To/Predict/Directory")

    loadData = True

    directories = ["Path/To/Images", "Path/To/Images"]

    CNN = tf.keras.models.load_model('Models/CNN.h5')
    CNN2 = tf.keras.models.load_model('Models/CNN2.h5')

    n = 5000  # Number of images per class

    if loadData:
        x_test_CNN = np.load("Data/X_test_crop_200.npy")
        y_test_CNN = np.load("Data/y_test_crop_200.npy")
        x_test_CNN2 = np.load("Data/X_test_Hist.npy")
        y_test_CNN2 = np.load("Data/y_test_Hist.npy")
        X_train_CNN = []
        X_train_CNN2 = []
        y_train_CNN = []
        y_train_CNN2 = []
    else:
        X_train_CNN2, X_test_CNN2, y_train_CNN2, y_test_CNN2 = loadHistogram(directories, n=n)
        X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN = loadRandCrop(directories, n=n, gs=False, Fourier=False,
                                                                        size=64)

    print("X_train: {s1}, y_train: {s2}, X_test: {s3}, y_test: {s4}".format(s1=X_train_CNN.shape, s2=y_train_CNN.shape,
                                                                            s3=x_test_CNN.shape, s4=y_test_CNN.shape))

    print(
        "X_train: {s1}, y_train: {s2}, X_test: {s3}, y_test: {s4}".format(s1=X_train_CNN2.shape, s2=y_train_CNN2.shape,
                                                                          s3=x_test_CNN2.shape, s4=y_test_CNN2.shape))
    x_test_CNN = x_test_CNN / 255.0

    y_pred_CNN2 = CNN2.predict(x_test_CNN2)
    y_pred_CNN = CNN.predict(x_test_CNN)
    y_pred = []

    for i in range(len(y_pred_CNN2)):
        if y_pred_CNN2[i] == y_pred_CNN[i]:
            y_pred.append(y_pred_CNN2[i])
        else:
            if y_pred_CNN[i] == 1:
                y_pred.append(y_pred_CNN[i])
            else:
                y_pred.append(y_pred_CNN2[i])

    y_pred = np.rint(np.array(y_pred).flatten())

    print(f"The model is {accuracy_score(np.rint(np.array(y_pred_CNN)), y_test_CNN) * 100}% accurate")
    print(f"The model is {accuracy_score(np.rint(np.array(y_pred_CNN2).flatten()), y_test_CNN2) * 100}% accurate")
    print(f"The model is {accuracy_score(y_pred, y_test_CNN2) * 100}% accurate")
    cm = confusion_matrix(y_pred, y_test_CNN2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
