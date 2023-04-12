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

    CNN = tf.keras.models.load_model('CNN.h5')
    CNN2 = tf.keras.models.load_model('CNN2.h5')
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

    # predict("C:/Users/arsh0/Downloads/archive/Predict/")

    macOS = False
    loadData = True

    if macOS:
        directories = ["/Users/arshbanerjee/Downloads/archive/StableDiff/StableDiff/StableDiff/",
                       "/Users/arshbanerjee/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m"
                       "-laion4.75+/laion400m-laion4.75+/"]
    else:
        directories = ["C:/Users/arsh0/Downloads/archive/Real_combined/",
                       "C:/Users/arsh0/Downloads/archive/AI_Combined"]

    CNN = tf.keras.models.load_model('CNN.h5')
    CNN2 = tf.keras.models.load_model('CNN2.h5')

    n = 1000

    if loadData:
        X_train_CNN = np.load('Data/X_train_crop_200.npy')
        x_test_CNN = np.load("Data/X_test_crop_200.npy")
        y_train_CNN = np.load("Data/y_train_crop_200.npy")
        y_test_CNN = np.load("Data/y_test_crop_200.npy")
        X_train_SVM = np.load('Data/X_train_Hist.npy')
        x_test_SVM = np.load("Data/X_test_Hist.npy")
        y_train_SVM = np.load("Data/y_train_Hist.npy")
        y_test_SVM = np.load("Data/y_test_Hist.npy")
    else:
        X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM = loadHistogram(directories, n=n)
        X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN = loadRandCrop(directories, n=n, gs=False, Fourier=False,
                                                                        size=64)

    print("X_train: {s1}, y_train: {s2}, X_test: {s3}, y_test: {s4}".format(s1=X_train_CNN.shape, s2=y_train_CNN.shape,
                                                                            s3=x_test_CNN.shape, s4=y_test_CNN.shape))

    print("X_train: {s1}, y_train: {s2}, X_test: {s3}, y_test: {s4}".format(s1=X_train_SVM.shape, s2=y_train_SVM.shape,
                                                                            s3=x_test_SVM.shape, s4=y_test_SVM.shape))
    x_test_CNN = x_test_CNN / 255.0


    y_pred_SVM = CNN2.predict(x_test_SVM)
    y_pred_CNN = CNN.predict(x_test_CNN)
    y_pred = []

    for i in range(len(y_pred_SVM)):
        if y_pred_SVM[i] == y_pred_CNN[i]:
            y_pred.append(y_pred_SVM[i])
        else:
            if y_pred_CNN[i] == 1:
                y_pred.append(y_pred_CNN[i])
            else:
                y_pred.append(y_pred_SVM[i])

    y_pred = np.rint(np.array(y_pred).flatten())

    print(f"The model is {accuracy_score(np.rint(np.array(y_pred_CNN)), y_test_CNN) * 100}% accurate")
    print(f"The model is {accuracy_score(np.rint(np.array(y_pred_SVM).flatten()), y_test_SVM) * 100}% accurate")
    print(f"The model is {accuracy_score(y_pred, y_test_SVM) * 100}% accurate")
    cm = confusion_matrix(y_pred, y_test_SVM)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
