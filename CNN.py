import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from PIL import Image, ImageOps
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import regularizers

def trainCNN(size=200, n=500, epochs=5, gs=False, Fourier=False, macOs=False, verbose=True, name="Model", plot=False):
    print("Image Size: {size}x{size}, N: {n}, Epochs: {e}, GS: {bool}".format(size=size, n=n, e=epochs, bool=gs))
    if macOs:
        directories = ["/Users/arshbanerjee/Downloads/archive/StableDiff/StableDiff/StableDiff/",
                       "/Users/arshbanerjee/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m-laion4.75+/laion400m-laion4.75+/"]
    else:
        directories = ["C:/Users/arsh0/Downloads/archive/StableDiff/StableDiff/StableDiff/",
                       "C:/Users/arsh0/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m-laion4.75+/laion400m-laion4.75+"]

    # Load Files and Create Features
    image_data = []
    category = []
    size = 200  # Image Size
    n = 500  # Number of images to pull from each class
    label = -1
    channels = 3
    if gs | Fourier:
        channels = 1

    for dirc in directories:
        i = 0
        label += 1
        for filename in os.listdir(dirc):
            f = os.path.join(dirc, filename)
            # checking if it is a file
            if os.path.isfile(f):
                image = Image.open(f)

                # Skip over corrupted images/incorrect shapes
                if len(np.array(image).shape) != 3:
                    continue
                if np.array(image).shape[2] == 4:
                    continue

                im = ImageOps.fit(image, (size, size))
                im = np.array(im)

                if gs:
                    im = rgb2gray(im)  # Grayscale
                if Fourier:
                    im = rgb2gray(im)
                    im = np.fft.fftshift(np.fft.fft2(im))  # Fourier Transform feature
                image_data.append(im)
                category.append(label)
            i += 1
            if i == n:
                #  print("Directory Loaded")
                break

    image_data = np.array(image_data)
    category = np.array(category)
    X_train, X_test, y_train, y_test = train_test_split(image_data, category, test_size=0.2, random_state=4)

    batch_size = 16

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                               input_shape=(size, size, channels), kernel_regularizer=regularizers.L1L2(l1=1, l2=1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=regularizers.L1L2(l1=1, l2=1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test Score: ", score[0])
    print("Test accuracy: ", score[1])

    if plot:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(name)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    trainCNN(size=200, n=1500, epochs=15, macOs=False, name="CNN with image size 200", plot=True)
    trainCNN(size=200, n=1500, epochs=15, Fourier=True, macOs=False, name="CNN with Fourier", plot=True)
