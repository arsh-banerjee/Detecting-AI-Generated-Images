import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from PIL import Image, ImageOps
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


dirc = "/Users/arshbanerjee/Downloads/archive/StableDiff/StableDiff/StableDiff/"
dirc2 = "/Users/arshbanerjee/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m-laion4.75+/laion400m-laion4.75+/"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    i = 0
    image_data = []
    category = []

    for filename in os.listdir(dirc):
        f = os.path.join(dirc, filename)
        # checking if it is a file
        if os.path.isfile(f):
            image = Image.open(f)
            im = ImageOps.fit(image, (512, 512))
            im = np.array(im)
            #im = rgb2gray(im)
            #im = np.fft.fftshift(np.fft.fft2(im))
            image_data.append(im)
            category.append(0)
        i+=1
        if i == 500:
            break

    i = 0

    for filename in os.listdir(dirc2):
        f = os.path.join(dirc2, filename)
        # checking if it is a file
        if os.path.isfile(f):
            image = Image.open(f)
            im = ImageOps.fit(image, (512, 512))
            im = np.array(im)
            #im = rgb2gray(im)
            #im = np.fft.fftshift(np.fft.fft2(im))
            image_data.append(im)
            category.append(1)

        i+=1
        if i == 500:
            break

    image_data = np.array(image_data)
    category = np.array(category)
    X_train, X_test, y_train, y_test = train_test_split(image_data, category, test_size=0.2, random_state=4)

    batch_size = 16
    nb_classes = 4
    nb_epochs = 5
    img_rows, img_columns = 200, 200
    img_channel = 3
    nb_filters = 32
    nb_pool = 2
    nb_conv = 3
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                               input_shape=(512, 512, 3)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test Score: ", score[0])
    print("Test accuracy: ", score[1])
