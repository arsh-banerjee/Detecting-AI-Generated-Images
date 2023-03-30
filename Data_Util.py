import numpy as np
import os
from PIL import Image, ImageOps
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
import cv2
from sklearn.model_selection import train_test_split


def loadDataLinear(directories=None, n=1500):
    if directories is None:
        directories = ["", ""]
    category = []
    label = -1
    Hist = []

    for directory in directories:
        i = 0
        label += 1
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                image = Image.open(f)

                # Skip over corrupted images/incorrect shapes
                if len(np.array(image).shape) != 3:
                    continue
                if np.array(image).shape[2] == 4:
                    continue

                im = ImageOps.fit(image, (512, 512))  # Resize image to 512x512
                im_gs = rgb2gray(im)  # Convert image to grayscale
                # image_data.append(im.flatten())
                # histogram, bin_edges = np.histogram(im_gs, bins=256, range=(0, 1))  # Create histogram of grayscale

                imcv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)  # Convert Pillow Image to cv2 image
                blue_color = cv2.calcHist([imcv], [0], None, [256], [0, 256])  # Hist of blue channel
                red_color = cv2.calcHist([imcv], [1], None, [256], [0, 256])  # Hist of red channel
                green_color = cv2.calcHist([imcv], [2], None, [256], [0, 256])  # Hist of green channel
                rgb = np.concatenate((blue_color.flatten(), red_color.flatten(), green_color.flatten()))

                category.append(label)
                Hist.append(rgb)

                i += 1
                if i == n:
                    print("Directory Loaded")
                    break

    x_train, x_test, y_train, y_test = train_test_split(Hist, category, test_size=0.20, random_state=4)
    return x_train, x_test, y_train, y_test


def loadDataCNN(directories=None, n=1500, gs=False, Fourier = False, size = 64):
    # Load Files and Create Features
    image_data = []
    category = []
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
                    #  im = np.fft.fftshift(np.fft.fft2(im))  # Fourier Transform feature
                    im = (im * 255).astype(np.uint8)
                    im = im - cv2.fastNlMeansDenoising(im)

                image_data.append(im)
                category.append(label)
            i += 1
            if i == n:
                #  print("Directory Loaded")
                break

    image_data = np.array(image_data)
    category = np.array(category)
    X_train, X_test, y_train, y_test = train_test_split(image_data, category, test_size=0.2, random_state=4)
    return X_train, X_test, y_train, y_test
