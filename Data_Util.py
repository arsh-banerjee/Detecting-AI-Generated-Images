import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import imageio.v3 as iio


def loadHistogram(directories=None, n=1500, plot=True, size=200):
    if directories is None:
        directories = ["", ""]
    category = []
    label = -1
    Hist = []

    colors = ("red", "green", "blue")

    for directory in directories:
        i = 0
        label += 1
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                image = iio.imread(uri=f)

                # Skip over corrupted images/incorrect shapes
                if len(np.array(image).shape) != 3 or np.array(image).shape[2] == 4:
                    continue
                if np.array(image).shape[0] - 1 < size or np.array(image).shape[1] - 1 < size:
                    continue

                singleImage = []
                for channel_id, color in enumerate(colors):
                    histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=256, range=(0, 256))
                    histogram = histogram / np.linalg.norm(histogram)
                    singleImage.append(histogram)

                Hist.append(singleImage)
                category.append(label)
                i += 1
                if i == n:
                    print("Directory Loaded")
                    break

    Hist = np.array(Hist)
    category = np.array(category)

    x_train, x_test, y_train, y_test = train_test_split(Hist, category, test_size=0.20,
                                                        random_state=4)
    print("X_train: {s1}, y_train: {s2}, X_test: {s3}, y_test: {s4}".format(s1=x_train.shape, s2=y_train.shape,
                                                                            s3=x_test.shape, s4=y_test.shape))

    if plot:
        # create the histogram plot, with three lines, one for
        # each color
        for i in [0, 1]:
            plt.figure(i + 1)
            plt.xlim([0, 256])
            for channel_id, color in enumerate(colors):
                histogram = np.average(Hist[category == i], axis=0)
                plt.plot(bin_edges[0:-1], histogram[channel_id], color=color)

            if i == 0: plt.title("RGB Histogram of Real Images")
            else: plt.title("RGB Histogram of Generated Images")
            plt.xlabel("Color value")
            plt.ylabel("Pixel count")
        plt.show()

    return x_train, x_test, y_train, y_test


def loadRandCrop(directories=None, n=1500, size=64):
    # Load Files and Create Features
    image_data = []
    category = []
    label = -1

    for dirc in directories:
        i = 0
        label += 1
        for filename in os.listdir(dirc):
            f = os.path.join(dirc, filename)
            # checking if it is a file
            if os.path.isfile(f):
                im = Image.open(f)

                # Skip over corrupted images/incorrect shapes
                if len(np.array(im).shape) != 3 or np.array(im).shape[2] == 4:
                    continue
                if np.array(im).shape[0] - 1 < size or np.array(im).shape[1] - 1 < size:
                    continue

                im = np.array(im)
                im = get_random_crop(im, size, size)

                image_data.append(im)
                category.append(label)
            i += 1
            if i == n:
                print("Directory Loaded")
                break

    image_data = np.array(image_data)
    category = np.array(category)
    X_train, X_test, y_train, y_test = train_test_split(image_data, category, test_size=0.2, random_state=4)
    print("X_train: {s1}, y_train: {s2}, X_test: {s3}, y_test: {s4}".format(s1=X_train.shape, s2=y_train.shape,
                                                                            s3=X_test.shape, s4=y_test.shape))
    return X_train, X_test, y_train, y_test


def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop
