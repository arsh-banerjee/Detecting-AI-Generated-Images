import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
import PIL
from PIL import Image, ImageOps
import os


dirc = "/Users/arshbanerjee/Downloads/archive/StableDiff/StableDiff/StableDiff/"
dirc2 = "/Users/arshbanerjee/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m-laion4.75+/laion400m-laion4.75+/"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    i = 0
    image_data = []
    image_data2 = []

    for filename in os.listdir(dirc):
        f = os.path.join(dirc, filename)
        # checking if it is a file
        if os.path.isfile(f):
            image = Image.open(f)
            im = rgb2gray(ImageOps.fit(image, (512, 512)))
            image_data.append(im)
        i+=1
        if i == 1:
            break

    i = 0

    for filename in os.listdir(dirc2):
        f = os.path.join(dirc2, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            image = Image.open(f)
            im = rgb2gray(ImageOps.fit(image, (512, 512)))
            image_data2.append(im)
        i+=1
        if i == 1:
            break

    average = np.mean(image_data, axis=0)
    average2 = np.mean(image_data2, axis=0)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(average, cmap='gray')

    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(average2, cmap='gray')

    histogram, bin_edges = np.histogram(average, bins=256, range=(0, 1))
    histogram2, bin_edges2 = np.histogram(average2, bins=256, range=(0, 1))

    plt.figure()
    plt.title("Grayscale Histogram 1")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram)  # <- or here

    plt.figure()
    plt.title("Grayscale Histogram 2")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram2)  # <- or here

    plt.show()

