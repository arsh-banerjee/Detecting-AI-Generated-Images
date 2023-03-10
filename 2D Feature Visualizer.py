import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
import cv2
from PIL import Image, ImageOps
import os

dirc = "/Users/arshbanerjee/Downloads/archive/StableDiff/StableDiff/StableDiff/"
dirc2 = "/Users/arshbanerjee/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m-laion4.75+/laion400m-laion4.75+/"

dirc = "C:/Users/arsh0/Downloads/archive/StableDiff/StableDiff/StableDiff/"
dirc2 = "C:/Users/arsh0/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m-laion4.75+/laion400m-laion4.75+"

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

            # Skip over corrupted images/incorrect shapes
            if len(np.array(image).shape) != 3:
                continue
            if np.array(image).shape[2] == 4:
                continue

            im = rgb2gray(ImageOps.fit(image, (512, 512)))
            #im_ftt = np.fft.fftshift(np.fft.fft2(im))
            im = (im*255).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoising(im)
            image_data.append(im-denoised)


        i+=1
        if i == 100:
            break

    i = 0

    for filename in os.listdir(dirc2):
        f = os.path.join(dirc2, filename)
        # checking if it is a file
        if os.path.isfile(f):
            image = Image.open(f)

            # Skip over corrupted images/incorrect shapes
            if len(np.array(image).shape) != 3:
                continue
            if np.array(image).shape[2] == 4:
                continue

            im = rgb2gray(ImageOps.fit(image, (512, 512)))
            #im_ftt = np.fft.fftshift(np.fft.fft2(im))
            im = (im * 255).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoising(im)
            image_data2.append(im-denoised)

        i+=1
        if i == 100:
            break

    average = np.mean(image_data, axis=0)
    average2 = np.mean(image_data2, axis=0)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    #plt.imshow(np.log(abs(average)), cmap='gray')
    plt.imshow(average, cmap='gray')


    plt.figure(num=None, figsize=(8, 6), dpi=80)
    #plt.imshow(np.log(abs(average2)), cmap='gray')
    plt.imshow(average2, cmap='gray')

    #plt.figure(num=None, figsize=(8, 6), dpi=80)
    #plt.imshow(np.log(abs(average))-np.log(abs(average2)), cmap='inferno')

    plt.show()
