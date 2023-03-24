import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from PIL import Image, ImageOps
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import cv2
from sklearn.model_selection import GridSearchCV


def trainSVM(n=1500, macOs=False):
    if macOs:
        directories = ["/Users/arshbanerjee/Downloads/archive/StableDiff/StableDiff/StableDiff/",
                       "/Users/arshbanerjee/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m"
                       "-laion4.75+/laion400m-laion4.75+/"]
    else:
        directories = ["C:/Users/arsh0/Downloads/archive/StableDiff/StableDiff/StableDiff/",
                       #  "C:/Users/arsh0/Downloads/archive/dalle-samples (6)/"]
                       "C:/Users/arsh0/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m-laion4.75"
                       "+/laion400m"
                       "-laion4.75+"]

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
                histogram, bin_edges = np.histogram(im_gs, bins=256, range=(0, 1))  # Create histogram of grayscale

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

    svc = svm.SVC(probability=True)
    x_train, x_test, y_train, y_test = train_test_split(Hist, category, test_size=0.20, random_state=77)

    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trainSVM()
