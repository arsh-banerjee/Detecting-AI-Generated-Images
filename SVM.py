import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn import svm
import pickle
from Data_Util import loadDataLinear

def trainSVM(X_train, y_train, save=False, plot=False):
    svc = svm.SVC(probability=True)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_train)
    print(f"The model is {accuracy_score(y_pred, y_train) * 100}% accurate")

    if plot:
        cm = confusion_matrix(y_train, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    if save:
        pickle.dump(svc, open('SVM.sav', 'wb'))

    return svm


# Press the green button in the gutter to run the script.
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
                       "C:/Users/arsh0/Downloads/archive/AI_Combined"]

    X_train, x_test, y_train, y_test = loadDataLinear(directories, n=7000)

    if load:
        model = pickle.load(open(filename, 'rb'))
    else:
        model = trainSVM(X_train, y_train)
