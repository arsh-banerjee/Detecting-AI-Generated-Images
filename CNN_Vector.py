import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Data_Util import loadHistogram
import tensorflow as tf


def trainSVM(X_train, X_test, y_train, y_test, save=False, plot=False, epochs=15):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer='he_normal', input_shape=(3, 256)),
        tf.keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)

    if plot:
        cm = confusion_matrix(y_train, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    if save:
        model.save('CNN2_Full.h5')

    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load = False
    macOS = False
    loadData = True

    n = 2500

    if macOS:
        directories = ["/Users/arshbanerjee/Downloads/archive/StableDiff/StableDiff/StableDiff/",
                       "/Users/arshbanerjee/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m"
                       "-laion4.75+/laion400m-laion4.75+/"]
    else:
        directories = ["C:/Users/arsh0/Downloads/archive/dalle-samples (6)/",
                       "C:/Users/arsh0/Downloads/archive/laion400m-laion4.75+/"]

    if loadData:
        X_train = np.load("Data/X_train_Hist_Full.npy")
        x_test = np.load("Data/X_test_Hist_Full.npy")
        y_train = np.load("Data/y_train_Hist_Full.npy")
        y_test = np.load("Data/y_test_Hist_Full.npy")
    else:
        X_train, x_test, y_train, y_test = loadHistogram(directories, n=n, plot=True)
        np.save('X_train_Hist_Dalle', X_train)
        np.save('X_test_Hist_Dalle', x_test)
        np.save('y_train_Hist_Dalle', y_train)
        np.save('y_test_Hist_Dalle', y_test)

    print(np.sum(y_train == 1) + np.sum(y_test == 1))
    print(np.sum(y_train == 0) + np.sum(y_test == 0))

    if load:
        model = tf.keras.models.load_model('Models/CNN2_Full.h5')
    else:
        model = trainSVM(X_train, x_test, y_train, y_test, epochs=50, save=True)

    y_pred = model.predict(x_test)
    score = model.evaluate(x_test, y_test, verbose=0)
    cm = confusion_matrix(y_test, np.rint(y_pred.flatten()))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()
    print(score)

