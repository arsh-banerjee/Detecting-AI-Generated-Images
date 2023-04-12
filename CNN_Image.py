import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from Data_Util import loadRandCrop
import numpy as np


def trainCNN(X_train, X_test, y_train, y_test, size=200, n=500, epochs=5, verbose=True, name="Model", plot=False):
    print("Image Size: {size}x{size}, Samples: {n}, Epochs: {e}".format(size=size, n=n, e=epochs))

    X_train = X_train / 255.0
    X_test = X_test / 255.0
    batch_size = 16

    model = tf.keras.models.Sequential([
        # since Conv2D is the first layer of the neural network, we should also specify the size of the input
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3), kernel_initializer='he_normal'),
        # apply pooling
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(0.2),
        # apply pooling
        tf.keras.layers.MaxPooling2D(10, 10),
        tf.keras.layers.Conv2D(64, (6, 6), activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Conv2D(64, (6, 6), activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        # and define 512 neurons for processing the output coming by the previous layers
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)
    y_pred = model.predict(X_test, verbose=0)

    if plot:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(name)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    model.save('CNN_Dalle.h5')
    return model


if __name__ == '__main__':
    load = False
    macOS = False
    loadData = False

    if macOS:
        directories = ["/Users/arshbanerjee/Downloads/archive/StableDiff/StableDiff/StableDiff/",
                       "/Users/arshbanerjee/Downloads/archive/laion400m-laion4.75/laion400m-laion4.75/laion400m"
                       "-laion4.75+/laion400m-laion4.75+/"]
    else:
        directories = ["C:/Users/arsh0/Downloads/archive/dalle-samples (6)/",
                       "C:/Users/arsh0/Downloads/archive/laion400m-laion4.75+/"]

    n = 2500
    size = 200

    if loadData:
        X_train = np.load('Data/X_train_crop_200_Full.npy')
        x_test = np.load("Data/X_test_crop_200_Full.npy")
        y_train = np.load("Data/y_train_crop_200_Full.npy")
        y_test = np.load("Data/y_test_crop_200_Full.npy")
    else:
        X_train, x_test, y_train, y_test = loadRandCrop(directories, n=n, size=size)
        np.save('X_train_crop_200_Dalle', X_train)
        np.save('X_test_crop_200_Dalle', x_test)
        np.save('y_train_crop_200_Dalle', y_train)
        np.save('y_test_crop_200_Dalle', y_test)

    print(np.sum(y_train==1)+np.sum(y_test==1))
    print(np.sum(y_train == 0) + np.sum(y_test == 0))
    if load:
        CNN = tf.keras.models.load_model('CNN_Full.h5')
    else:
        CNN = trainCNN(X_train, x_test, y_train, y_test, size=size, n=n, epochs=12,
                       name="CNN (200x200)", plot=False)

    y_pred = CNN.predict(x_test)
    print(classification_report(y_test, np.rint(y_pred.flatten())))
    cm = confusion_matrix(y_test, np.rint(y_pred.flatten()))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()
