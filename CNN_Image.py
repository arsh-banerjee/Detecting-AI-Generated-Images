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
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3),
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
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
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()

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

    model.save('CNN.h5')
    return model


if __name__ == '__main__':
    load = True
    loadData = False

    directories = ["/Path",
                   "/Path"]

    n = 5000
    size = 200

    if loadData:
        X_train = np.load('/Path')
        x_test = np.load("/Path")
        y_train = np.load("/Path")
        y_test = np.load("/Path")
    else:
        X_train, x_test, y_train, y_test = loadRandCrop(directories, n=n, size=size)
        np.save('File Name', X_train)
        np.save('File Name', x_test)
        np.save('File Name', y_train)
        np.save('File Name', y_test)

    print(np.sum(y_train == 1) + np.sum(y_test == 1))
    print(np.sum(y_train == 0) + np.sum(y_test == 0))

    if load:
        CNN = tf.keras.models.load_model('CNN.h5')
    else:
        CNN = trainCNN(X_train, x_test, y_train, y_test, size=size, n=n, epochs=25,
                       name="CNN (200x200)", plot=False)

    y_pred = CNN.predict(x_test)
    print(classification_report(y_test, np.rint(y_pred.flatten())))
    cm = confusion_matrix(y_test, np.rint(y_pred.flatten()))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()
