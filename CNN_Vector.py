import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from Data_Util import loadHistogram
import tensorflow as tf


def trainHistCNN(X_train, X_test, y_train, y_test, save=False, plot=False, epochs=15):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(256, 3)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)

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
    loadData = True

    n = 2500

    directories = ["Path",
                   "Path"]

    if loadData:
        X_train = np.load("Path")
        x_test = np.load("Path")
        y_train = np.load("Path")
        y_test = np.load("Path")
    else:
        X_train, x_test, y_train, y_test = loadHistogram(directories, n=n, plot=False)
        np.save('X_train_Hist', X_train)
        np.save('X_test_Hist', x_test)
        np.save('y_train_Hist', y_train)
        np.save('y_test_Hist', y_test)

    print(np.sum(y_train == 1) + np.sum(y_test == 1))
    print(np.sum(y_train == 0) + np.sum(y_test == 0))

    if load:
        model = tf.keras.models.load_model('Path')
    else:
        model = trainHistCNN(X_train, x_test, y_train, y_test, epochs=150, save=False, plot=False)

    y_pred = model.predict(x_test)
    print(f"The model is {accuracy_score(np.rint(np.array(y_pred).flatten()), y_test) * 100}% accurate")
    score = model.evaluate(x_test, y_test, verbose=0)
    cm = confusion_matrix(y_test, np.rint(y_pred.flatten()))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()
