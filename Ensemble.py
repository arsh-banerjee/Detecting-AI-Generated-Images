import numpy as np
import tensorflow as tf
import pickle

if __name__ == '__main__':
    CNN = tf.keras.models.load_model('CNN.h5')
    SVM = pickle.load(open('SVM.sav', 'rb'))