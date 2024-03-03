import cProfile
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from SpectralLayer import*
from utilsSimpleConv2D import*
from spectralconvolutions import *

from tensorflow.keras.layers import Layer, Dense
from typing import Tuple,List,Any,Dict


class ModelSpectral(object):
    def __init__(self,
                 batch_size=200,
                 epochs=20,
                 verbose=1,
                 learning_rate=0.03,
                 name_data="mnist",
                 strides=1,
                 padding="VALID",
                 kernel_size=3,
                 filters=1,
                 use_bias=False,
                 spectral_config={'is_base_trainable': False,
                                  'is_diag_start_trainable': False,
                                  'is_diag_end_trainable': True,
                                  },
                 ):

        self.verbose = verbose
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.name_data = name_data
        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size
        self.filters = filters
        self.use_bias = use_bias
        self.spectral_config = spectral_config

    def build_model(self, units=1000, Pad_Jacob=True,*args):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(28, 28, 1), dtype=tf.float32))
        
        if Pad_Jacob:
            self.model.add(PaddingJacobiens(kernel_size=self.kernel_size, strides=self.strides, padding=self.padding))
        
        self.model.add(SpecConv2D(filters=self.filters, kernel_size=self.kernel_size, use_lambda_in=True, use_bias=self.use_bias,
                       activation="relu"))

        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(Spectral(units, **self.spectral_config, use_bias=self.use_bias, activation=None))
        self.model.add(Spectral(10, **self.spectral_config, use_bias=self.use_bias, activation='softmax'))

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit_model(self, *args):

        self.accuracy: List[Any] = list()
        if self.name_data == 'mnist':
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)
            x_train, x_test = x_train / 255.0, x_test / 255.0
        elif self.name_data == 'fashion_mnist':
            fashion_mnist = tf.keras.datasets.fashion_mnist
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)
            x_train, x_test = x_train / 255.0, x_test / 255.0
        else:
            raise NotImplemented

        flat_train = np.reshape(x_train, [x_train.shape[0], 28 * 28])
        flat_test = np.reshape(x_test, [x_test.shape[0], 28 * 28])
        print("Beging fitting...\n")
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                       validation_data=(x_test, y_test))
        print("Done.\n")
        self.accuracy.append(self.model.evaluate(x_test, y_test, batch_size=32, verbose="auto"))



