import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from typing import List, Dict
from keras.layers import Layer
from tensorflow.python.keras import activations
# Initializer
# Layer
class SimpleLayer(Layer):
    def __init__(self, units=32, number_params_train=20, seed=(1, 2), use_bias=False, activation=None):
        super(SimpleLayer, self).__init__()
        self.seed = seed
        self.units = units
        self.use_bias = use_bias
        self.parms = number_params_train
        self.activation = activations.get(activation)
        
      
    @property
    def indice(self):
        return self.__indice
    
    @property
    def get_mask(self):
        return self.mask
    
    def build(self, input_shape):
        row = input_shape[-1]
        column = self.units
        rows = tf.random.uniform(shape=(self.parms, 1), minval=0, maxval=row, dtype=tf.int32, seed=self.seed[0],
                                 name=None)
        columns = tf.random.uniform(shape=(self.parms, 1), minval=0, maxval=column, dtype=tf.int32,
                                    seed=self.seed[1], name=None)
        indices = tf.concat([rows, columns], 1)
        indices = indices.numpy()
        self.__indice: List[List] = list()
        for index in indices:
            self.__indice.append(list(index))
           
        values = [1 for i in range(len(self.__indice))]
        booleen = tf.scatter_nd(self.indice, values, (input_shape[-1], self.units)) > 0
        self.mask = tf.cast(booleen, dtype=tf.float32)
        self.one = tf.ones((input_shape[-1], self.units), dtype=tf.float32)


        self.w1 = self.add_weight(
            name='w1',
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.GlorotUniform(seed=1),
            dtype=tf.float32,
            trainable=True)

        self.w2 = self.add_weight(
            name='w2',
            shape=(input_shape[-1], self.units),
            dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotUniform(seed=1),
            trainable=False)

            

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                dtype=tf.float32,
                trainable=self.use_bias)
        else:
            self.bias = None
        
        self.build=True

    def call(self, inputs):
        kernel = tf.multiply(self.w1,self.mask) + tf.multiply(self.w2,self.one-self.mask)
        ouputs = tf.matmul(a=inputs, b=kernel)

        if self.use_bias:
            ouputs = tf.nn.bias_add(ouputs, self.bias)
            ouputs = tf.cast(ouputs, dtype=tf.float32)

        if self.activation is not None:
            ouputs = self.activation(ouputs)
        else:
            pass

        return ouputs

if __name__=='__main__':
    linear_layer = SimpleLayer(10, number_params_train=40, activation='softmax')
    Input = tf.ones((2, 10), dtype=tf.float32)
    output = linear_layer(Input)
    pprint(output)
