import random
import numpy as np
import pandas as pd

import tensorflow as tf
from pprint import pprint
from typing import List, Dict
from keras.layers import Layer
from tensorflow.python.keras import activations

class SimpleLayer(Layer):
    def __init__(self, units=32, number_params_train=20, activity=True, use_bias=False, activation=None):
        super(SimpleLayer, self).__init__()

        self.units = units
        self.activity = activity
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
        
        assert self.units>=self.parms
        column=[val for val in range(self.units)]
        columns:List=[]
        count:int=0
        while count!=self.parms:
            choice=random.choice(column)
            columns.append(choice)
            column.remove(choice)
            count+=1
        columns = tf.constant(columns, dtype=tf.int32)
        columns = tf.reshape(columns, shape=(self.parms,1))
        #-----------------------------------------------------

        if input_shape[-1]>self.parms:
            row=[val for val in range(input_shape[-1])]
            rows:List=[]
            count:int=0
            while count!=self.parms:
                choice=random.choice(row)
                rows.append(choice)
                row.remove(choice)
                count+=1
        else:
            row=[val for val in range(input_shape[-1])]
            rows:List=[]
            count:int=0
            while count!=self.parms:
                choice=random.choice(row)
                rows.append(choice)
                count+=1
        rows = tf.constant(rows, dtype=tf.int32)
        rows = tf.reshape(rows, shape=(self.parms,1))
        #------------------------------------------------


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
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True)

        self.w2 = self.add_weight(
            name='w2',
            shape=(input_shape[-1], self.units),
            dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotUniform(),
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
        if self.activity:
            kernel = tf.multiply(self.w1,self.mask)+tf.multiply(self.w2,self.one-self.mask)
        else:
            kernel = tf.multiply(self.w1,self.mask)
        
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
    linear_layer = SimpleLayer(5, number_params_train=5,activity=False, activation='softmax')
    Input = tf.ones((2, 3), dtype=tf.float32)
    output = linear_layer(Input)
    pprint(output)

