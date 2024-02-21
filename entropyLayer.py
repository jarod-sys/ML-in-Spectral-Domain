
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from pprint import pprint
from typing import List, Dict, Tuple,Any
from keras.layers import Layer
from tensorflow.python.keras import activations

class EntropyLayer(Layer):
    def __init__(self, units=32, use_bias=False,flux_in=True,flux_out=False, activity=False,activation=None):
        super(EntropyLayer, self).__init__()
        
        self.flux_in=flux_in
        self.flux_out=flux_out
        self.units = units
        self.activity = activity
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        
      
    @property
    def indice(self):
        return self.indices
    
    @property
    def get_mask(self):
        return self.mask
    def build_repartition(self,*args):
        if self.flux_in and not self.flux_out:
            self.residu=self.inputshape[-1]%self.units
            self.parms=math.floor(self.inputshape[-1]/self.units)
        elif not self.flux_in and  self.flux_out:
            self.residu=self.units%self.inputshape[-1]
            self.parms=math.floor(self.units/self.inputshape[-1])
            
        
    def set_indices(self,*args):
        self.build_repartition()
        if self.flux_in and not self.flux_out:
            tempon_value=[val for val in range(self.inputshape[-1])]
            self.indices:List[Tuple]=list()
            for j in range(self.units):
                count:int=0
                while count!=self.parms:
                    choice=random.choice(tempon_value)
                    self.indices.append((choice,j))
                    tempon_value.remove(choice)
                    count+=1
                    
            #residu
            assert len(tempon_value)==self.residu
            new_tempon_value=[val for val in range(self.units)]
            for i in tempon_value:
                choice=random.choice(new_tempon_value)
                self.indices.append((i,choice))
                new_tempon_value.remove(choice)
                count+=1
             
        elif not self.flux_in and  self.flux_out:
            tempon_value=[val for val in range(self.units)]
            self.indices:List[Tuple]=list()
            for i in range(self.inputshape[-1]):
                count:int=0
                while count!=self.parms:
                    choice=random.choice(tempon_value)
                    self.indices.append((i,choice))
                    tempon_value.remove(choice)
                    count+=1

            #residu
            assert len(tempon_value)==self.residu
            new_tempon_value=[val for val in range(self.inputshape[-1])]
            for j in tempon_value:
                choice=random.choice(new_tempon_value)
                self.indices.append((choice,j))
                new_tempon_value.remove(choice)
                count+=1
        else:
            raise NotImplemented
    
                        
    def build(self, input_shape):
        self.inputshape = tf.TensorShape(input_shape)
        self.set_indices()

        if self.flux_in and not self.flux_out:
            self.w = self.add_weight(
                name='w_in',
                shape=(input_shape[-1],),
                initializer=tf.keras.initializers.GlorotUniform(),
                dtype=tf.float32,
                trainable=self.flux_in)
        elif not self.flux_in and  self.flux_out:
            self.w = self.add_weight(
                name='w_out',
                shape=(self.units,),
                initializer=tf.keras.initializers.GlorotUniform(),
                dtype=tf.float32,
                trainable=self.flux_out)
        else:
            raise NotImplemented
            
            
        if self.activity:
            self.A =tf.constant(np.random.uniform(low=-0.1,high=0.1,size=(input_shape[-1],self.units)),dtype=tf.float32)
            values = [1 for i in range(len(self.indices))]
            booleen = tf.scatter_nd(self.indices, values, (self.inputshape[-1], self.units)) > 0
            self.mask = tf.cast(booleen, dtype=tf.float32)
            self.one = tf.ones((self.inputshape[-1], self.units), dtype=tf.float32)
        else:
            pass
            
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
        
        kernel = tf.sparse.SparseTensor(
        indices=self.indices, values=self.w,
        dense_shape=(self.inputshape[-1],self.units)
        )
        kernel=tf.sparse.reorder(kernel,name="kernel")
        
        kernel = tf.sparse.to_dense(kernel)
        if self.activity:
            kernel = kernel+tf.multiply(self.A,self.one-self.mask)
            
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

    print("View\n")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(28 * 28), dtype='float32'))
    model.add(EntropyLayer(units=1000, use_bias=False,flux_in=True,flux_out=False,activity=True , activation=None))
    model.add(EntropyLayer(units=10, use_bias=False,flux_in=True,flux_out=False,activity=True ,activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()