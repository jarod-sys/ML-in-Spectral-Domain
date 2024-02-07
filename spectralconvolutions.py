

import math
import numpy as np
import tensorflow as tf
from utilsSimpleConv2D import*
from tensorflow.keras.layers import Layer
from typing import Tuple,List,Any,Dict
from tensorflow.python.keras import activations, initializers, regularizers, constraints

class SimpleSpectralConv2D(Layer):

    def __init__(self ,filters,
                 kernel_size=3,
                 strides=1,
                 padding='VALID',
                 use_lambda_out=False,
                 use_lambda_in=False,
                 trainable_SM_kernel=True,
                 use_bias=False,
                 kernel_initializer="glorot_uniform",
                 activation="relu"):

        super(SimpleSpectralConv2D ,self).__init__()

        self.filters =filters
        self.strides =strides
        self.padding =padding
        self.use_bias =use_bias
        self.use_lambda_in=use_lambda_in
        self.use_lambda_out=use_lambda_out
        self.trainable_SM_kernel=trainable_SM_kernel
       
        self.kernel_size =kernel_size
        self.activation =activations.get(activation)
        self.initializer =initializers.RandomUniform(-1, 1)
        self.kernel_initializer = initializers.get(kernel_initializer)



    def build(self ,input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = input_shape[-1]
        self.pad = math.floor((self.kernel_size - 1) / 2)

        # -----------------------------------------matrix_pad-----------------------------------------------
        if self.padding == "SAME":
            # Right_shape
            self.Right_shape: Tuple = input_shape[1] + 2 *self.pad, input_shape[2 ] + 2 *self.pad
            inputShape =input_shape[1] ,input_shape[2]
            # matrix_pad
            self.matrix_pad =build_matrix_padding(input_shape=inputShape, pad=self.pad)
            self.compute_output_shape()
            if self.strides >1:
                raise Exception("Not implemented: paddind=SAME and strides>1. if padding=SAME, strides=1")
            self.matrix_strides=build_matrix_strides(self.out_shape,self.strides)
            

        elif self.padding =="VALID":
            # Right_shape
            self.Right_shape: Tuple =input_shape[1], input_shape[2]
            width_matrix_padding: int = input_shape[1] * input_shape[2]
            # matrix_pad
            self.matrix_pad = matrix = tf.constant(np.identity(width_matrix_padding), dtype="float32")
            self.compute_output_shape()
            self.matrix_strides=build_matrix_strides(self.out_shape,self.strides)
            
        else:
            raise Exception("Padding not found")


        # ------------------------------------------------------------------------------------

        # ------------------------------out_in_shape_kernel_indices---------------------------
        self.output_lenght: int = self.out_shape[1] * self.out_shape[2]
        self.size_matrix_toeplitz: int = self.Right_shape[0] * self.Right_shape[1]
        self.set_indices_kernel()
        # -------------------------------------------------------------------------------------

        
        # SM_kernel
        if self.trainable_SM_kernel:
            self.SM_kernel = self.add_weight(
            name='SM_kernel',
            shape=(self.filters,self.kernel_size*self.kernel_size),
            initializer=self.initializer,
            dtype=tf.float32,
            trainable=self.trainable_SM_kernel)
        else:
            self.SM_kernel=tf.constant(np.random.random((self.filters,self.kernel_size*self.kernel_size)),dtype="float32") 
        
        
        # Lambda_in
        if self.use_lambda_in:
            self.Lambda_in = self.add_weight(
                name='Lambda_in',
                shape=(1, self.size_matrix_toeplitz),
                initializer=self.initializer,
                dtype=tf.float32,
                trainable=self.use_lambda_in)
        
        else:
            self.Lambda_in=tf.constant(np.ones((1, self.size_matrix_toeplitz)),dtype="float32")
            

        # Lambda_out
        if self.use_lambda_out:
            self.Lambda_out = self.add_weight(
                name='Lambda_out',
                shape=(self.output_lenght, 1),
                initializer=self.initializer,
                dtype=tf.float32,
                trainable=self.use_lambda_out)
          
        else:
            self.Lambda_out=tf.constant(np.zeros((self.output_lenght, 1)),dtype="float32")

        # --------------------------------------------bias---------------------------------------
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                dtype=tf.float32,
                trainable=self.use_bias)
        else:
            self.bias = None
        # ---------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------
        self.build = True
        # ---------------------------------------------------------------------------------------
        
    def set_indices_kernel(self,*args):
        self.indices: List[Tuple]=list()
        for filters in range(self.filters):
            count: int = 1
            shift: int=0
            for i in range(self.output_lenght):
                if i== count*(self.out_shape[2]):
                    count+=1
                    shift+=2*self.pad+1
                else:
                    shift+=1
                for block in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        self.indices.append((filters,i,block*self.Right_shape[1]+shift-1+j))

    def get_indices_kernel(self,*args):
        return self.indices
    
    def compute_output_shape(self, *args):
        out_shape: Tuple = self.Right_shape[0] - 2 * self.pad, self.Right_shape[1] - 2 * self.pad
        self.out_shape: Tuple = -1, out_shape[0], out_shape[1], self.filters
        height = math.floor(self.out_shape[1] /self.strides)
        width = math.floor(self.out_shape[2] / self.strides)
        self.output_strides_shape=(height,width)

    def call(self, inputs):
        
        flatten =tf.reshape(inputs, shape=(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]))
        upFlatten = tf.matmul(a=self.matrix_pad, b=flatten)
        #-----------------------------------------------------------------------------------------------------
        kernel=tf.repeat(self.SM_kernel, repeats=self.output_lenght, axis=0, name=None)
         
        kernel=tf.reshape(kernel,shape=(-1,self.filters*self.output_lenght*self.kernel_size*self.kernel_size))

        kernel=tf.sparse.SparseTensor(
        indices=self.indices, values=kernel[0], dense_shape=(self.filters,self.output_lenght,self.size_matrix_toeplitz)
        )

        kernel=tf.sparse.to_dense(kernel)
        
        kernel=tf.linalg.matmul(kernel,tf.linalg.diag(self.Lambda_in[0,:], k = 0))-tf.linalg.matmul(tf.linalg.diag(self.Lambda_out[:,0], k = 0),kernel)

        outputs=tf.matmul(a=kernel,b=upFlatten)  

        outputs = tf.reshape(tf.matmul(a=self.matrix_strides, b=outputs),shape=(-1,self.output_strides_shape[0],
                                                                               self.output_strides_shape[1],outputs.shape[2]))

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            pass

        return outputs
    
    
    def get_kernel(self,*args):
        return tf.reshape(self.SM_kernel,shape=(self.filters,self.kernel_size,self.kernel_size))

