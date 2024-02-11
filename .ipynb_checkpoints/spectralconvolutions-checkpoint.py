

import math
import numpy as np
import tensorflow as tf
from utilsSimpleConv2D import*
from tensorflow.keras.layers import Layer
from typing import Tuple,List,Any,Dict
from tensorflow.python.keras import activations, initializers, regularizers, constraints


class SpectralConv2D_one(Layer):

    def __init__(self, filters,
                 kernel_size=3,
                 strides=1,
                 padding='VALID',
                 use_lambda_out=False,
                 use_lambda_in=False,
                 trainable_SM_kernel=True,
                 use_bias=False,
                 kernel_initializer="glorot_uniform",
                 activation="relu"):

        super(SpectralConv2D_one, self).__init__()

        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_size = kernel_size
        self.use_lambda_in = use_lambda_in
        self.use_lambda_out = use_lambda_out
        self.trainable_SM_kernel = trainable_SM_kernel

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = input_shape[-1]
        self.pad = math.floor((self.kernel_size - 1) / 2)

        # -----------------------------------------matrix_pad-----------------------------------------------
        if self.padding == "SAME":
            if self.strides > 1:
                raise Exception("Not implemented: paddind=SAME and strides>1. if padding=SAME, strides=1")

            # Right_shape
            self.Right_shape: Tuple = input_shape[1] + 2 * self.pad, input_shape[2] + 2 * self.pad
            inputShape = input_shape[1], input_shape[2]
            # matrix_pad
            self.matrix_pad = build_matrix_padding(input_shape=inputShape, pad=self.pad)

        elif self.padding == "VALID":
            # Right_shape
            self.Right_shape: Tuple = input_shape[1], input_shape[2]
            # matrix_pad
            self.matrix_pad = matrix = tf.constant(np.identity(input_shape[1] * input_shape[2]), dtype="float32")

        else:
            raise Exception("Padding not found")

        # ------------------------------------------------------------------------------------

        # ------------------------------out_in_shape_phi_indices---------------------------
        self.set_indices_phi()
        # -------------------------------------------------------------------------------------

        # SM_kernel
        if self.trainable_SM_kernel:
            self.SM_kernel = self.add_weight(
                name='SM_kernel',
                shape=(self.filters, self.kernel_size * self.kernel_size),
                initializer=self.kernel_initializer,
                dtype=tf.float32,
                trainable=self.trainable_SM_kernel)
        
        else:
            self.SM_kernel = tf.constant(np.random.random((self.filters, self.kernel_size * self.kernel_size)),
                                         dtype="float32")

        # Lambda_in
        if self.use_lambda_in:
            self.Lambda_in =self.add_weight(name='Lambda_in',
                                        shape=(1, self.Right_shape[0] * self.Right_shape[1]),
                                        initializer=tf.ones_initializer(),
                                        dtype=tf.float32,
                                        trainable=self.use_lambda_in)
        else:
            
            self.Lambda_in = tf.constant(np.ones((1, self.Right_shape[0] * self.Right_shape[1])),
                                         dtype="float32")
            
        
        # Lambda_out
        if self.use_lambda_out:
            self.Lambda_out=self.add_weight(name='Lambda_out',
                                            shape=(self.output_lenght, 1),
                                            initializer=tf.zeros_initializer(),
                                            dtype=tf.float32,
                                            trainable=self.use_lambda_out)
        
        else:
            self.Lambda_out = tf.constant(np.zeros((self.output_lenght, 1)),
                                         dtype="float32")
            
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


    def set_indices_phi(self, *args):
        self.indices: List[Tuple] = list()

        self.out_shape1: int = math.floor((self.Right_shape[0] - self.kernel_size) / self.strides) + 1
        self.out_shape2: int = math.floor((self.Right_shape[1] - self.kernel_size) / self.strides) + 1
        self.output_lenght: int = self.out_shape1 * self.out_shape2

        for filters in range(self.filters):
            count: int = 1
            shift: int = 0
            for i in range(self.output_lenght):
                if i == count * (self.out_shape2):
                    count += 1
                    shift += self.kernel_size + (self.strides - 1) * self.Right_shape[1]
                else:
                    if shift:
                        shift += self.strides
                    else:
                        shift += 1
                for block in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        self.indices.append((filters, i, block * self.Right_shape[1] + shift - 1 + j))

    def get_indices_phi(self, *args):
        return self.indices

    def call(self, inputs):

        flatten = tf.reshape(inputs, shape=(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]))
        upFlatten = tf.matmul(a=self.matrix_pad, b=flatten)
        # -----------------------------------------------------------------------------------------------------
        kernel = tf.repeat(self.SM_kernel, repeats=self.output_lenght, axis=0, name=None)

        kernel = tf.reshape(kernel, shape=(-1, self.filters * self.output_lenght * self.kernel_size * self.kernel_size))

        kernel = tf.sparse.SparseTensor(
            indices=self.indices, values=kernel[0],
            dense_shape=(self.filters, self.output_lenght, self.Right_shape[0] * self.Right_shape[1])
        )

        kernel = tf.sparse.to_dense(kernel)

        kernel = tf.linalg.matmul(kernel, tf.linalg.diag(self.Lambda_in[0, :], k=0)) - tf.linalg.matmul(
            tf.linalg.diag(self.Lambda_out[:, 0], k=0), kernel)

        outputs = tf.matmul(a=kernel, b=upFlatten)

        outputs = tf.reshape(outputs, shape=(-1, self.out_shape1, self.out_shape2, outputs.shape[2]))

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            pass

        return outputs

    def get_omega(self,*args):
        return tf.reshape(self.SM_kernel, shape=(self.filters,self.kernel_size, self.kernel_size))

    def get_kernel(self, *args):
        kernel = tf.repeat(self.SM_kernel, repeats=self.output_lenght, axis=0, name=None)

        kernel = tf.reshape(kernel, shape=(-1, self.filters * self.output_lenght * self.kernel_size * self.kernel_size))

        kernel = tf.sparse.SparseTensor(
            indices=self.indices, values=kernel[0],
            dense_shape=(self.filters, self.output_lenght, self.Right_shape[0] * self.Right_shape[1])
        )

        kernel = tf.sparse.to_dense(kernel)

        kernel = tf.linalg.matmul(kernel, tf.linalg.diag(self.Lambda_in[0, :], k=0)) - tf.linalg.matmul(
            tf.linalg.diag(self.Lambda_out[:, 0], k=0), kernel)
        return kernel


class SpectralConv2D_two(Layer):

    def __init__(self, filters,
                 kernel_size=3,
                 strides=1,
                 padding='VALID',
                 use_lambda_in=True,
                 use_bias=False,
                 activation="relu"):

        super(SpectralConv2D_two, self).__init__()

        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.use_lambda_in = use_lambda_in

        self.kernel_size = kernel_size
        self.activation = activations.get(activation)
        self.initializer = initializers.RandomUniform(-1, 1)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = input_shape[-1]
        self.pad = math.floor((self.kernel_size - 1) / 2)
    
        # -----------------------------------------matrix_pad-----------------------------------------------
        if self.padding == "SAME":
            if self.strides > 1:
                raise Exception("Not implemented: paddind=SAME and strides>1. if padding=SAME, strides=1")
    
            # Right_shape
            self.Right_shape: Tuple = input_shape[1] + 2 * self.pad, input_shape[2] + 2 * self.pad
            inputShape = input_shape[1], input_shape[2]
            # matrix_pad
            self.matrix_pad = build_matrix_padding(input_shape=inputShape, pad=self.pad)
            # Jacobien_strides
            self.Build_J()
            # ------------------------------out_in_shape_phi_indices---------------------------
            self.set_indices_phi(N=self.J2.shape[0],M=self.J1.shape[1])
            # -----------------------------------------------------------------------------------
    
        elif self.padding == "VALID":
            # Right_shape
            self.Right_shape: Tuple = input_shape[1], input_shape[2]
            # matrix_pad
            self.matrix_pad = matrix = tf.constant(np.identity(input_shape[1] * input_shape[2]), dtype="float32")
            
            if self.strides>self.kernel_size:
                raise Exception("Not implemented")
            else:
                # Jacobien_strides
                self.Build_J()
                # ------------------------------out_in_shape_phi_indices---------------------------
                self.set_indices_phi(N=self.J2.shape[0],M=self.J1.shape[1])
                # -----------------------------------------------------------------------------------
            
            
    
        else:
            raise Exception("Padding not found")
        # ------------------------------------------------------------------------------------

        
        # noyau_of_phi
        self.noyau_of_phi = tf.constant(np.ones((self.filters,
                                                self.kernel_size * self.kernel_size)),
                                                dtype="float32")
        # -------------------------------------------------------------------------------------

        # Lambda_in
        if self.use_lambda_in:
            self.Lambda_in = self.add_weight(
                name='Lambda_in',
                shape=(1, self.J2.shape[0]* self.J1.shape[1]),
                initializer=tf.ones_initializer(),
                dtype=tf.float32,
                trainable=self.use_lambda_in)

        else:
            raise Exception("Not implemented.")

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
    
    def set_indices_phi(self,N:int,M:int, *args):
        self.indices: List[Tuple] = list()
    
        self.out_shape1: int = math.floor((N - self.kernel_size) / self.kernel_size) + 1
        self.out_shape2: int = math.floor((M - self.kernel_size) / self.kernel_size) + 1
        self.output_lenght: int = self.out_shape1 * self.out_shape2
    
        for filters in range(self.filters):
            count: int = 1
            shift: int = 0
            for i in range(self.output_lenght):
                if i == count * (self.out_shape2):
                    count += 1
                    shift += self.kernel_size + (self.kernel_size - 1) * M
                else:
                    if shift:
                        shift += self.kernel_size
                    else:
                        shift += 1
                for block in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        self.indices.append((filters, i, block * M + shift - 1 + j))

    def Build_J(self,*args):
        out_shape1:int=math.floor((self.Right_shape[0]-self.kernel_size)/self.strides) + 1
        out_shape2:int=math.floor((self.Right_shape[1]-self.kernel_size)/self.strides) + 1
    
        row = tf.Variable(np.zeros(shape=(self.Right_shape[1],1)), dtype=tf.float32, trainable=False)
        row[0,0].assign(1)
        for j in range(out_shape2):
            for k in range(self.kernel_size):
                try:
                    new_line=shift_(row,k,axis=0)
                    self.J1 = tf.concat([self.J1, new_line], 1)
                except:
                    self.J1 = row
            row=shift_(row, self.strides,axis=0)
            
        del new_line
        
        col = tf.Variable(np.zeros(shape=(1,self.Right_shape[0])), dtype=tf.float32, trainable=False)
        col[0,0].assign(1)
        
        for i in range(out_shape1):
            for l in range(self.kernel_size):
                try:
                    new_line=shift_(col,l,axis=1)
                    self.J2 = tf.concat([self.J2, new_line], 0)
                except:
                    self.J2 = col
            col=shift_(col, self.strides,axis=1)

    def get_indices_phi(self, *args):
        return self.indices

    def call(self, inputs):

        flatten = tf.reshape(inputs, shape=(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]))
        upFlatten = tf.matmul(a=self.matrix_pad, b=flatten)
        inputs_x=tf.reshape(upFlatten, shape=(-1, inputs.shape[3],self.Right_shape[0] , self.Right_shape[0]))

        
        inputs_y=tf.matmul(a=self.J2,b=tf.matmul(a=inputs_x,b=self.J1))
        inputs_y=tf.reshape(inputs_y,shape=(-1, inputs_y.shape[2] * inputs_y.shape[3], inputs_y.shape[1]))
        
        # -----------------------------------------------------------------------------------------------------

        kernel = tf.repeat(self.noyau_of_phi, repeats=self.output_lenght, axis=0, name=None)

        kernel = tf.reshape(kernel, shape=(-1, self.filters * self.output_lenght * self.kernel_size * self.kernel_size))

        kernel = tf.sparse.SparseTensor(
            indices=self.indices, values=kernel[0],
            dense_shape=(self.filters, self.output_lenght, self.J1.shape[1] * self.J2.shape[0])
        )

        kernel = tf.sparse.to_dense(kernel)
        kernel = tf.linalg.matmul(kernel, tf.linalg.diag(self.Lambda_in[0, :], k=0))
        # -----------------------------------------------------------------------------------------------------

        outputs = tf.matmul(a=kernel, b=inputs_y)

        outputs = tf.reshape(outputs, shape=(-1, self.out_shape1, self.out_shape1, outputs.shape[2]))

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            pass

        return outputs


    def get_kernel(self, *args):
        kernel = tf.repeat(self.noyau_of_phi, repeats=self.output_lenght, axis=0, name=None)
        kernel = tf.reshape(kernel, shape=(-1, self.filters * self.output_lenght * self.kernel_size * self.kernel_size))
        kernel = tf.sparse.SparseTensor(
            indices=self.indices, values=kernel[0],
            dense_shape=(self.filters, self.output_lenght, self.Right_shape[0] * self.Right_shape[1])
        )
        kernel = tf.sparse.to_dense(kernel)
        kernel = tf.linalg.matmul(kernel, tf.linalg.diag(self.Lambda_in[0, :], k=0))
        return kernel

