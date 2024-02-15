
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Tuple,List,Any,Dict
from tensorflow.python.keras import activations, initializers, regularizers, constraints

def shift_(weight:tf.Tensor, strides: int,axis:int=1):
    return  tf.roll(weight, shift=strides, axis=axis)


def indices_phi(filters: int, N: int, M: int, F: int = 3, S: int = 1, *args):
    indices: List[Tuple] = list()

    out_shape1: int = math.floor((N - F) / S) + 1
    out_shape2: int = math.floor((M - F) / S) + 1
    output_lenght: int = out_shape1 * out_shape2

    for filter in range(filters):
        count: int = 1
        shift: int = 0
        for i in range(output_lenght):
            if i == count * (out_shape2):
                count += 1
                shift += F + (S - 1) * M
            else:
                if shift:
                    shift += S
                else:
                    shift += 1
            for block in range(F):
                for j in range(F):
                    indices.append((filter, i, block * M + shift - 1 + j))
    return indices

def Build_J(N:int,M:int,F:int=3,S:int=1,*args):
    out_shape1:int=math.floor((N-F)/S)+1
    out_shape2:int=math.floor((M-F)/S)+1
        
    row = tf.Variable(np.zeros(shape=(M,1)), dtype=tf.float32, trainable=False)
    row[0,0].assign(1)
    for j in range(out_shape2):
        for k in range(F):
            try:
                new_line=shift_(row,k,axis=0)
                J1 = tf.concat([J1, new_line], 1)
            except:
                J1 = row
        row=shift_(row, S,axis=0)
        
    del new_line
    col = tf.Variable(np.zeros(shape=(1,N)), dtype=tf.float32, trainable=False)
    col[0,0].assign(1)
    
    for i in range(out_shape1):
        for l in range(F):
            try:
                new_line=shift_(col,l,axis=1)
                J2 = tf.concat([J2, new_line], 0)
            except:
                J2 = col
        col=shift_(col, S,axis=1)
        
    return J1, J2

def build_matrix_strides(out_shape,strides):

    height = math.floor(out_shape[1] / strides)
    width = math.floor(out_shape[2] / strides)
    count = 1
    line = tf.Variable(np.zeros(shape=(out_shape[1] * out_shape[2])), dtype=tf.float32, trainable=False)
    line[0].assign(1)
    line = tf.reshape(line, shape=(1, line.shape[0]))
    S = line
    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                continue
            position = strides * (i * out_shape[2] + j)
            new_line = shift_(line, position)
            S = tf.concat([S, new_line], 0)
            count += 1
    return S


def build_matrix_padding(input_shape: Tuple, pad: int):
    # block
    out_shape: Tuple = input_shape[0] + 2 * pad, input_shape[1] + 2 * pad
    width_matrix_padding: int = input_shape[0] * input_shape[1]
    height_matrix_padding: int = out_shape[0] * out_shape[1]

    size_block1: Tuple = out_shape[1], width_matrix_padding
    block1 = tf.zeros(shape=size_block1)
    size_block2: Tuple = pad, width_matrix_padding
    block2 = tf.zeros(shape=size_block2)

    line = tf.Variable(np.zeros(shape=(width_matrix_padding)), dtype=tf.float32, trainable=False)
    line[0].assign(1)
    line = tf.reshape(line, shape=(1, line.shape[0]))
    # initialisation
    M = line
    new_line = line
    matrix = block1

    for i in range(1, out_shape[0] - 1):
        matrix = tf.concat([matrix, block2], 0)

        for j in range(1, input_shape[1]):
            new_line = shift_(new_line, 1)
            M = tf.concat([M, new_line], 0)
        matrix = tf.concat([matrix, M], 0)
        matrix = tf.concat([matrix, block2], 0)
        del M
        new_line = shift_(new_line, 1)
        M = new_line

    matrix = tf.concat([matrix, block1], 0)
    assert matrix.shape == (height_matrix_padding, width_matrix_padding)
    return matrix

def convolution_at(img: np.ndarray,kernel: np.ndarray,i:int,j:int)->float:
    output = 0
    kernel_shape:Tuple =kernel.shape
    img_shape: Tuple = img.shape
    center_point:int =math.floor((kernel_shape[0]-1)/2)
    height=i-center_point
    width=j-center_point
    for s in range(height,i+center_point+1):
        for r in range(width, j+ center_point+1):
            if (s<0 or s>img_shape[0]-1 or r>img_shape[1]-1 or r<0):continue
            output+=img[s,r]*kernel[s-height,r-width]
    return output

def convolution(img: np.ndarray,kernel: np.ndarray)->np.ndarray:
    img_shape:Tuple =img.shape
    output =np.zeros(shape=img_shape,dtype="float32")
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            output[i,j]=convolution_at(img=img,kernel=kernel,i=i,j=j)
    return output

class PaddingJacobiens(Layer):

    def __init__(self,
                 kernel_size=3,
                 strides=1,
                 padding='VALID'):

        super(PaddingJacobiens, self).__init__()
        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size


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
        else:
            raise Exception("Padding not found")
        # ------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------
        self.build = True
        # --------------------------------------------------------------------------------------

    def Build_J(self,*args):
        out_shape1:int=math.floor((self.Right_shape[0]-self.kernel_size)/self.strides) + 1
        out_shape2:int=math.floor((self.Right_shape[1]-self.kernel_size)/self.strides) + 1
    
        row = np.zeros(shape=(self.Right_shape[1],1))
        row[0,0]=1
        row=tf.constant(row,dtype=tf.float32)
        for j in range(out_shape2):
            for k in range(self.kernel_size):
                try:
                    new_line=shift_(row,k,axis=0)
                    self.J1 = tf.concat([self.J1, new_line], 1)
                except:
                    self.J1 = row
            row=shift_(row, self.strides,axis=0)
            
        del new_line
        
        col =np.zeros(shape=(1,self.Right_shape[0]))
        col[0,0]=1
        col=tf.constant(col, dtype=tf.float32)
        for i in range(out_shape1):
            for l in range(self.kernel_size):
                try:
                    new_line=shift_(col,l,axis=1)
                    self.J2 = tf.concat([self.J2, new_line], 0)
                except:
                    self.J2 = col
            col=shift_(col, self.strides,axis=1)

    def call(self, inputs):

        # -----------------------------------------------------------------------------------------------------
        flatten = tf.reshape(inputs, shape=(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]))
        upFlatten = tf.matmul(a=self.matrix_pad, b=flatten)
        inputs_x=tf.reshape(upFlatten, shape=(-1, inputs.shape[3],self.Right_shape[0] , self.Right_shape[0]))
        inputs_y=tf.matmul(a=self.J2,b=tf.matmul(a=inputs_x,b=self.J1))
        inputs_y=tf.reshape(inputs_y,shape=(-1, inputs_y.shape[2], inputs_y.shape[3], inputs_y.shape[1]))
        # -----------------------------------------------------------------------------------------------------
        
        # -----------------------------------------------------------------------------------------------------
        return inputs_y