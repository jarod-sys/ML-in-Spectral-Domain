
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Tuple,List,Any,Dict
from tensorflow.python.keras import activations, initializers, regularizers, constraints

def shift_(weight:tf.Tensor, strides: int):
    return  tf.roll(weight, shift=strides, axis=1)


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