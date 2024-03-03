from typing import Tuple, List

from tensorflow.keras.layers import Layer, Dense
from tensorflow.python.keras import activations, initializers

from utilsSimpleConv2D import *


class SpecConv2D(Layer):

    def __init__(self, filters,
                 kernel_size=3,
                 use_lambda_in=True,
                 use_bias=False,
                 activation="relu"):

        super(SpecConv2D, self).__init__()

        self.filters = filters
        self.use_bias = use_bias
        self.kernel_size = kernel_size
        self.use_lambda_in = use_lambda_in
        self.activation = activations.get(activation)

        
        self.initializer = initializers.RandomUniform(-1, 1)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.input_channel = input_shape[-1]
        # ------------------------------out_in_shape_phi_indices---------------------------
        self.set_indices_phi(N=input_shape[1],M=input_shape[2])
        # -----------------------------------------------------------------------------------
    
        # --------------------------------noyau_of_phi----------------------------------------
        self.noyau_of_phi = tf.constant(np.ones((self.filters,
                                                self.kernel_size * self.kernel_size)),
                                                dtype="float32")
        
        # -----------------------------------kernel-------------------------------------------
        kernel = tf.repeat(self.noyau_of_phi, repeats=self.output_lenght, axis=0, name=None)
        
        kernel = tf.reshape(kernel, shape=(-1, self.filters * self.output_lenght * self.kernel_size * self.kernel_size))
        
        kernel = tf.sparse.SparseTensor(
        indices=self.indices, values=kernel[0],
        dense_shape=(self.filters, self.output_lenght, input_shape[1]*input_shape[2])
        )
        
        self.kernel = tf.sparse.to_dense(kernel)
        # -------------------------------------------------------------------------------------

        # Lambda_in
        if self.use_lambda_in:
            self.Lambda_in = self.add_weight(
                name='Lambda_in',
                shape=(1, input_shape[1]*input_shape[2]),
                initializer=self.initializer,
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

    def get_indices_phi(self, *args):
        return self.indices

    def call(self, inputs):

        
        # -----------------------------------------------------------------------------------------------------
        flatten = tf.reshape(inputs, shape=(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]))
        # -----------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------
        if 0<self.filters<2:
            outputs = tf.matmul(a=tf.linalg.matmul(self.kernel, tf.linalg.diag(self.Lambda_in[0, :], k=0)), b=flatten)
            outputs = tf.reshape(outputs, shape=(-1, self.out_shape1, self.out_shape2,self.input_channel))
            # -----------------------------------------------------------------------------------------------------#
        else:
             raise Exception("Not implemented for this filter.")

       
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            pass

        return outputs


    def get_kernel(self, *args):
        return self.kernel


class SpectralConv2D_T(Layer):

    def __init__(self, filters,
                 kernel_size=3,
                 strides=1,
                 padding='VALID',
                 use_lambda_out=False,
                 use_lambda_in=False,
                 use_encode=False,
                 use_decode=False,
                 trainable_omega_diag=True,
                 trainable_omega_triu=True,
                 trainable_omega_tril=True,
                 use_bias=False,
                 kernel_initializer="glorot_uniform",
                 activation="relu"):

        super(SpectralConv2D_T, self).__init__()

        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_size = kernel_size

        self.use_encode = use_encode
        self.use_decode = use_decode

        self.use_lambda_in = use_lambda_in
        self.use_lambda_out = use_lambda_out

        self.trainable_omega_tril = trainable_omega_tril
        self.trainable_omega_triu = trainable_omega_triu
        self.trainable_omega_diag = trainable_omega_diag

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.size_omega_part = self.filters * self.kernel_size * math.floor((self.kernel_size - 1) / 2)
        self.pad = math.floor((self.kernel_size - 1) / 2)

        # -----------------------------------------matrix_pad------------------------------------
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
            self.matrix_pad = tf.constant(np.identity(input_shape[1] * input_shape[2]), dtype="float32")

        else:
            raise Exception("Padding not found")

        # -------------------------------------------------------------------------------------

        # ------------------------------out_in_shape_phi_indices-------------------------------
        self.set_indices_phi()
        self.set_indices_ul_triangular()
        # -------------------------------------------------------------------------------------

        # \omega_diag
        if self.trainable_omega_diag:
            self.omega_diag = self.add_weight(
                name='omega_diag',
                shape=(self.filters, self.kernel_size,),
                initializer=self.kernel_initializer,
                dtype=tf.float32,
                trainable=self.trainable_omega_diag)

        else:
            self.omega_diag = tf.constant(
                np.random.uniform(size=(self.filters, self.kernel_size,), low=-0.05, high=0.05), dtype=tf.float32,
                name='omega_diag')

        # \omega_triu
        if self.trainable_omega_triu:
            self.omega_triu = self.add_weight(
                name='omega_triu',
                shape=(1, self.size_omega_part),
                initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None),
                dtype=tf.float32,
                trainable=self.trainable_omega_triu)

        else:
            self.omega_triu = tf.constant(np.random.uniform(size=(1, self.size_omega_part), low=-0.05, high=0.05),
                                          dtype=tf.float32, name='omega_triu')

        # \omega_tril
        if self.trainable_omega_tril:
            self.omega_tril = self.add_weight(
                name='omega_tril',
                shape=(1, self.size_omega_part),
                initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None),
                dtype=tf.float32,
                trainable=self.trainable_omega_tril)

        else:
            self.omega_tril = tf.constant(np.random.uniform(size=(1, self.size_omega_part), low=-0.05, high=0.05),
                                          dtype=tf.float32, name='omega_tril')

        # ---------------------------------indeice_Omega:part-----------------------------------
        self.set_indices_ul_triangular()
        # -------------------------------------------------------------------------------------

        # \use_lambda_in
        if self.use_lambda_in:
            self.use_lambda_in = self.add_weight(name='use_lambda_in',
                                                 shape=(self.kernel_size, 1),
                                                 initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05,
                                                                                           seed=None),
                                                 dtype=tf.float32,
                                                 trainable=self.use_lambda_in)

        else:
            self.use_lambda_in = tf.random.uniform(shape=(self.kernel_size, 1), minval=-0.05, maxval=0.05,
                                                   dtype=tf.float32, name='use_lambda_in')

        # \use_lambda_out
        if self.use_lambda_out:
            self.use_lambda_out = self.add_weight(name='use_lambda_out',
                                                  shape=(1, self.kernel_size),
                                                  initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05,
                                                                                            seed=None),
                                                  dtype=tf.float32,
                                                  trainable=self.use_lambda_out)
        else:
            self.use_lambda_out = tf.random.uniform(shape=(1, self.kernel_size), minval=-0.05, maxval=0.05,
                                                    dtype=tf.float32, name='use_lambda_out')

        # use_encode
        if self.use_encode:
            self.use_encode = self.add_weight(name='use_encode',
                                              shape=(1, self.Right_shape[0] * self.Right_shape[1]),
                                              initializer=tf.ones_initializer(),
                                              dtype=tf.float32,
                                              trainable=self.use_encode)
        else:

            self.use_encode = tf.ones(shape=(1, self.Right_shape[0] * self.Right_shape[1]), dtype=tf.float32,
                                      name='use_encode')

        # use_decode
        if self.use_decode:
            self.use_decode = self.add_weight(name='use_decode',
                                              shape=(self.output_lenght, 1),
                                              initializer=tf.zeros_initializer(),
                                              dtype=tf.float32,
                                              trainable=self.use_decode)

        else:
            self.use_decode = tf.zeros(shape=(self.output_lenght, 1), dtype=tf.float32, name='use_decode')

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

    def call(self, inputs):

        # ----------------------------------Inputs---------------------------------------------
        flatten = tf.reshape(inputs, shape=(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]))
        upFlatten = tf.matmul(a=self.matrix_pad, b=flatten)
        upFlatten = tf.math.reduce_mean(upFlatten, axis=-1)

        # ----------------------------------\Omega---------------------------------------------
        omega_high = tf.sparse.SparseTensor(
            indices=self.indices_triu, values=self.omega_triu[0],
            dense_shape=(self.filters, self.kernel_size, self.kernel_size)
        )
        omega_lower = tf.sparse.SparseTensor(
            indices=self.indices_tril, values=self.omega_tril[0],
            dense_shape=(self.filters, self.kernel_size, self.kernel_size)
        )
        omega_high = tf.sparse.to_dense(omega_high)
        omega_lower = tf.sparse.to_dense(omega_lower)
        omega = omega_lower + tf.linalg.diag(self.omega_diag, k=0) + omega_high

        # ----------------------------------Base-----------------------------------------------

        base = tf.multiply(omega, self.use_lambda_in - self.use_lambda_out)

        # ------------------------------Build Noyau--------------------------------------------

        kernel = tf.reshape(base, shape=(self.filters, self.kernel_size * self.kernel_size))

        kernel = tf.repeat(kernel, repeats=self.output_lenght, axis=0, name=None)

        kernel = tf.reshape(kernel, shape=(-1, self.filters * self.output_lenght * self.kernel_size * self.kernel_size))

        kernel = tf.sparse.SparseTensor(
            indices=self.indices, values=kernel[0],
            dense_shape=(self.filters, self.output_lenght, self.Right_shape[0] * self.Right_shape[1])
        )

        kernel = tf.sparse.to_dense(kernel)

        kernel = tf.linalg.matmul(kernel, tf.linalg.diag(self.use_encode[0, :], k=0)) - tf.linalg.matmul(
            tf.linalg.diag(self.use_decode[:, 0], k=0), kernel)

        # -----------------------------------Outputs-------------------------------------------------
        outputs = tf.transpose(tf.matmul(a=kernel, b=upFlatten, transpose_b=True))
        outputs = tf.reshape(outputs, shape=(-1, self.out_shape1, self.out_shape2, self.filters))

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            pass

        return outputs

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

    def set_indices_ul_triangular(self, *args):
        self.indices_triu: List[Tuple] = list()
        self.indices_tril: List[Tuple] = list()

        for filters in range(self.filters):
            for i in range(1, self.kernel_size):
                for j in range(i):
                    self.indices_tril.append((filters, i, j))

        for filters in range(self.filters):
            for i in range(self.kernel_size):
                for j in range(i + 1, self.kernel_size):
                    self.indices_triu.append((filters, i, j))

    def get_indices_phi(self, *args):
        return self.indices

    def get_base(self, *args):
        # ----------------------------------\Omega---------------------------------------------
        omega_high = tf.sparse.SparseTensor(
            indices=self.indices_triu, values=self.omega_triu[0],
            dense_shape=(self.filters, self.kernel_size, self.kernel_size)
        )
        omega_lower = tf.sparse.SparseTensor(
            indices=self.indices_tril, values=self.omega_tril[0],
            dense_shape=(self.filters, self.kernel_size, self.kernel_size)
        )
        omega_high = tf.sparse.to_dense(omega_high)
        omega_lower = tf.sparse.to_dense(omega_lower)
        omega = omega_lower + tf.linalg.diag(self.omega_diag, k=0) + omega_high
        # ----------------------------------Base-----------------------------------------------

        base = tf.multiply(omega, self.use_lambda_in - self.use_lambda_out)
        return base


if __name__=='__main__':

    epochs = 10
    batch_size=200
    use_bias=False
    learning_rate=0.003
    activation=None
    parameters={ "use_lambda_out":False,
                "use_lambda_in":False,
                    "use_encode":False,
                    "use_decode":False,
                    "trainable_omega_diag":True,
                    "trainable_omega_triu":True,
                    "trainable_omega_tril":True,
                    "kernel_initializer":"glorot_uniform"
            }


    model= tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(28,28,1)))


    model.add(SpectralConv2D_T(filters=1,kernel_size=3,use_bias=use_bias,activation="relu",**parameters))
    model.add(tf.keras.layers.MaxPooling2D((2,2))) 

# """  """
    model.add(tf.keras.layers.Flatten())  

    model.add(Dense(1000,use_bias=use_bias,activation=activation))
    model.add(Dense(10, use_bias=use_bias, activation='softmax'))

        
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()




    #------------------------------\train_model---------------------------------------------
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    x_train, x_test=x_train.reshape(-1,28,28,1), x_test.reshape(-1,28,28,1)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    history=model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1)
