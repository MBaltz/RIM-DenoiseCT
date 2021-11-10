import tensorflow as tf
import numpy as np
from ConvGRU2D import ConvGRU2D

class RNN(tf.keras.Model):
    def __init__(self, in_size):
        super(RNN, self).__init__()
        self.in_size = in_size

    def build(self, input_shape):
        self.hidden_1_conv = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=2,
            padding="same", activation="tanh", name="hidden_1_conv")

        self.hidden_2_rnn = ConvGRU2D(
            filters=16, kernel_size=3, strides=1, padding="same",
            return_state=False, name="hidden_2_rnn")
        
        self.hidden_3_convTranspose = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, padding="same",
            activation="tanh", name="hidden_3_convTranspose")

        self.hidden_4_conv = tf.keras.layers.Conv2D(
            filters=1, kernel_size=3, strides=1,
            padding="same", activation="tanh", name="hidden_4_conv")


    def call(self, x, grad, training=True):
        # x = (batch_size, 1000, 513, 2)
        x = tf.reshape(tf.squeeze(tf.stack([x, grad]), axis=-1), [self.in_size[0], self.in_size[1], self.in_size[2], 2])
        out = self.hidden_1_conv(x)
        out = tf.expand_dims(out, axis=1)
        out = self.hidden_2_rnn(out)
        out = self.hidden_3_convTranspose(out)
        out = self.hidden_4_conv(out) # tanh pois será somado

        # valor que será somado à x e gerando x_t+1
        return out



class RIM(tf.keras.Model):

    def __init__(self, qnt_recurrence, in_size):
        super(RIM, self).__init__()
        self.qnt_recurrence = qnt_recurrence
        self.in_size = in_size
        # Dimensão de entrada é (batch_size, x, y)

    def build(self, input_shape):
        self.rnn = RNN(self.in_size)

    def call(self, x, training=True):
        # x = tf.expand_dims(x, axis=-1) # (batch_size, 1000, 513, 1)

        noised = tf.Variable(x, trainable=False) # Copia x para salvar a imagem ruidosa primária
        # Primeiro hidden state [0 .. 0] com dim=(batch_size, 1000, 513, 1)
        # hidden = tf.Variable(tf.zeros(x.shape), trainable=False)

        for r in range(self.qnt_recurrence):
            delta_x_noised = self.__gradient_x_noised(noised, x)

            to_att_x = self.rnn(x, delta_x_noised)

            x = x + to_att_x
            print(r, np.min(x), np.max(x))
            # após somar surgem grandes valores ou negativos
            x = tf.keras.activations.sigmoid(x)
        return x

    def __gradient_x_noised(self, noised, x):
        return tf.square(tf.subtract(noised, x))
