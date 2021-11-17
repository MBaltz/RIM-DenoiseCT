# from tensorflow import keras
import tensorflow as tf
import numpy as np
from Network.ConvGRU2D.ConvGRU2D import ConvGRU2D

class RNN(tf.keras.layers.Layer):
    def __init__(self):
        super(RNN, self).__init__()

    def build(self, input_shape):
        self.hidden_1_conv = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=2,
            padding="same", activation="tanh", name="hidden_1_conv")

        self.hidden_2_rnn = ConvGRU2D(
            filters=256, kernel_size=3, strides=1, padding="same",
            return_state=False, name="hidden_2_rnn")
        
        self.hidden_3_convTranspose = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, padding="same",
            activation="tanh", name="hidden_3_convTranspose")

        self.hidden_4_conv = tf.keras.layers.Conv2D(
            filters=1, kernel_size=3, strides=1, padding="same",
            activation="tanh", name="hidden_4_conv")


    def call(self, x_grad, training=True):
        # x_grad.shape = (batch_size, 362, 362, 1)

        out = self.hidden_1_conv(x_grad)
        out = tf.expand_dims(out, axis=1)
        out = self.hidden_2_rnn(out)
        out = self.hidden_3_convTranspose(out)
        out = self.hidden_4_conv(out) # tanh pois será somado
        # Valor que será somado à x e gerando x_t+1
        return out



class RIM(tf.keras.Model):

    def __init__(self, qnt_recurrence):
        super(RIM, self).__init__()
        self.qnt_recurrence = qnt_recurrence

    def build(self, input_shape):
        self.rnn = RNN()

    def call(self, x, training=True): # x.shape = (batch_size, 362, 362, 1)
        
        # Salva a imagem ruidosa primária
        noised = tf.Variable(x, trainable=False)

        for r in range(self.qnt_recurrence):
            delta_x_noised = self.__gradient_x_noised(noised, x)
            

            # x/grad.shape = (batch_size, 362, 362, 1)
            x_grad = tf.squeeze(tf.stack([x, delta_x_noised], axis=-1), axis=-2)

            to_att_x = self.rnn(x_grad)
            x = x + to_att_x

            # Após somar pode surgir valores negativos ou grandes positivos
            x = tf.keras.activations.relu(x) #TODO: testar dnv usar sigmoid
        return x

    def __gradient_x_noised(self, noised, x):
        return tf.square(tf.subtract(noised, x))
