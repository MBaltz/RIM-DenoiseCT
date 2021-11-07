import tensorflow as tf

class RNN(tf.keras.Model):
    def __init__(self, in_size):
        super(RNN, self).__init__()
        self.in_size = in_size

    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=8, kernel_size=(3,3), strides=(1,1),
            padding="same", activation="relu", name="conv_1", use_bias=False)
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(3,3), strides=(1,1),
            padding="same", activation="relu", name="conv_2", use_bias=False)
        self.conv_3 = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3,3), strides=(1,1),
            padding="same", activation="tanh", name="conv_3", use_bias=False) # tanh (pois será somado)

        self.conv_hidden = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(3,3), strides=(1,1),
            padding="same", activation="relu", name="conv_hidden", use_bias=False)

    def call(self, x, grad, hidden, training=True):
        x_bkp = tf.Variable(x, trainable=False)
        grad_bkp = tf.Variable(grad, trainable=False)
        # x = (batch_size, 1000, 513, 2)
        x = tf.reshape(tf.stack([x, grad]), [self.in_size[0], self.in_size[1], self.in_size[2], 2])
        out = self.conv_1(x)
        out = self.conv_2(out)

        new_hidden = self.conv_hidden(
            tf.reshape(tf.stack([x_bkp, grad_bkp, hidden]), [self.in_size[0], self.in_size[1], self.in_size[2], 3]))
        
        to_att_x = self.conv_3(tf.multiply(out, new_hidden))

        # valor que será somado à x e gerando x_t+1 e o novo hidden state
        return to_att_x, new_hidden



class RIM(tf.keras.Model):

    def __init__(self, qnt_recurrence, in_size):
        super(RIM, self).__init__()
        self.qnt_recurrence = qnt_recurrence
        self.in_size = in_size
        # Dimensão de entrada é (batch_size, x, y)

    def build(self, input_shape):
        self.rnn = RNN(input_shape)

    def call(self, x, training=True):
        # x = tf.expand_dims(x, axis=-1) # (batch_size, 1000, 513, 1)

        noised = tf.Variable(x, trainable=False) # Copia x para salvar a imagem ruidosa primária
        # Primeiro hidden state [0 .. 0] com dim=(batch_size, 1000, 513, 1)
        hidden = tf.Variable(tf.zeros(x.shape), trainable=False)
        delta_x_noised = self.__gradient_x_noised(noised, x)

        for r in range(self.qnt_recurrence):
            to_att_x, hidden = self.rnn(x, delta_x_noised, hidden, training=True)
            x = x + to_att_x
            x = tf.keras.activations.relu(x) # após somar surgem grandes valores
            delta_x_noised = self.__gradient_x_noised(noised, x)
        return x

    def __gradient_x_noised(self, noised, x):
        return tf.square(tf.subtract(noised, x))
