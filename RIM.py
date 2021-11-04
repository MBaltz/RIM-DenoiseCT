import tensorflow as tf


class RNN(tf.keras.Model):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size

    def call(self, x_and_grad, hidden):

        # TODO: Layers

        # valor que será comado à x e gerando x_t+1 e o novo hidden state
        return to_att_x, new_hidden

class RIM(tf.keras.Model):

    def __init__(self, qnt_recurrence, in_size, hidden_size):
        super().__init__()
        self.qnt_recurrence = qnt_recurrence
        self.in_size = in_size
        self.hidden_size = hidden_size
        # Dimensão de entrada é (2, x, y) por conta do gradiente/delta
        self.rnn = RNN((2, self.in_size[0], self.in_size[1]), self.hidden_size)


    def call(self, x):

        noised = tf.Variable(x) # Copia x para salvar a imagem ruidosa primária
        hidden = tf.Variable(tf.zeros(x.shape)) # Primeiro hidden state [0 .. 0]
        delta_x_noised = self.gradient_x_noised(noised, x)

        for r in range(qnt_recurrence):
            to_att_x, hidden = self.rnn(tf.stack([x, delta_x_noised]), hidden)
            x = x + to_att_x
            x = tf.keras.activations.relu(x) # após somar surgem grandes valores
            delta_x_noised = self.gradient_x_noised(noised, x)
        return x


    def gradient_x_noised(self, noised, x):
        return tf.subtract(noised, x)
