from Network.RIM import RIM

import tensorflow as tf
import numpy as np
import cv2

import glob
import os

from dival.datasets.standard import get_standard_dataset
from dival import get_reference_reconstructor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ignora GPU [RODA APENAS NA CPU]
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name(): print('GPU found')
else: print("No GPU found")

for a in glob.glob("./aux/*.png"):
    os.remove(a)

batch_size = 1
seed = 1
impl = "astra_cpu"
epocas = 5
qnt_amostras = 4 # amostras por época
numero_recorrencias = 4
mse = tf.keras.losses.MeanSquaredError()
# mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00007)


dataset = get_standard_dataset('lodopab', impl=impl)
keras_generator = dataset.create_keras_generator("train", batch_size=batch_size, shuffle=False)

rec = get_reference_reconstructor("fbp", "lodopab", impl=impl)

model_rim = RIM(numero_recorrencias, (batch_size, 362, 362, 1))

if seed != None:
    tf.random.set_seed(seed)


for epoca in range(epocas):
    list_loss = []
    for i_amostra in range(qnt_amostras):
        x, y = keras_generator.__getitem__(i_amostra) # [(batch_size, 1000, 513), (batch_size, 362, 362)]
        # Reconstrói Sinogramas
        x = np.array([rec.reconstruct(i_x) for i_x in x])
 
        # Normaliza entre 0 e 1
        for i_batch in range(batch_size):
            x[i_batch] -= np.min(x[i_batch]); x[i_batch] /= np.max(x[i_batch])
            x[i_batch] *= np.max(y[i_batch])
        
        x = tf.expand_dims(x, axis=-1) # (batch_size, 1000, 513, 1)
        y = tf.expand_dims(y, axis=-1) # (batch_size, 1000, 513, 1)

        # Salva operações (execução da rede) para derivar e att os pesos
        with tf.GradientTape() as tape:
            prediction = model_rim(x, training=True)
            loss = mse(y, prediction)
        print("min max predicao", np.min(prediction), np.max(prediction))

        for i_batch in range(batch_size):
            print(f"Epoca {epoca},\ti_amostra {i_amostra}, i_batch {i_batch} \tLoss {mse(y[i_batch], prediction[i_batch]).numpy()}")
        print(f"Epoca {epoca},\ti_amostra {i_amostra}\tLoss {loss.numpy()}\n-----------")
        
        # Calcula gradiente
        grad = tape.gradient(loss, model_rim.trainable_weights)
        # Atualiza os pesos
        optimizer.apply_gradients(zip(grad, model_rim.trainable_weights))


        # Salva as imagens
        for i_batch in range(batch_size):
            y_cv2 = np.squeeze(np.array(y[i_batch]*255), axis=-1)
            # y_cv2 = cv2.normalize(np.float64(y[i_batch]), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            # cv2.imwrite('gt.png', i_gt)
            
            x_cv2 = np.squeeze(np.array(x[i_batch]*255), axis=-1)
            # x_cv2 = cv2.normalize(np.float64(x[i_batch]), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            # cv2.imwrite('aux/x_'+str(i)+'.png', x_cv2)

            precic_rec = np.squeeze(np.array(prediction[i_batch]*255), axis=-1)
            # precic_rec = cv2.normalize(np.float64(prediction[i_batch]), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            vis = np.concatenate((y_cv2, x_cv2, precic_rec), axis=1)
            cv2.imwrite("aux/concat_epo"+str(epoca)+"_amostra"+str(i_amostra)+"_batch"+str(i_batch)+".png", vis)


        list_loss.append(loss.numpy())

    print(f"Média loss epoca {epoca}: {np.sum(list_loss)/len(list_loss)}")

# seed = 1: Média loss epoca 4: 0.0008389844442717731
# seed = 1: Média loss epoca 4: 0.0008389844442717731
