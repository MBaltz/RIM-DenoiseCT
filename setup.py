from RIM import RIM

import tensorflow as tf
import numpy as np
import cv2

from dival.datasets.standard import get_standard_dataset
from dival import get_reference_reconstructor

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ignora GPU [RODA APENAS NA CPU]
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name(): print('GPU found')
else: print("No GPU found")


batch_size = 2
impl = "astra_cpu"
epocas = 20
qnt_amostras = 1
numero_recorrencias = 3
mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

dataset = get_standard_dataset('lodopab', impl=impl)
# print(f"\n\nChecando LoDoPaB-CT: {dataset.check_for_lodopab()}\n")
keras_generator = dataset.create_keras_generator("train", batch_size=batch_size, shuffle=False)

rec = get_reference_reconstructor("fbp", "lodopab", impl=impl)

model_rim = RIM(numero_recorrencias, (batch_size, 362, 362))


for epoca in range(epocas):
    for i_batch in range(qnt_amostras):
        x, y = keras_generator.__getitem__(i_batch) # [(batch_size, 1000, 513), (batch_size, 362, 362)]
        # Reconstrói Sinogramas
        x = np.array([rec.reconstruct(i_x) for i_x in x])
 
        # Normaliza entre 0 e 1
        x[:,0,0] = 1.0; x[:,0,1] = 0.0
        x -= np.min(x); x /= np.max(x)
        x[:,0,0] = 0.0
        y[:,0,0] = 1.0; y[:,0,1] = 0.0
        y -= np.min(y); y /= np.max(y)
        y[:,0,0] = 0.0
        
        x = tf.expand_dims(x, axis=-1) # (batch_size, 1000, 513, 1)
        y = tf.expand_dims(y, axis=-1) # (batch_size, 1000, 513, 1)

        # Salva operações (execução da rede) para derivar e att os pesos
        with tf.GradientTape() as tape:
            prediction = model_rim(x, training=True)
            loss = mse(y, prediction)
        print("min max sigmoid", np.min(prediction), np.max(prediction))

        for i_img in range(batch_size):
            print(f"Epoca {epoca},\ti_batch {i_batch}, Amostra {i_img} \tLoss {mse(y[i_img], prediction[i_img]).numpy()}")
        print(f"Epoca {epoca},\ti_batch {i_batch}\tLoss {loss.numpy()}\n-----------")
        
        # Calcula gradiente
        grad = tape.gradient(loss, model_rim.trainable_weights)
        # Atualiza os pesos
        optimizer.apply_gradients(zip(grad, model_rim.trainable_weights))

        # Salva as imagens
        for i_img in range(batch_size):
            y_cv2 = np.squeeze(np.array(y[i_img]*255), axis=-1)
            # y_cv2 = cv2.normalize(np.float64(y[i_img]), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            # cv2.imwrite('gt.png', i_gt)
            
            x_cv2 = np.squeeze(np.array(x[i_img]*255), axis=-1)
            # x_cv2 = cv2.normalize(np.float64(x[i_img]), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            # cv2.imwrite('aux/x_'+str(i)+'.png', x_cv2)

            precic_rec = np.squeeze(np.array(prediction[i_img]*255), axis=-1)
            # precic_rec = cv2.normalize(np.float64(prediction[i_img]), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            vis = np.concatenate((y_cv2, x_cv2, precic_rec), axis=1)
            cv2.imwrite("aux/concat_epo"+str(epoca)+"_amostra"+str(i_img)+".png", vis)
