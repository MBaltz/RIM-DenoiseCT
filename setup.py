from Network.RIM import RIM
from Dataset.LoDoPaB-CT_Reconstructor"import TFDataset

import tensorflow as tf
import numpy as np
from cv2 import imwrite
from os import remove, mkdir
from os.path import join, exists
from time import time


################################################################################
id_exec = -1 # -1 se for para uma nova execução. Outro número para retomar.
dir_execucoes = "/tmp/execucoes/"
treinar = True
testar = True
epoca_inicial = 0 # Se id_exec != -1, alterar época que será carregada
################################################################################
epocas = 5
numero_recorrencias = 4
loss_fn = tf.keras.losses.MeanSquaredError()
# loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
learning_rate = 0.00007
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
seed = 1
################################################################################
dir_dataset_npy = "/home/baltz/dados/Dados_2/tcc-database/reco_dataset"
batch_size = 1
shuffle = True
################################################################################


# Se não for pra treinar ou testar, não tem o que fazer
if True not in [treinar, testar]:
    exit(f"\nNada p/ fazer: True not in [treinar, testar]")

retomar_exec = False
if id_exec == -1: id_exec = int(time())
else: retomar_exec = True

# Diretórios importantes
dir_tfboard = join(dir_execucoes, f"{id_exec}/tfboard/")
dir_pesos = join(dir_execucoes, f"{id_exec}/pesos/")
dir_imgs = join(dir_execucoes, f"{id_exec}/imgs/")

# Verifica e constrói os diretorios necessários
dir_verificar = [dir_execucoes, dir_tfboard, dir_pesos, dir_imgs]
if not retomar_exec:
    for d in dir_verificar: mkdir(join(dir_execucoes, d))
for d in dir_verificar:
    if not exists(d): exit(f"Diretorio inexistente: {d}")

# Verifica se usará GPU
if not tf.test.gpu_device_name(): print('No ', end=""); print('GPU found\n\n')

# Utiliza a seed (caso tenha) antes de instanciar qualquer coisa randomica
if seed != None: tf.random.set_seed(seed)

# Carrega todos os Datasets
train_dataset = TFDataset(dir_dataset_npy, "train", batch_size,
    shuffle=shuffle, return_with_channel=True)
test_dataset = TFDataset(dir_dataset_npy, "test", batch_size,
    shuffle=shuffle, return_with_channel=True)
vali_dataset = TFDataset(dir_dataset_npy, "validation", batch_size,
    shuffle=shuffle, return_with_channel=True)


model = RIM(numero_recorrencias, (batch_size, 362, 362, 1))


def treinar(epoca):
    list_loss = []
    count_batch = 0
    for x, y in train_dataset: # x/y.shape: (batch_size, 1000, 513, 1)
        # Salva operações (execução da rede) para calcular o gradiente
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            loss = loss_fn(y, prediction)

        print(f"Epoca {epoca}, \tcount_batch {count_batch}, \tLoss {loss.numpy()}")
        list_loss.append(loss.numpy())

        # Calcula gradiente
        grad = tape.gradient(loss, model.trainable_weights)
        # Atualiza os pesos
        optimizer.apply_gradients(zip(grad, model.trainable_weights))

        # Salva as imagens
        for i_amostra in range(len(x)):
            y_cv2 = np.squeeze(np.array(y[i_amostra]*255), axis=-1)
            x_cv2 = np.squeeze(np.array(x[i_amostra]*255), axis=-1)
            precic_cv2 = np.squeeze(np.array(prediction[i_amostra]*255), axis=-1)

            vis = np.concatenate((y_cv2, x_cv2, precic_cv2), axis=1)
            nome_img = f"concat_epo"{epoca}"_batch"{count_batch}"_amostra"{i_amostra}".png"
            imwrite(join(dir_imgs, nome_img), vis)

    print(f"Média do loss da epoca {epoca}: {np.sum(list_loss)/len(list_loss)}")


def testar(epoca):
    raise NotImplementedError

# model.save_weights(dir_pesos, overwrite=True)
# exit()

for epoca in range(epocas):
    if treinar:
        treinar(epoca)
    if testar:
        testar(epoca)
    

# seed = 1: Média loss epoca 4: 0.0008389844442717731
# seed = 1: Média loss epoca 4: 0.0008389844442717731
