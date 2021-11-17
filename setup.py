from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from Network.RIM import RIM
from Dataset.LoDoPaBCT_Reconstructor.TFDataset import TFDataset

import tensorflow as tf
import numpy as np
from cv2 import imwrite
from os import remove, mkdir
from os.path import join, exists
from time import time

import wandb


################################################################################
id_exec = -1 # -1 se for para uma nova execução. Outro número para retomar.
dir_execucao = "/tmp/execucoes/"
treinar = True
testar = False
epoca_inicial = 0 # Se id_exec != -1, alterar para a epoca carregada + 1, else=0
################################################################################
epocas = 4
numero_recorrencias = 4
loss_fn = tf.keras.losses.MeanSquaredError()
# loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
learning_rate = 0.00005
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
seed = 1
################################################################################
dir_dataset_npy = "/home/baltz/Documentos/tcc-database/reco_dataset"
batch_size = 1
shuffle = False
################################################################################


# Se não for pra treinar ou testar, não tem o que fazer
if True not in [treinar, testar]:
    exit(f"\nNada p/ fazer: True not in [treinar, testar]")

retomar_exec = False
if id_exec == -1: id_exec = int(time())
else: retomar_exec = True; print(f"\n\n>> RETOMANDO!\t\t!RETOMANDO!\n")
print(f"\n\n>> EXECUTANDO REDE DE ID: [{id_exec}]\n")

# Diretórios importantes
dir_execucao = join(dir_execucao, f"{id_exec}/")
dir_wandb = join(dir_execucao, "wandb/")
dir_pesos = join(dir_execucao, "pesos/")
dir_imgs = join(dir_execucao, "imgs/")

# Verifica e constrói os diretorios necessários
dir_verificar = [dir_execucao, dir_wandb, dir_pesos, dir_imgs]
if not retomar_exec:
    for d in dir_verificar: print(d); mkdir(d)
for d in dir_verificar:
    if not exists(d): exit(f"Diretorio inexistente: {d}")

# Verifica se usará GPU
if not tf.test.gpu_device_name(): print('No ', end=""); print('GPU found\n')

# Utiliza a seed (caso tenha) antes de instanciar qualquer coisa randomica
if seed != None: tf.random.set_seed(seed)


# Carrega todos os Datasets
train_dataset = TFDataset(dir_dataset_npy, "train", batch_size,
    shuffle=shuffle, return_with_channel=True)
test_dataset = TFDataset(dir_dataset_npy, "test", batch_size,
    shuffle=shuffle, return_with_channel=True)
vali_dataset = TFDataset(dir_dataset_npy, "validation", batch_size,
    shuffle=shuffle, return_with_channel=True)

wandb.init(project=f"RIM_LowDoseCT", entity="mbaltz", id=str(id_exec),
    dir=dir_wandb)
wandb.run.name = str(id_exec)

model = RIM(numero_recorrencias)

if retomar_exec:
    # Inicia os pesos da rede executando uma amostra aleatória
    model(np.zeros((1, 362, 362, 1)))
    dir_peso_carregar = join(dir_pesos, f"epoca_{epoca_inicial-1}.pesos") 
    print(f"\n\n>> Carregando pesos de: {dir_peso_carregar}\n")
    model.load_weights(dir_peso_carregar)


def fn_treinar(epoca):
    print(f"\n\n>> TREINANDO ÉPOCA {epoca}\n")
    losses_fim_epoca = []
    losses_durante_epoca = []
    count_batch = 0
    for x, y in train_dataset: # x/y.shape: (batch_size, 1000, 513, 1)
        # Salva operações (execução da rede) para calcular o gradiente
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            loss = loss_fn(y, prediction)

        print(f"[{id_exec}] Epoca: {epoca}, \tLoss {loss.numpy():.8f},", end="")
        print(f" \tProgresso: {count_batch}/{train_dataset.__len__()}")

        losses_fim_epoca.append(loss.numpy())
        losses_durante_epoca.append(loss.numpy())

        # Calcula gradiente
        grad = tape.gradient(loss, model.trainable_weights)
        # Atualiza os pesos
        optimizer.apply_gradients(zip(grad, model.trainable_weights))

        # Salva dados no wandb
        if count_batch%10 == 0:
            media = np.sum(losses_durante_epoca)/len(losses_durante_epoca)
            wandb.log({"loss/durante_epoca/train": media},
            step=(epoca*train_dataset.__len__())+count_batch)
            losses_durante_epoca = []


        # Salva as imagens
        if count_batch%50 == 0:
            for i_amostra in range(len(x)):
                y_cv2 = np.squeeze(np.array(y[i_amostra]*255), axis=-1)
                x_cv2 = np.squeeze(np.array(x[i_amostra]*255), axis=-1)
                precic_cv2 = np.squeeze(np.array(prediction[i_amostra]*255), axis=-1)
                vis = np.concatenate((y_cv2, x_cv2, precic_cv2), axis=1)
                nome_img = f"concat_epo{epoca}_batch{count_batch}_amostra{i_amostra}.png"
                imwrite(join(dir_imgs, nome_img), vis)

        count_batch += 1


    model.save_weights(join(dir_pesos, f"epoca_{epoca}.pesos"), overwrite=True)

    media_loss_fim_epoca = np.sum(losses_fim_epoca)/len(losses_fim_epoca)
    print(f"Média do loss da epoca {epoca}: {media_loss_fim_epoca}")

    wandb.log({"loss/fim_epoca/train": media_loss_fim_epoca}, step=epoca)


def fn_testar(epoca):
    raise NotImplementedError


for epoca in range(epoca_inicial, epocas):
    if treinar:
        fn_treinar(epoca)
    if testar:
        fn_testar(epoca)

print(f"\n>> Execução [{id_exec}] concluída com sucesso!")

# seed = 1: Média loss epoca 4: 0.0008389844442717731
# seed = 1: Média loss epoca 4: 0.0008389844442717731
