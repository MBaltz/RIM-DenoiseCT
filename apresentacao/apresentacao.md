## Apresentação - RIM

### 1) Tensorflow

- Até então utilizei apenas Pytorch para DL
    - Preciso aprender Tensorflow para o BeMIDAR

- Não existe implementação da RIM em Tensorflow
    - Mesmo no artigo indicando que foi implementado em Tensorflow

### 2) LoDoPaB-CT + DiVal

#### Utilização do dataset em Tensorflow

- O DiVal foi implementado em Pytorch e feito para utilizar com o Pytorch.

- Alguma boa alma implementou a classe `KerasGenerator()` com os métodos `__geitem__` e `__len__`. Facilitando o uso com o Tensorflow.

    ```python
    class KerasGenerator(Sequence):
        def __init__(self, dataset, part, batch_size, shuffle, reshape=None)
    ```

#### Normalização dos dados

- Ground Truth é normalizado em [0, 1].

    - Todo Ground Truth tem 0.0 (preto absoluto).

    - Há Ground Truth que tem no máximo 1.0 (branco absoluto)

- A reconstrução do sinograma retorna valores em um intervalo estranho
    
    - Min e Máx de 100 reconstruções: [-0.0535979, 1.07661]

    ```python
    x = np.array([rec.reconstruct(i_x) for i_x in x])
    for i_batch in range(batch_size):
        # Normaliza entre 0 e 1
        x[i_batch] -= np.min(x[i_batch]);
        x[i_batch] /= np.max(x[i_batch])
        # Máx da rec é o máx do gt
        x[i_batch] *= np.max(y[i_batch])
    ```

### 3) Decisões de implementação

#### Problema de utilizar o Sinograma como entrada da rede

Para calcular o loss é necessário reconstruir a predição, mas a reconstrução é NÃO diferenciável.

__Solução__: Fazer uso da reconstrução como entrada da rede.

_Vantagem_: Torna o processo bem menos custoso!

### Utilização da GRU do Tensorflow

__`tf.keras.layers.GRU`__

- Dado de entrada unidimensional
    - Seria necessário DIMINUIR A DIMENSÃO do dado de entrada

- Acredito que a implementação do artigo fez uso dessa abordagem

- Similaridade com a Attention dos modelos Transformers


### Utilização da ConvGRU não oficial do Tensorflow

https://github.com/KoertS/ConvGRU2D


#### Problema com a ConvTranspose2D:

- Dimensão de saída
    - Deveria ser

