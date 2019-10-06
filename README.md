## Classificação de imagens de pintores/artistas famosos

* Trabalho de conclusão de cadeira (Deep Learning I) da pós-gradução de Ciência de Dados, da PUC/RS
<br> Professor: JONATAS WERHMANN

Integrantes do grupo: Daniel Paiva, Carlos Átila e Vinícius Appel


### Execução 01

Execução realizada com a Resnet34, em **4 épocas** <br>
Total de imagens: 2629

**Modelo Resnet34**

![Resnet34](https://www.researchgate.net/profile/Aaron_Vose/publication/330400293/figure/fig6/AS:715395283558403@1547574935970/ResNet-neural-network-architecture-ResNet-34-pictured-image-from-11.ppm)

```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.vision import *
from fastai.metrics import error_rate

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np```

```

```python
batch = 59
np.random.seed(1)
```

```python
pasta = "C:\\Users\\appel\\Desktop\\Pos_Data_Science\\9_Deep_Learning_I\\Projeto\\Artistas\\classes"
```

```python
nomes_artistas = get_image_files(pasta)
```

```python
pat = r'/([^/]+)_\d+.jpg$'
```

```python
dataset = ImageDataBunch.from_name_re(pasta, nomes_artistas, pat,
                                      ds_tfms=get_transforms(),
                                      size=224, bs=batch
                                      ).normalize(imagenet_stats)

print('Tamanho da base de treinamento: {}'.format(len(dataset.train_ds)))
print('Tamanho da base de validação: {}'.format(len(dataset.valid_ds)))
print('Classes: {}'.format(dataset.classes))
print('Total de classes: {}'.format(len(dataset.classes)))
```
Tamanho da base de treinamento: 2104 <br>
Tamanho da base de validação: 525 <br>
Classes: ['Albrecht_Dürer', 'Alfred_Sisley', 'Edgar_Degas', 'Francisco_Goya', 'Marc_Chagall', 'Pablo_Picasso', 'Paul_Gauguin', 'Pierre-Auguste_Renoir', 'Rembrandt', 'Titian', 'Vincent_van_Gogh'] <br>
Total de classes: 11 <br>

```python
dataset.show_batch(rows=3, figsize=(7,6))
```
![Amostra aleatória de imagens](https://github.com/appelvini/DLI_PUCRS/blob/master/Capturar.JPG)

```python
modelo = cnn_learner(dataset, models.resnet34, metrics=accuracy)
```

```python
modelo.fit_one_cycle(4)
```
| epoch | train_loss | valid_loss | accuracy | time  |
|-------|------------|------------|----------|-------|
| 0     | 2.341908   | 0.934602   | 0.704762 | 01:53 |
| 1     | 1.527976   | 0.819618   | 0.725714 | 02:06 |
| 2     | 1.109361   | 0.649435   | 0.794286 | 01:54 |
| 3     | 0.874055   | 0.628907   | 0.773333 | 01:39 |
```python
modelo.save('4Epocas')
```

```python
interpretacao = ClassificationInterpretation.from_learner(modelo)
```

```python
interpretacao.plot_top_losses(9, figsize=(15,11), heatmap=False)
```
![Mostra os 9 principais erros](https://github.com/appelvini/DLI_PUCRS/blob/master/Anota%C3%A7%C3%A3o%202019-10-06%20180409.jpg)
```python
interpretacao.plot_confusion_matrix(figsize=(12,12), dpi=60)
```
![Matriz de confusão](https://github.com/appelvini/DLI_PUCRS/blob/master/Anota%C3%A7%C3%A3o%202019-10-06%20180746.jpg)

### Com 4 épocas tivemos uma acurácia de 77,33%
Podemos verificar que houveram alguns erros na validação, com destaque para imagens que eram de Rembrandt, mas foram classificados erroneamente como Titian (7 casos);

| Correto                 | Classificado       | Qtd |
|-------------------------|--------------------|-----|
| 'Rembrandt'             | 'Titian'           | 7   |
| 'Francisco_Goya'        | 'Titian'           | 6   |
| 'Paul_Gauguin'          | 'Pablo_Picasso'    | 6   |
| 'Titian'                | 'Rembrandt'        | 6   |
| 'Pierre-Auguste_Renoir' | 'Edgar_Degas'      | 5   |
| 'Rembrandt'             | 'Francisco_Goya'   | 5   |
| 'Pablo_Picasso'         | 'Vincent_van_Gogh' | 4   |
| 'Vincent_van_Gogh'      | 'Alfred_Sisley'    | 4   |
| 'Vincent_van_Gogh'      | 'Pablo_Picasso'    | 4   |


### Execução 02

Execução realizada com a Resnet34, em **10 épocas** <br>
Total de imagens: 2629

```python
modelo.fit_one_cycle(10)
```

| epoch | train_loss | valid_loss | accuracy | time  |
|-------|------------|------------|----------|-------|
| 0     | 0.681377   | 0.758513   | 0.771429 | 01:39 |
| 1     | 0.637672   | 0.735984   | 0.790476 | 01:41 |
| 2     | 0.609460   | 0.762438   | 0.779048 | 01:35 |
| 3     | 0.592992   | 0.756005   | 0.761905 | 01:35 |
| 4     | 0.546762   | 0.725455   | 0.794286 | 01:33 |
| 5     | 0.494583   | 0.688622   | 0.788571 | 01:33 |
| 6     | 0.457078   | 0.654186   | 0.794286 | 01:34 |
| 7     | 0.428280   | 0.624490   | 0.805714 | 01:36 |
| 8     | 0.375367   | 0.622810   | 0.819048 | 01:34 |
| 9     | 0.369685   | 0.615050   | 0.815238 | 01:33 |
```python
interpretacao = ClassificationInterpretation.from_learner(modelo)

interpretacao.plot_top_losses(9, figsize=(15,11), heatmap=False)
```
![Amostra aleatória de imagens](https://github.com/appelvini/DLI_PUCRS/blob/master/Anota%C3%A7%C3%A3o%202019-10-06%20184917.jpg)
```python
interpretacao.plot_confusion_matrix(figsize=(12,12), dpi=60)
```
![Matriz de confusão](https://github.com/appelvini/DLI_PUCRS/blob/master/Anota%C3%A7%C3%A3o%202019-10-06%20185101.jpg)
```python
### Com 10 épocas tivemos uma acurácia de **81,52%**
Podemos ver que o número total de erros diminuiu, porém o destaque continua o mesmo;

| Correto                 | Classificado    | Qtd |
|-------------------------|-----------------|-----|
| 'Rembrandt',            | 'Titian'        | 7   |
| 'Paul_Gauguin'          | 'Alfred_Sisley' | 5   |
| 'Pierre-Auguste_Renoir' | 'Edgar_Degas'   | 5   |
| 'Pierre-Auguste_Renoir' | 'Alfred_Sisley' | 4   |
| 'Titian'                | 'Rembrandt'     | 4   |
| 'Francisco_Goya'        | 'Rembrandt'     | 3   |
| 'Pablo_Picasso'         | 'Edgar_Degas'   | 3   |
| 'Pablo_Picasso'         | 'Paul_Gauguin'  | 3   |
| 'Paul_Gauguin'          | 'Edgar_Degas'   | 3   |
```

### Execução 03

Execução realizada com a *squeezenet1_0*, em **8 épocas** <br>
Total de imagens: 2629

**Modelo squeezenet1_0** <br>
![Modelo squeezenet1_0](https://pytorch.org/assets/images/squeezenet.png)

```python
modelo = cnn_learner(dataset, models.squeezenet1_0, metrics=accuracy)

modelo.fit_one_cycle(8)
```

| epoch | train_loss | valid_loss | accuracy | time  |
|-------|------------|------------|----------|-------|
| 0     | 2.447169   | 1.314411   | 0.600000 | 01:31 |
| 1     | 1.554746   | 0.925522   | 0.729524 | 01:30 |
| 2     | 1.178366   | 0.843570   | 0.760000 | 01:31 |
| 3     | 0.938747   | 0.796989   | 0.752381 | 01:32 |
| 4     | 0.750881   | 0.733911   | 0.761905 | 01:35 |
| 5     | 0.610888   | 0.653338   | 0.784762 | 01:31 |
| 6     | 0.532116   | 0.645012   | 0.786667 | 01:30 |
| 7     | 0.452528   | 0.641234   | 0.788571 | 01:31 |
```python
interpretacao.plot_confusion_matrix(figsize=(12,12), dpi=60)
```

![Matriz de confusão](https://github.com/appelvini/DLI_PUCRS/blob/master/Anota%C3%A7%C3%A3o%202019-10-06%20193800.jpg)

Com a arquitetura Squeezenet tivemos uma acurácia de **78,85%**


```python

```

```python

```

```python

```
