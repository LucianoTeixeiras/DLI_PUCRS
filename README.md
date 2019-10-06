## Classificação de imagens de pintores/artistas famosos

* artigo de conclusão de cadeira (Deep Learning I) da pós-gradução de Ciência de Dados, da PUC/RS
<br> Professor: JONATAS WERHMANN

Integrantes do grupo: Daniel Paiva, Carlos Átila e Vinícius Appel


### Execução 01
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

```python
batch = 59
np.random.seed(1)```

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

```

```python

```
