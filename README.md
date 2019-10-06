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


```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
