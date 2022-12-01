Step-by-step para rodar esse repositório com o DOTA e ProbIoU:

1. Baixar a docker image localmente:
```
docker pull yangxue2docker/py3-tf1.15.2-nv-torch1.8.0-cuda11
```

2. Inicializar a imagem:
```
docker run -it --ipc=host -p <port para jupyterlab>:<port para jupyterlab> -v <path_to_workdir>:/workdir -v <path_to_datasets>:/datasets --net host --gpus all -d yangxue2docker/py3-tf1.15.2-nv-torch1.8.0-cuda11:v1.0
```

3. Dentro da imagem iniciar um host para um juypter-lab (opcional):
```
jupyter-lab --ip 0.0.0.0 --allow-root --no-browser --port <port para jupyterlab>
```

3.1. Copiar o token do jupyter-lab e colar no browser como (exemplo):
```
http://localhost:8889/?token=98d9c589733adab715c6d9c63fff035315fd2144429e9c12
```

** Para as próximas etapas, é possível usar como exemplo o Notebbok _run.ipynb **

4. Fazer as compilação necessárias:
```
%cd /workdir/RotationDetection/libs/utils/cython_utils
!rm *.so
!rm *.c
!rm *.cpp
!python setup.py build_ext --inplace

%cd /workdir/RotationDetection/libs/utils/
!rm *.so
!rm *.c
!rm *.cpp
!python setup.py build_ext --inplace
```

5. Copiar o config file correto. Os arquivos de configuração se encontram aqui: https://github.com/LucasKirsten/RotationDetection/tree/main/libs/configs No seu caso, copie 
todo o conteúdo do código em https://github.com/LucasKirsten/RotationDetection/blob/main/libs/configs/DOTA/r3det_gwd/cfgs_res50_dota_r3det_gwd_v6.py para aqui: 
https://github.com/LucasKirsten/RotationDetection/blob/main/libs/configs/cfgs.py

5.1. O parâmetro *REG_LOSS_MODE* define a loss de regressão que tu quer usar. Para usar a ProbIoU, a princípio é só setar esse parâmetro para 3, e para usar a GWD setar para 2
Eu defini esse comportamento aqui (para o caso da RetinaNet): https://github.com/LucasKirsten/RotationDetection/blob/main/libs/models/detectors/gwd/build_whole_network.py e a
implementação da ProbIoU está aqui (linha 106): https://github.com/LucasKirsten/RotationDetection/blob/main/libs/models/losses/losses_gwd.py 
Caso a rede for treinada com a ProbIoU, eu adicionei um print para nos alertar que está tudo funcionando corretamente :D Contudo, talvez seja necessário alguma alteração para
rodar com outros detectores!

6. Para o dataset do DOTA, primeiro é necessário fazer os crops da imagem. Para isso, é necessário alterar o path para as imagens no código (ao final) aqui: https://github.com/LucasKirsten/RotationDetection/blob/main/dataloader/dataset/DOTA/data_crop.py

6.1. Depois, é necessário converter os dados para TFrecords (exemplo abaixo):
```
%cd /workdir/RotationDetection/dataloader/dataset/  
!python convert_data_to_tfrecord.py --VOC_dir='/datasets/dataset/DOTA/crop/train' \
                                   --xml_dir='labeltxt' \
                                   --image_dir='images' \
                                   --save_name='train'  \
                                   --img_format='.png'  \
                                   --dataset='DOTA'
```

7. Depois disso é só treinar :) Você pode até mesmo visualizar o treino usando o Tensorboard
```
%cd /workdir/RotationDetection/tools/gwd
!python train.py
```

Quanto a organização dos meus dados:
- Os dados do DOTA estão separados em pastas: train, val e test
- Dentro de cada uma dessas pastas eu criar as pastas: images e labels que eu organizei manualmente do que eu baixei do Drive






