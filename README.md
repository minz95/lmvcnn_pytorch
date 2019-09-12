# lmvcnn_pytorch
 Pytorch Implementation of Learning Local Shape Descriptors from Part Correspondences(ToG 2017, H Huang et al.): https://people.cs.umass.edu/~hbhuang/local_mvcnn/

## python package
* numpy
* pillow
* pytorch cuda=9.0

## directory architecture
```sh
layers.py
pair_datasets.py
pair_fcn.py
pair_loader.py
pair_trainer.py
extract_feature.py
train_corr_list.npy

# dataset
image_data

#store the result models
model
```

## Training
```sh
python pair_trainer.py
```

## Test
extract features from 3D models and compare the similarity

```sh
python extract_feature.py
python ranking.py
```

## Dataset
48 multiviews from the 3D model
use BlenderPhong to generate multiviews

## Results

