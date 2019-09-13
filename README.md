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
image_data/

# store the result models
model/

# store the sample points
samples/
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
Original 3D models for training can be downloaded at [here](https://people.cs.umass.edu/~hbhuang/local_mvcnn/). 3D models and point correspondence can be found.

In the paper, possible sample points are calculated based on the raycast

We used 48 multiviews from the 3D model. 

Scales are: [0.25, 0.5, 0.75]
, 12 rotation angles from sample points
, Normals are: [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

Use [BlenderPhong](https://github.com/WeiTang114/BlenderPhong) to generate multiviews, or C++ source files are given in the authors' [project page](https://people.cs.umass.edu/~hbhuang/local_mvcnn/). Theses sources are not covered in this repository.

## Results

