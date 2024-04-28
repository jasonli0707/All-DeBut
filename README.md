# All-Deformable-Butterfly Network
This repository serves as the official code release for the IEEE TNNLS paper titled: [Lite It Fly: An All-Deformable-Butterfly Network](https://ieeexplore.ieee.org/document/10329580).

***[Rui Lin*](https://rlin27.github.io/)***, ***[Jason Chun Lok Li*](https://www.linkedin.com/in/jason-chun-lok-li-0590b3166/)***, Jiajun Zhou, Binxiao Huang, Jie Ran, Ngai Wong

(*Equal contribution)

## Running

This repository has been tested with Ubuntu 20.04.1 LTS, Python 3.8, Pytorch 1.10.1 and CUDA 11.3.

### 1. Automated Chain Generation & MNIST:

All codes regarding automated chain generation and MNIST experiment shown in the paper is under the `AutoChain/`. To run the experiment, first download the MNIST and save it to the `data/mnist_data`, and then run:

```
python ./AutoChain/main.py
```

### 2. PointNet & ModelNet40:

First download the **ModelNet40** dataset [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save it to `data/modelnet40_normal_resampled/`.

We prepare some simple bash scripts, to train a teacher model simply run: 

```
./scripts/run_pointnet_vanilla.sh
```

To train a student model using CRD framework, run:
```
./scripts/run_pointnet_distill.sh
```

To evaluate a saved checkpoint:
```
python test_pointnet.py \
--model_path [path to the saved checkpoint] \
--r_shape_txt [path to .txt files specifing the structure of debut chains] \

```

Some useful flags to know:
```
--path_t: The path to the teacher's checkpoint
--distill: Select the distillation method to use
--model_s: Select the type of student model (i.e SVD, Butterfly, Fastfood, DeBut)
-a: Balancing weight for KD Loss
-b: Balancing weight for CRD Loss
--r_shape_txt: The path to .txt files describing the shapes of the factors in the given monotonic or bulging DeBut chains 
```



### 3. CIFAR-100:
First download the CIFAR-100 dataset and save it to `data/cifar-100-python`. 

Similarly, to train a teacher model simply run: 

```
./scripts/run_[vgg/resnet]_vanilla_cifar100.sh
```

To train a student model using CRD framework, run:
```
./scripts/run_[vgg/resnet]_distill_cifar100.sh
```

To evaluate a saved checkpoint:
```
python test.py \
--model_path [path to the saved checkpoint] \
--r_shape_txt [path to .txt files specifing the structure of debut chains] \
```

## Citation
If you find All-DeBut useful for your research and applications, please consider citing it using this BibTeX:
```
@article{lin2023lite,
  title={Lite It Fly: An All-Deformable-Butterfly Network},
  author={Lin, Rui and Li, Jason Chun Lok and Zhou, Jiajun and Huang, Binxiao and Ran, Jie and Wong, Ngai},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```


## Acknowledgements
Our codes are adapted from official released codes for [CRD](https://github.com/HobbitLong/RepDistiller) by Yonglong Tian et al. and Pytorch implementation of [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) by Xu Yan. 