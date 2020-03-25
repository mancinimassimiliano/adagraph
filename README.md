# Predictive Domain Adaptation with AdaGraph 
This is the official PyTorch code of [AdaGraph: Unifying Predictive and ContinuousDomain Adaptation through Graphs](http://research.mapillary.com/img/publications/CVPR19b.pdf).
![alt text](https://raw.githubusercontent.com/mancinimassimiliano/adagraph/master/img/teaser.png)

This version has been tested on:
* PyTorch 1.0.1
* python 3.5
* cuda 9.2

## Installation
To install all the dependencies, please run:
```
pip install -r requirements.txt
```

## Datasets
For setting up the datasets, please download [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html) and [A Century of Portraits](http://people.eecs.berkeley.edu/~shiry/projects/yearbooks/yearbooks.html), unpack them and move them into the ```data``` folder. This folder contains also the a file ```data/car-images.txt``` which contains the list of images and metadata+class annotations for CompCars.

## Pretrained architectures
As pretrained models we use the default PyTorch ResNet and the the converted Caffe AlexNet for experiments with decaf. You can find the network at the following [link](https://drive.google.com/file/d/1QoVr4qqbc6RPG0XX-H3SwSAabDxr-Is6/view?usp=sharing). Please download the network and either place it on ```pretrained``` or modify the relative path in ```models/networks.py```.


## Predictive DA experiments
Our code allows to perform predictive DA experiments, producing the results for the following methods:

* _Baseline_ (the source only baseline)
* _AdaGraph_ (our method)
* _Baseline/AdaGraph + Refinement_ (either the _Baseline_ or _AdaGraph_ + our continuous DA strategy)
* _DA Upper Bound_ (a BN based DA approach with target available before-hand)

For reproducing the results on CompCars, please run:
```
python3 main.py --network decaf --suffix test_comp_alexnet --dataset compcars
```
for the experiments with the decaf features and:
```
python3 main.py --network resnet --suffix test_comp_resnet --dataset compcars
```
for the experiments with ResNet-18.

For Portraits, please run:
```
python3 main.py --network resnet --suffix portraits_test --dataset portraits --skip regions
```
for the experiments across regions (i.e. source and target of different regions) and:
```
python3 main.py --network resnet --suffix portraits_test --dataset portraits --skip decades
```
for the experiments across decades (i.e. source and target of different decades).

The full results of all the models will be stored in the ```./results``` folder, as ```.npy```.


## References

If you find this code useful, please cite:

    @inProceedings{mancini2019adagraph,
	author = {Mancini, Massimiliano and Rota Bul\`o, Samuel and Caputo, Barbara and Ricci, Elisa},
  	title  = {AdaGraph: Unifying Predictive and ContinuousDomain Adaptation through Graphs},
  	booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  	year      = {2019},
  	month     = {June}
    }


