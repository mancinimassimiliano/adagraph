# ADAGraph 
This is the official PyTorch code of [AdaGraph: Unifying Predictive and ContinuousDomain Adaptation through Graphs](http://research.mapillary.com/img/publications/CVPR19b.pdf).

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
python3 main.py --network resnet --suffix test_comp_alexnet --dataset compcars
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
for the experiments across decades (i.e. source and target of different decades) .

## Usage 
The layer is initialized as standard BatchNorm except that requires an additional parameter regarding the number of latent domains. 
As an example, considering an input `x` of dimension NxCxHxW where N is the batch-size, C the number of channels and H, W the spatial dimensions.
The `WBN2d` counterpart of a *BatchNorm2d*  would be initialized through:

    wbn = wbn_layers.WBN2d(C, k=D) 

with D representing the desired number of latent domains.
Differently from a standard BatchNorm, this layer will receive an additional input, a vector `w` of shape NxD:

    out = wbn(x, w) 

`w` represents the probability that each sample belongs to each of the latent domains thus it **must sum to 1** in the domain dimensions (i.e. dim=1). Notice that for the `WBN` case `w` has dimension Nx1 and **must sum to 1** in the batch dimension (i.e. dim=0).



## References

If you find this code useful, please cite:

    @inProceedings{mancini2018boosting,
	author = {Mancini, Massimilano and Porzi, Lorenzo and Rota Bul\`o, Samuel and Caputo, Barbara and Ricci, Elisa},
  	title  = {Boosting Domain Adaptation by Discovering Latent Domains},
  	booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  	year      = {2018},
  	month     = {June}
    }


