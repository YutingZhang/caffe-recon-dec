Introduction
============

This software is for the following paper:

* Yuting Zhang, Kibok Lee, Honglak Lee, “Augmenting Supervised Neural Networks with Unsupervised Objectives for Large-scale Image Classification”, *The 33rd International Conference on Machine Learning (ICML)*, 2016. 

Please cite the above paper, if you use this software for your publications.

You can find our paper, slides, and poster at <http://www.ytzhang.net/>

This software provides implementation of stacked convolutional (“what-where”) autoencoders augmented from AlexNet [1] and VGGNet [2]. Network definitions and trained models are included for 

* Decoders for reconstructions.
* Networks for joint reconstruction and classification. 

This software is based on Caffe [3]. When complying with the license of the Caffe toolbox, you can distribute this software freely if its authors (Yuting Zhang, Kibok Lee, and Honglak Lee) are properly acknowledged. 

Installation
============

Please refer to the [official Caffe tutorial](http://caffe.berkeleyvision.org/installation.html) for compiling the code. 

Apart from the requirement of the original Caffe, the support of C++11 standard, e.g., `-std=c++11` for GCC, are needed. 
*Remark*: It is not hard to translate the code to non-C++11 version.

It has been tested with gcc-4.8.3, cuda-7.0, cudnn-4.0 on RedHat 6.x/7.x and Ubuntu 14.04. It should also work on other similar platforms. 

Modifications to Official Caffe
===============================

This code is based on a fork of [the official Caffe `master` branch](https://github.com/BVLC/caffe/tree/master) on May 26, 2016. Please refer to the `original-master` branch for the official Caffe. 

In terms of functionality, this code extends the official Caffe in the following ways. 

* `PoolingLayer` can output the pooling switches, using either global index (standard and faster) or patch-level index (more flexible and slightly slower). 
* It provides a `DepoolingLayer`, which is the unpooling operator using fixed, averaged, or known switches. It can optionally output the “unpooling weights” (i.e., how many elements are unpooled to a single location). We name it *depooling*, because: 
	* We implement its `Forward` as the `Backward` of  `PoolingLayer`, and vice versa. This is analog to `ConvolutionLayer` and `DeconvolutionLayer` 
	* We want to avoid conflicts with some (non-official) implementation of unpooling layers. 
* It provides a `SafeInvLayer`, which performs `f(x)=0` if `x=0`, and `f(x)=1/x` otherwise. It is useful to normalize the unpooled activations by unpooling weights.
* It provides an `ImageOutputLayer` to easily dump activations to image files. 
* The `matcaffe` wrapper support reset a single `caffe.Net` and `caffe.Solver` 

Network Definitions
===================

Network definitions used in our paper are provided in the `recon-dec` folder.

* The base network `[base_network]` can be `alexnet` and `vggnet`.
* The autoencoder model `[model_type]` type can be `SAE-?` and `SWWAE-?`, where `?` can be `layerwise`, `first`, and `all`. 

The first-level subfolders indicate the base network, i.e.  

	recon-dec/[base_network]

The baseline classification network is provided as

	recon-dec/[base_network]/baseline/cls_only_deploy.prototxt

The decoder for reconstructing network activations to images are provided as

	recon-dec/[base_network]/recon/[model_type]/layer[layer_id]_depoly.prototxt
	recon-dec/[base_network]/recon/[model_type]/layer[layer_id]_dump.prototxt

where `[layer_id]` indicates which macro-layer of the encoder produces the input activations to the decoder. The decoders are supposed to be trained without affecting the encoder parameters. `*_deploy.prototxt` is the normal version of the network, and `*_dump.prototxt` is used to dump reconstructed images into disk.  

The networks for joint reconstruction and classification are provided as

	recon-dec/[base_network]/cls/[model_type]/layer[layer_id]_depoly.prototxt

Here, classification pathways are supposed to be finetuned with decoding pathways.

All the network definitions are provided in the `deploy` version. In particular, the data loading module is not provided, since it is up to the users to provide the training and testing data. 

In addition, the naming convention of layers is not the same as the official AlexNet and VGGNet. More specifically, 

* The naming for convolutional and inner-product layers are kept. 
* An auxiliary layer (e.g., `ReLU`, `Pooling`, etc) is named after the preceding convolutional or fully connected layer with a type postfix. For example, `conv1_2/relu` and `conv1_2/pool`.
* A decoding layer is named after the associate encoding layer with a `dec:` prefix. For example, `dec:conv1_2/relu` and `dec:conv1_2/pool`.

Download Trained Models
=======================

The trained models for networks in `recon-dec` can be downloaded by the bash script `recon-dec/fetch_model.sh`. 

* Without any argument, it fetches all available models. 
* The user can also specify a particular model to download by `recon-dec/fetch_model.sh [model_name]`, where name of all available models can be obtained by `recon-dec/list_model.sh` 

As to the baseline classification-only networks: 

* We trained an AlexNet from scratch with all `LRNLayer` removed for a cleaner architecture. 
* We downloaded the 16-layer VGGNet (Model D in [2]) from <http://www.robots.ox.ac.uk/~vgg/research/very_deep/>, and converted/re-saved the `caffemodel` to match the recent version of Caffe. 

Demo in MATLAB
=============

Code in MATLAB for using trained models to reconstruct and classify images are in `recon-dec/matlab`

A quick demo is at `recon-dec/matlab/rdDemo.m`. To run it, just do

	$ cd CAFFE_ROOT/recon-dec/matlab
	$ matlab
	> rdDemo

MATLAB need to be started in desktop mode in order to show the demo figures.

Note that the `matcaffe` wrapper need to be compiled in advance by

	$ make matcaffe

References
==========

[1] Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.

[2] Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[3] Caffe toolbox: <http://caffe.berkeleyvision.org/>

