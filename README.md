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
=========

Please refer to the [official Caffe tutorial](http://caffe.berkeleyvision.org/installation.html) for compiling the code. 

Apart from the requirement of the original Caffe, the support of C++11 standard, e.g., `-std=c++11` for GCC, are needed. 
Remark: It is not hard to translate the code to non-C++11 version.

It has been tested with gcc-4.8.3, cuda-7.0, cudnn-4.0 on RedHat 6.x/7.x and Ubuntu 14.04. It should also work on other similar platforms. 

Modifications to Official Caffe
=====================

This code is based on a fork of [the official Caffe `master`](https://github.com/BVLC/caffe/tree/master) branch on May 26, 2016. Please refer to the `original-master` branch for the official Caffe. 

In terms of functionality, this code extends the official Caffe in the following ways. 

* `PoolingLayer` can output the pooling switches, using either global index (standard and faster) or patch-level index (more flexible and slightly slower). 
* It provides a `DepoolingLayer`, which is the unpooling operator using fixed, averaged, or known switches. It can optionally output the “unpooling weights” (i.e., how many elements are unpooled to a single location). We name it *depooling*, because: 
	* We implement its `Forward` as the `Backward` of  `PoolingLayer`, and vice versa. This is analog to `ConvolutionLayer` and `DeconvolutionLayer` 
	* We want to avoid conflicts with some (non-official) implementation of unpooling layers. 
* It provides a `SafeInvLayer`, which performs `f(x)=0` if `x=0`, and `f(x)=1/x` otherwise. It is useful to normalize the unpooled activations by unpooling weights.
* It provides an `ImageOutputLayer` to easily dump activations to image files. 

Network Definition
==============


Directory architecture
Layer naming convention
LossWeight
DataLoader

Download Trained Models
====================



References
=========

[1] Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.

[2] Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[3] Caffe toolbox: <http://caffe.berkeleyvision.org/>

