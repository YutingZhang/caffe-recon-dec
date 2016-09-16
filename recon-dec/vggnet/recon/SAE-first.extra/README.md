Extra VGG Models
================

In this folder, we provide SWWAE-first image reconstruction models for fc6,7,8 features from the 16-layer VGGNet.
We did not report results in our paper using these model, since the image classification session uses only the decoder up to pool5.

Compared to ../SWWAE-first, the caffe prototxt files here might be less cleaned up, but they should be good enough to use. 
The pretrained caffe models associated with these prototxt files were also from different trials, and they were also less tuned. Still, they should be good enough for reconstructing the features to images. 

Please cite our paper if these models are useful for your research. 

* Yuting Zhang, Kibok Lee, Honglak Lee, “Augmenting Supervised Neural Networks with Unsupervised Objectives for Large-scale Image Classification”, *The 33rd International Conference on Machine Learning (ICML)*, 2016. 


