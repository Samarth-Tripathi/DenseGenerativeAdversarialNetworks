# Dense-Generators
Generative models with DenseNet and ResNet equivalents

Already Available code online under Creative Commons License

DenseNet.py  => DenseNet classifier 

train.py => trains DenseNet on Cifar10

gan.py => Runs DCGan on Cifar10

*************************************
*************************************

New Files - 

ganDense.py - This is the file we use going forward. DOESNOT WORK YET. Concatenating inputs is throwing incompatible shape error. Will have to debug better. Replace DCGAN with few residual connections.

summary.py - Prints summary of the architectures to understand architecture of models and sizes of layers. Works.

densenet_notransition.py - DenseNet without transition layers, with thresholded connections. Doesn't work yet, only created it to experiment and better understand DenseNet architecture

fake_samples => Generated Image from DCGAN
