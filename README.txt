Dense-Generative models with DenseNet equivalent for Fisher GANs

###updates 10/27
Implemented SSL on the model, however after 8 epochs the loss hasn’t changed a bit. 

The last convolution layer is now followed by a fully-connected layer and then softmax is computed.
###





New Files -

DenseFisher/main_dense.py - run a Dense F-GAN with 2 dense connections

Run as (for Cifar 10) python3 main_dense.py --dataset cifar10 --dataroot dataroot/ --niter 50 --imageSize 32 --cuda

DenseFisher/models/densegan.py has Pytorch Models

DenseFisher/samples/ Has Generated Samples on Cifar10

DenseFisher/outputs/ Has output loss/epoch on Cifar10

Architecture -



###############################G###############################

torch.Size([64, 100, 1, 1])
torch.Size([64, 256, 4, 4])
torch.Size([64, 256, 4, 4])
New size= torch.Size([64, 512, 4, 4])
torch.Size([64, 128, 8, 8])
torch.Size([64, 128, 8, 8])
New size= torch.Size([64, 256, 8, 8])
torch.Size([64, 64, 16, 16])
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
###############################G###############################

###############################D###############################
torch.Size([64, 3, 32, 32])
torch.Size([64, 64, 16, 16])
torch.Size([64, 128, 8, 8])
torch.Size([64, 128, 8, 8])
New size= torch.Size([64, 256, 8, 8])
torch.Size([64, 256, 4, 4])
torch.Size([64, 256, 4, 4])
New size= torch.Size([64, 512, 4, 4])
torch.Size([64, 1, 1, 1])

###############################D###############################


Also in DCGAN folder you will find a DCGAN with Dense Connectioins - 

ganGraph.py - gan.py converted to graph like Neural Architecture instead of NN.sequential architecture, precursor to ganDense

ganDenseAlternate.py - DOESNOT WORK. Another attempt at a dense connection with different kernel size, do not use output padding in Generators, only input padding!

summary.py - Prints summary of the architectures to understand architecture of models and sizes of layers. Works.

densenet_notransition.py - DenseNet without transition layers, with thresholded connections. Doesn't work yet, only created it to experiment and better understand DenseNet architecture

fake_samples => Generated Image from DCGAN



Already Available code online under Creative Commons License

DenseNet.py => DenseNet classifier (https://github.com/bamos/densenet.pytorch/blob/master/densenet.py)

train.py => trains DenseNet on Cifar10 . (https://github.com/bamos/densenet.pytorch/blob/master/train.py)

gan.py => Runs DCGan on Cifar10 . (https://github.com/pytorch/examples/blob/master/dcgan/main.py)
