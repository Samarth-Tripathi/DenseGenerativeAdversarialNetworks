Dense-Generative models with DenseNet equivalent for Fisher GANs

###updates 11/1

Added Complete Dense Connected Generator and Discriminator (~50 layers each). Size is still hard coded for Cifar, so 32*32 (will make it a parameter) Architecture is available below. The Generations dont converge properly, needs more tweaking.

Please refer to the DenseFisher Folder for latest Source/Sample/Details.

######################D#####################
Input shape torch.Size([64, 3, 32, 32])
con1 shape torch.Size([64, 48, 32, 32])
dense internal shape 1 torch.Size([64, 48, 32, 32])
dense internal shape 2 torch.Size([64, 96, 32, 32])
dense internal shape 3 torch.Size([64, 24, 32, 32])
dense internal shape 4 torch.Size([64, 72, 32, 32])
transition internal shape 1torch.Size([64, 72, 32, 32])
transition internal shape 2torch.Size([64, 72, 32, 32])
transition internal shape 3torch.Size([64, 72, 16, 16])
dense + trans 1 finished shape torch.Size([64, 72, 16, 16])
dense internal shape 1 torch.Size([64, 72, 16, 16])
dense internal shape 2 torch.Size([64, 96, 16, 16])
dense internal shape 3 torch.Size([64, 24, 16, 16])
dense internal shape 4 torch.Size([64, 96, 16, 16])
transition internal shape 1torch.Size([64, 96, 16, 16])
transition internal shape 2torch.Size([64, 96, 16, 16])
transition internal shape 3torch.Size([64, 96, 8, 8])
dense + trans 2 finished shape torch.Size([64, 96, 8, 8])
dense internal shape 1 torch.Size([64, 96, 8, 8])
dense internal shape 2 torch.Size([64, 96, 8, 8])
dense internal shape 3 torch.Size([64, 24, 8, 8])
dense internal shape 4 torch.Size([64, 120, 8, 8])
transition internal shape 1torch.Size([64, 120, 8, 8])
transition internal shape 2torch.Size([64, 120, 8, 8])
transition internal shape 3torch.Size([64, 120, 4, 4])
dense + trans 3 finished shape torch.Size([64, 120, 4, 4])
dense internal shape 1 torch.Size([64, 120, 4, 4])
dense internal shape 2 torch.Size([64, 96, 4, 4])
dense internal shape 3 torch.Size([64, 24, 4, 4])
dense internal shape 4 torch.Size([64, 144, 4, 4])
transition internal shape 1torch.Size([64, 144, 4, 4])
transition internal shape 2torch.Size([64, 144, 4, 4])
transition internal shape 3torch.Size([64, 144, 2, 2])
dense + trans 4 finished shape torch.Size([64, 144, 2, 2])
dense internal shape 1 torch.Size([64, 144, 2, 2])
dense internal shape 2 torch.Size([64, 96, 2, 2])
dense internal shape 3 torch.Size([64, 24, 2, 2])
dense internal shape 4 torch.Size([64, 168, 2, 2])
transition internal shape 1torch.Size([64, 168, 2, 2])
transition internal shape 2torch.Size([64, 168, 2, 2])
transition internal shape 3torch.Size([64, 168, 1, 1])
dense + trans 5 finished shape torch.Size([64, 168, 1, 1])
Final shape torch.Size([64, 1, 1, 1])
######################D#####################
######################G#####################
Input shape torch.Size([64, 100, 1, 1])
conv1 shape torch.Size([64, 168, 1, 1])
dense internal shape 1 torch.Size([64, 168, 1, 1])
dense internal shape 2 torch.Size([64, 96, 1, 1])
dense internal shape 3 torch.Size([64, 24, 1, 1])
dense internal shape 4 torch.Size([64, 192, 1, 1])
transition internal shape 1torch.Size([64, 192, 1, 1])
transition internal shape 2torch.Size([64, 144, 1, 1])
transition internal shape 3torch.Size([64, 144, 2, 2])
dense + trans 1 finished shape torch.Size([64, 144, 2, 2])
dense internal shape 1 torch.Size([64, 144, 2, 2])
dense internal shape 2 torch.Size([64, 96, 2, 2])
dense internal shape 3 torch.Size([64, 24, 2, 2])
dense internal shape 4 torch.Size([64, 168, 2, 2])
transition internal shape 1torch.Size([64, 168, 2, 2])
transition internal shape 2torch.Size([64, 120, 2, 2])
transition internal shape 3torch.Size([64, 120, 4, 4])
dense + trans 2 finished shape torch.Size([64, 120, 4, 4])
dense internal shape 1 torch.Size([64, 120, 4, 4])
dense internal shape 2 torch.Size([64, 96, 4, 4])
dense internal shape 3 torch.Size([64, 24, 4, 4])
dense internal shape 4 torch.Size([64, 144, 4, 4])
transition internal shape 1torch.Size([64, 144, 4, 4])
transition internal shape 2torch.Size([64, 96, 4, 4])
transition internal shape 3torch.Size([64, 96, 8, 8])
dense + trans 3 finished shape torch.Size([64, 96, 8, 8])
dense internal shape 1 torch.Size([64, 96, 8, 8])
dense internal shape 2 torch.Size([64, 96, 8, 8])
dense internal shape 3 torch.Size([64, 24, 8, 8])
dense internal shape 4 torch.Size([64, 120, 8, 8])
transition internal shape 1torch.Size([64, 120, 8, 8])
transition internal shape 2torch.Size([64, 72, 8, 8])
transition internal shape 3torch.Size([64, 72, 16, 16])
dense + trans 4 finished shape torch.Size([64, 72, 16, 16])
dense internal shape 1 torch.Size([64, 72, 16, 16])
dense internal shape 2 torch.Size([64, 96, 16, 16])
dense internal shape 3 torch.Size([64, 24, 16, 16])
dense internal shape 4 torch.Size([64, 96, 16, 16])
transition internal shape 1torch.Size([64, 96, 16, 16])
transition internal shape 2torch.Size([64, 48, 16, 16])
transition internal shape 3torch.Size([64, 48, 32, 32])
dense + trans 5 finished shape torch.Size([64, 48, 32, 32])
Output shape torch.Size([64, 3, 32, 32])
######################G#####################



###


###updates 10/27
Implemented SSL on the model, however after 8 epochs the loss hasn’t changed a bit. 

The last convolution layer is now followed by a fully-connected layer and then softmax is computed.
###




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
