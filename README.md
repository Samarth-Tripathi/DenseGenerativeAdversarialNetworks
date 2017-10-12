# Dense-Generators Generative models with DenseNet and ResNet equivalents 

New Files - 

ganDense.py - Finally works, a DCGAN with 1 Dense Connection in both Generator and Discriminator each.

Architecture - 

###############################D###############################

torch.Size([64, 3, 64, 64])

torch.Size([64, 64, 32, 32])

torch.Size([64, 128, 16, 16])

x=torch.Size([64, 256, 8, 8])

y=torch.Size([64, 512, 8, 8])

x,y concatenated.

New size= torch.Size([64, 768, 8, 8])

torch.Size([64, 512, 4, 4])

torch.Size([64, 1, 1, 1])

torch.Size([64, 1, 1, 1])

###############################D###############################


###############################G###############################

torch.Size([64, 100, 1, 1])

torch.Size([64, 512, 4, 4])

x=torch.Size([64, 256, 8, 8])

y=torch.Size([64, 512, 8, 8])

x,y concatenated.

New size= torch.Size([64, 768, 8, 8])

torch.Size([64, 128, 16, 16])

torch.Size([64, 64, 32, 32])

torch.Size([64, 3, 64, 64])

torch.Size([64, 3, 64, 64])


###############################G###############################


ganGraph.py - gan.py converted to graph like Neural Architecture instead of NN.sequential architecture, precursor to ganDense


ganDenseAlternate.py - DOESNOT WORK. Another attempt at a dense connection with different kernel size, do not use output padding in Generators, only input padding!


summary.py - Prints summary of the architectures to understand architecture of models and sizes of layers. Works.


densenet_notransition.py - DenseNet without transition layers, with thresholded connections. Doesn't work yet, only created it to experiment and better understand DenseNet architecture


fake_samples => Generated Image from DCGAN


*************************************
*************************************


Already Available code online under Creative Commons License


DenseNet.py  => DenseNet classifier  (https://github.com/bamos/densenet.pytorch/blob/master/densenet.py)


train.py => trains DenseNet on Cifar10 .     (https://github.com/bamos/densenet.pytorch/blob/master/train.py)


gan.py => Runs DCGan on Cifar10 .  (https://github.com/pytorch/examples/blob/master/dcgan/main.py)




