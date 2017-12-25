Dense-Generative models with DenseNet equivalent for Fisher GANs

###updates 12/25

Please refer to the pdf file for details of the project.
We present GANs with deeper Generators using Dense connections with a simple DCGAN based Generators to achieve better quality of generated images. The dense generative models also converge faster and are more stable than vanilla Fisher or DCGANs.

main_dense.py - Main Fisher-GAN python file that runs for CIFAR-10 based generation, change the generator to your choice of models from models/densegan.py .
main_dense_celebA.py - Main Fisher-GAN python file that runs for CelebA based generation, change the generator to your choice of models from models/densegan_64.py .

Samples folder holds generated Samples from Generator architectures in models/ folder. 

In the models/densegan.py and densegan_64.py uncomment print statements to get the architectures.
