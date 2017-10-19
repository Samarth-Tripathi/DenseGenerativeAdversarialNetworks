"""This class provides evaluation methods for the FisherGan
GAN
1) Inception score
2) If we fix the model, then we can compare the loss of the generator
3) Classifier performance on Semi-supervised learning

By Renbo Tu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

def generator_score():
	#print generator loss in the network
	#print(-vphi_fake.mean())
	pass

def get_inception_score(images, splits=10):
	assert(type(images) == list)
	assert(type(images[0]) == np.ndarray)
	assert(len(images[0].shape) == 3)

	inputs = []

	for img in images:
		img = img.astype(np.float32)
		inputs.append(np.expand_dims(img, 0))
	bs = 100

	preds = []
	n_bathces = int(math.ceil(float(len(inps))/float(bs)))



def _init_inception():
	global softmax
	filepath = None
	if not os.path.exists(filepath):
		def _progress(count, block_size, totalsize):



if spftmax is None:
	_init_inception()



		

