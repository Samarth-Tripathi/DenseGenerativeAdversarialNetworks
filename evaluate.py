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

	#gathering inputs
	inputs = []
	for img in images:
		img = img.astype(np.float32)
		inputs.append(np.expand_dims(img, 0))
	
	#batches
	bs = 100
	n_bathces = int(math.ceil(float(len(inps))/float(bs)))

	#get predictions
	preds = []
	for i in range(n_batches):
		sys.stdout.write(".")
		sys.stdout.write(str(i))
		sys.stdout.flush()

		inp = inputs[(i * bs):min((i + 1) * bs, len(inputs))]
		inp = np.concatenate(inp, 0)
		pred = nn.Softmax(inp)
		preds.append(pred)
	preds = np.concatenate(preds, 0)

	#get score
	scores = []
	for i in range(splits):
		part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
		kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
		kl = np.mean(np.sum(kl,1))
		scores.append(np.exp(kl))

	return np.mean(score), np.std(score)


def ssl_performance():
	pass



		

