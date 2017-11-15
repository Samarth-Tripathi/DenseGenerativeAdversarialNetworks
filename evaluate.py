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
import torch.utils.data
from torchvision.models.inception import inception_v3
from torch.autograd import Variable

import sys
import argparse
import os 
from PIL import Image 
from scipy.stats import entropy

#DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to the images')
parser.add_argument('--cuda', default=False, help='whether to use gpu or not')
opt = parser.parse_args()

print(opt)

folder = opt.dataroot
gpu = opt.cuda

def load_images(path):
	#return image array

	imageList = os.listdir(path)

	loadedImages = []
	for image in imageList:
		img = Image.open(path + '/' + image)
		img = np.asarray(img)
		#print(img.shape)
		img = np.transpose(img, (2, 0, 1))
		img = img/255.0
		#print(img.shape)
		loadedImages.append(img)

	return loadedImages



def generator_score():
	#print generator loss in the network
	#print(-vphi_fake.mean())
	pass

def inception_score(imgs, cuda=gpu, batch_size=2, resize=False):
	"""Computes the inception score of the generated images imgs
		imgs -- list of (HxWx3) numpy images normalized in the range [0,1]
		cuda -- whether or not to run on GPU
		batch_size -- batch size to feed into inception
		https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
		"""

	N = len(imgs)
	print(imgs[0].shape)

	assert batch_size > 0

		
	assert N > batch_size


	if cuda:
		dtype = torch.cuda.FloatTensor

	else:
		if torch.cuda.is_available():
			print("You have an unused CUDA device")

		dtype = torch.FloatTensor

	dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)


	#load model
	inception_model = inception_v3(pretrained=True, transform_input=True).type(dtype)
	inception_model.eval()
	up = nn.Upsample(size=(299,299), mode='bilinear').type(dtype)

	def get_pred(x):
		if resize:
			x = up(x)

		x = inception_model(x)

		return F.softmax(x)

	#predictions
	preds = np.zeros((N,1000))

		#class ImageDataSet(torch.utils.data.Dataset):
			#def __init__(self, imgs):
				#self.imgs = imgs

			#def __getitem__(self, index):
				#return self.imgs[index]

			#def __len(self):
				#return len(self.imgs)

	#imgs_data = ImageDataSet(imgs)

	for i, batch in enumerate(dataloader, 0):
		batch = batch.type(dtype)
		batchv = Variable(batch)
		batch_size_i = batch.size()[0]

		p_out = get_pred(batchv)
		preds[i*batch_size:i*batch_size + batch_size_i] = p_out.data.cpu().numpy()


	#compute mean k1-div
	py = np.mean(preds, axis=0)
	scores = []
	for i in range(preds.shape[0]):
		pyx = preds[i, :]
		scores.append(entropy(pyx, py))
		mean_kl = np.mean(scores)

	return np.exp(mean_kl)


def ssl_performance():
		pass


imgs = load_images(folder)
print(inception_score(imgs))

'''if __name__ == '__main__':
    print ("Generating images...")
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, imgs):
            self.imgs = imgs

        def __getitem__(self, index):
            return self.imgs[index]

        def __len__(self):
            return len(self.imgs)
    imgs = [np.random.uniform(0, 1, size=(3, 32, 32)).astype(np.float32) for _ in range(10)]
    imgs_dset = ImageDataset(imgs)

    print ("Calculating Inception Score...")
    print (inception_score(imgs_dset, cuda=False, batch_size=2, resize=True))
    '''





