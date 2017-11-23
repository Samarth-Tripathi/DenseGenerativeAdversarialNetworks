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

def crop_images(path):
	"""crop a image to nine sections and save image"""
	imgList = os.listdir(path)
	#count1 = 1 
	count2 = 0
	for image in imgList:
		if not image.startswith('.'):
			print(image[:])
			imagePath = image[:-4]
			os.mkdir(path + '/' + imagePath)
			img = Image.open(path + '/' + image)
			#print(np.asarray(img))
			w = img.size[0]
			h = img.size[1]
			for x in range(1, w-1, (w-2)//8):
				for y in range(1, h-1, (h-2)//8):
					newImage = img.crop((x, y, x+(w-2)//8, y+(h-2)//8))
					count2 = count2 + 1
					newImage.save(path + '/' + imagePath + '/' + str(count2) + '.png')
					print("saving images")

			count2 = 0
			#count1 = count1 + 1
	return

def normalize(arr):
	"""normalize an array to [0,1]"""
	arr = arr.astype('float32')
	if arr.max() > 1.0:
		arr /= 255.0
	return arr



def load_images(path):
	#return image array

	imageList = os.listdir(path)

	loadedImages = []
	for image in imageList:
		img = Image.open(path + '/' + image)
		img = np.asarray(img)
		#print(img.shape)
		img = np.transpose(img, (2, 0, 1))
		img = normalize(img)
		#print(img.shape)
		loadedImages.append(img)

	return loadedImages




def generator_score():
	#print generator loss in the network
	#print(-vphi_fake.mean())
	pass

def inception_score(imgs, cuda=gpu, batch_size=2, resize=True):
	"""Computes the inception score of the generated images imgs
		imgs -- list of (HxWx3) numpy images normalized in the range [0,1]
		cuda -- whether or not to run on GPU
		batch_size -- batch size to feed into inception
		https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
		"""

	N = len(imgs)
	#print(imgs[0].shape)

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

running_sum = 0
count = 0
folder_list = os.listdir(folder)
with open('inception_score', 'a') as file:
	for fol in folder_list:
		if os.path.isdir(folder + "/" + fol):
			imgs = load_images(folder + "/" + fol)
			score = inception_score(imgs)
			running_sum = running_sum + score
			count = count + 1
			print(fol)
			print(score)
			file.write(fol + ":" + str(score) + "\n")
	mean = running_sum / count
	print(mean)
	file.write(str(mean))


#crop_images(folder)

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





