import torch 
import torch.nn as nn
import torch.nn.functional as F

def generate_label(batch_size, label, nlabels = 11):
	"""
	batch_size: how many labels to generate
	label: label to generate 
	nlabels: number of labels in each sample
	"""
	labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
	y = torch.zeros((batch_size, nlabels))
	y.scatter_(1, labels, 1)
	return y.type(torch.LongTensor)


def onehot(k, label):
	"""
	k: length of vector
	label: position of label
	"""
	def hot_vector(label):
		y = torch.LongTensor(k)
		y.zero_()
		y[label] = 1
		return y
	return hot_vector(label)

class Classifier(nn.Modules):
	def __init__(self, dims):
		"""
		dims: dimension of input
		can change output length later
		"""
		super(Classifier, self).__init__()
		[x_dim, h_dim, y_dim] = dims 
		self.fc1 = nn.Linear(x_dim * h_dim * y_dim, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 11)
		self.output_activation = F.softmax

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.output_activation(self.fc3(x))
		return x



