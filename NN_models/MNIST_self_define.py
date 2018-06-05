import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch as th
import torch.optim as optim
import torchvision
import torch
from torch.nn.modules.module import Module
from torchvision import datasets, transforms
import cv2
import os
import numpy as np
import math
import NN_models.ops as ops
def train_test(training):
	transform = transforms.Compose([transforms.ToTensor(),
								   transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
	data_train = datasets.MNIST(root = os.path.join('NN_models','data/'),
								transform=transform,
								train = True,
								download = True)

	data_test = datasets.MNIST(root=os.path.join('NN_models','data/'),
							   transform = transform,
							   train = False)
	data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
													batch_size = 64,
													shuffle = True)

	data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
												   batch_size = 64,
												   shuffle = True)


	class Net(nn.Module):
		def __init__(self):
			super(Net,self).__init__()
			self.cov1 = nn.Conv2d(1,6,5)
			self.cov2 = nn.Conv2d(6,16,5)
			self.fc1 = nn.Linear(16*4*4,120)
			self.fc2 = nn.Linear(120,84)
			self.fc3 = nn.Linear(84,10)

		def forward(self,x):
			x = F.max_pool2d(self_define_torch(self.cov1(x)),(2,2))
			x = F.max_pool2d(self_define_torch(self.cov2(x)),(2,2))
			x = x.view(-1,16*4*4)
			x = self_define_torch(self.fc1(x))
			x = self_define_torch(self.fc2(x))
			x = self.fc3(x)
			return x
		def forward_apx(self,x):
			x = F.max_pool2d(ops.self_define_torch(self.cov1(x)), (2, 2))
			x = F.max_pool2d(ops.self_define_torch(self.cov2(x)), (2, 2))
			x = x.view(-1, 16 * 4 * 4)
			x = ops.self_define_torch(self.fc1(x))
			x = ops.self_define_torch(self.fc2(x))
			x = self.fc3(x)
			return x

	net = Net()
	net = net.cuda()
	creterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	if(training == True):
		for epoch in range(3):
			running_loss = 0
			for i,data in enumerate(data_loader_train, 0):
				input,label = data
				input,label = Variable(input).cuda(), Variable(label).cuda()
				optimizer.zero_grad()
				output = net.forward(input)
				loss = creterion(output,label)
				loss.backward()
				optimizer.step()
				running_loss += loss.data.item()
				print('the running_loss in epoch %d step %d is %f'%(epoch,i,running_loss))
				running_loss = 0.0
				torch.save(net.state_dict(), os.path.join('NN_models','MNIST_data','model_parameter_self.pkl'))
	else:
		net.load_state_dict(torch.load(os.path.join('NN_models','MNIST_data','model_parameter_self.pkl')))
		correct = 0
		total = 0
		for data in data_loader_test:
			images, labels = data
			outputs = net.forward(Variable(images).cuda())
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += np.sum((predicted==labels).numpy())
		print('Accuracy of the network on the 10000 test images: %f' % (correct / total))
		acc1 = str(correct / total)
		correct = 0
		total = 0
		for data in data_loader_test:
			images, labels = data
			outputs = net.forward_apx(Variable(images).cuda())
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += np.sum((predicted==labels).numpy())
		print('apx Accuracy of the network on the 10000 test images: %f' % (correct / total))
		with open(os.path.join('NN_models','Acc', 'MNIST_acc.txt'), 'a') as f:
			f.write(file_name + ' ')
			f.write(acc1 + ' ')
			f.write(acc2 + ' \r\n')