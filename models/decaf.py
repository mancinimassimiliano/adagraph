import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision.models import ResNet,DenseNet, VGG, AlexNet
from torchvision.models.resnet import BasicBlock, Bottleneck, conv3x3
from torchvision import models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import copy

from layers import GraphBN


###############################
##### Define Base Modules #####
###############################

class Identity(nn.Module):

    def __init__(self, sub=1.0):
        super(Identity, self).__init__()
	self.sub=sub

    def forward(self,x):
	return x*self.sub


class LockGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return 0.*grad_output

def grad_lock(x):
    return LockGrad.apply(x)



class DomainPredictor(nn.Module):

    def __init__(self, features=4096, domains=1):
        super(DomainPredictor, self).__init__()
	self.features=features
	self.domains=domains
	
	#self.mapper=nn.Linear(features, features/8)
	self.predictor=nn.Linear(features, domains)

	for m in self.modules():
		if isinstance(m, nn.Linear):
                	m.weight.data.normal_(0,0.001)

    def forward(self,x):
	#x=self.mapper(F.relu(x))
	x=self.predictor(F.relu(x))
	return x




##################################
##### Define AdaGraph ResNet #####
##################################


class AlexNet_BVLC(nn.Module):
    def __init__(self, num_classes=1000, dropout=False):
        super(AlexNet_BVLC, self).__init__()
	self.features = nn.Sequential(OrderedDict([
		("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
		("relu1", nn.ReLU(inplace=True)),
		("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
		("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
		("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
		("relu2", nn.ReLU(inplace=True)),
		("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
		("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
		("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
		("relu3", nn.ReLU(inplace=True)),
		("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
		("relu4", nn.ReLU(inplace=True)),
		("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
		("relu5", nn.ReLU(inplace=True)),
		("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
	]))
	self.classifier = nn.Sequential(OrderedDict([
		("fc6", nn.Linear(256 * 6 * 6, 4096)),
		("relu6", nn.ReLU(inplace=True)),
		("drop6", nn.Dropout() if dropout else Identity(sub=0.5)),
		("fc7", nn.Linear(4096, 4096)),
		("relu7", nn.ReLU(inplace=True)),
		("drop7", nn.Dropout() if dropout else Identity(sub=0.5)),
		("fc8", nn.Linear(4096, 1000))
	]))
	self.final=nn.Linear(4096, num_classes)

    def forward(self, x, t):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        x = self.classifier._modules['drop6'](x)
        x = self.classifier._modules['fc7'](x)
        x = self.classifier._modules['relu7'](x)
        x = self.classifier._modules['drop7'](x)
	x = self.final(x)
        return x


    def fix(self, alpha):
	for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum=alpha

    def do_stuff(self, x, t):
	return

    def to_classes(self, num_classes):
	num_ftrs = self.final.in_features
	self.final = nn.Linear(num_ftrs, num_classes)
	self.final.weight.data.normal_(0,0.001)







class AlexNet_BN(AlexNet_BVLC):

    def __init__(self, num_classes=1000, dropout=False, bn_after=False):
        super(AlexNet_BN, self).__init__(num_classes=1000, dropout=False)
	self.bn=nn.BatchNorm1d(4096)
	self.bn2=nn.BatchNorm1d(4096)
	self.bn_after=bn_after

	for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x,t):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        x = self.classifier._modules['drop6'](x)
        x = self.classifier._modules['fc7'](x)
	x = self.bn(x)
        x = self.classifier._modules['relu7'](x)
	x = self.final(x)
        return x


    def to_classes(self, num_classes):
	num_ftrs = self.final.in_features
	if self.bn_after:
		self.final = nn.Linear(num_ftrs, num_classes, bias=False)
		self.bn2=nn.BatchNorm1d(num_classes)
	else:
		self.final = nn.Linear(num_ftrs, num_classes)
		self.final.weight.data.normal_(0,0.001)





class AlexNet_GraphBN(AlexNet_BN):

    def __init__(self, num_classes=1000, dropout=False, bn_after=False, domains=29):
        super(AlexNet_GraphBN, self).__init__(num_classes=1000, dropout=False)

	self.bns=GraphBN(4096, domains=domains)

	self.final=nn.Linear(4096, num_classes)


	for m in self.modules():
		if isinstance(m, nn.Linear):
                	m.weight.data.normal_(0,0.01)

	self.bandwidth=0.01


    def forward(self, x,t):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        x1 = self.classifier._modules['fc7'](x)
	x1 = x1.view(x1.shape[0],-1,1,1)
	x = self.bns(x1,t)
	x = x.view(x.shape[0],-1)
        x = self.classifier._modules['relu7'](x)
	x = self.final(x)
        return x



##############################
##### INSTANTIATE MODELS #####
##############################

# Standard ResNet
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


# AdaGraph ResNet
def resnet18_domain(num_classes=4, domains=30, **kwargs):
    	"""Constructs a ResNet-18 model.
    	Args:
        	num_classes (int): the number of classes of the classification model
        	domains (int): the number of domains included in the model (#source + #auxuliary)
    	"""

	# Instantiate original ResNet
    	model_origin = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    	model_origin.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

	# Instatiate Domain ResNet
    	model = DomainResNet(DomainBlock, [2, 2, 2, 2], domains=domains, **kwargs)

	# Copy BN stats and params from the original to the domain-based
	for m_orig in model_origin.named_modules():
	    if 'bn' in m_orig[0] or 'downsample.1' in m_orig[0]:
		for m_doms in model.named_modules():
		    if m_doms[0]==m_orig[0]:
		        for i in range(domains):
		            m_doms[1].bns[i].running_var[:]=m_orig[1].running_var.data[:]
		            m_doms[1].bns[i].running_mean[:]=m_orig[1].running_mean.data[:]          
		            m_doms[1].scale.data[i,:]=m_orig[1].weight.data[:]                                   
		            m_doms[1].bias.data[i,:]=m_orig[1].bias.data[:]

	    elif 'conv' in m_orig[0] or 'downsample.0' in m_orig[0]:
		for m_doms in model.named_modules():
		    if m_doms[0]==m_orig[0]:
		        m_doms[1].weight.data[:]=m_orig[1].weight.data[:]

	# Init classifier    
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, num_classes)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.0001)

    	return model



##############################
##### INSTANTIATE MODELS #####
##############################

def get_graph_net(classes=3, domains=30, url=None):

	model=AlexNet_GraphBN(domains=domains)
	state = model.state_dict()
	state.update(torch.load(url))
	model.load_state_dict(state)

	model.to_classes(classes)

	return model


