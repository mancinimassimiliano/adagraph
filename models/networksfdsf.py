import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import wbn_layers
import math
import numpy as np
from torchvision.models import ResNet,DenseNet, VGG, AlexNet
from torchvision import models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import copy

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'alexnet': './alexnet_caffe/alexnet_caffe.pth.tar'
}












def safe_print(x):
	try:
		print(x)
	except:
		return




class AlexNet_GraphBN(AlexNet_BN):

    def __init__(self, num_classes=1000, dropout=False, bn_after=False, domains=29,dim=1,sw=1.0):
        super(AlexNet_GraphBN, self).__init__(num_classes=1000, dropout=False)

	self.bns=GraphBN(4096, domains=domains,dim=dim, sw=sw)

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
	x = self.bns(x1,t)
        x = self.classifier._modules['relu7'](x)
	x = self.final(x)
        return x


    def forward_test(self, x,t):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        x = self.classifier._modules['fc7'](x)
	x = self.bns.forward_node(x,t)
        x = self.classifier._modules['relu7'](x)
	x = self.final(x)
        return x




class AlexNet_GraphBN_X(AlexNet_BN):

    def __init__(self, num_classes=1000, dropout=False, bn_after=False, domains=29,dim=1,sw=1.0):
        super(AlexNet_GraphBN_X, self).__init__(num_classes=1000, dropout=False)

	self.bns=GraphBN(4096, domains=domains,dim=dim, sw=sw)

	self.final=nn.Linear(4096, num_classes)
	self.domain_classifier1=DomainPredictor(4096, 5)
	self.domain_classifier2=DomainPredictor(4096, 6)


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
	x = self.bns(x1,t)
        x = self.classifier._modules['relu7'](x)
	x = self.final(x)
	d1=self.domain_classifier1(grad_lock(F.relu(x1)))
	d2=self.domain_classifier2(grad_lock(F.relu(x1)))
        return x, d1, d2

    def forward_img(self, x, idx):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        x1 = self.classifier._modules['fc7'](x)
	d1=F.softmax(self.domain_classifier1((F.relu(x1))),dim=1)
	d2=F.softmax(self.domain_classifier2((F.relu(x1))),dim=1)
	probs=torch.bmm(d1.unsqueeze(2), d2.unsqueeze(1)).view(x.size(0),-1)
	x = self.bns.bn_from_probs1d(x1,probs)
        x = self.classifier._modules['relu7'](x)
	x = self.final(x)
        return x, d1, d2

    def forward_test(self, x,t):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        x = self.classifier._modules['fc7'](x)
	x = self.bns.forward_node(x,t)
        x = self.classifier._modules['relu7'](x)
	x = self.final(x)
        return x

    def forward_img_check(self, x,t):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        x = self.classifier._modules['fc7'](x)
	x = self.bns.forward_node_manual(x,t)
        x = self.classifier._modules['relu7'](x)
	x = self.final(x)
        return x



def get_graph_net(classes=3, domains=30, dim=1, bn=False,img=False):
	if img:
		model=AlexNet_GraphBN_X(domains=domains,dim=dim)
	else:
		model=AlexNet_GraphBN(domains=domains,dim=dim)
	
	state = model.state_dict()
	state.update(torch.load(model_urls['alexnet']))
	model.load_state_dict(state)

	model.to_classes(classes)

	return model








