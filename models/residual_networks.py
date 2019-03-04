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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DomainBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, domains=30):
        super(DomainBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = GraphBN(planes, domains=domains)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = GraphBN(planes, domains=domains)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x,t):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out,t)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,t)
        if self.downsample is not None:
            residual = self.downsample(x,t)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DomainBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, domains=30):
        super(DomainBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = GraphBN(planes, domains=domains)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = GraphBN(planes, domains=domains)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = GraphBN(planes * self.expansion, domains=domains)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, t):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out,t)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,t)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out,t)

        if self.downsample is not None:
            residual = self.downsample(x,t)

        out += residual
        out = self.relu(out)

        return out

class DomainSequential(nn.Sequential):
	def forward(self,input,t=None):
		for module in self._modules.values():
			if isinstance(module, GraphBN) or isinstance(module, DomainBlock) or isinstance(module, DomainBottleneck):
            			input = module(input,t)
			else:
				input = module(input)
		return input

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x,t):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




class DomainResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, domains=30):
        self.inplanes = 64
        super(DomainResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = GraphBN(64, domains=domains)
        self.relu = nn.ReLU(inplace=False)
	self.domains=domains
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
	print('Init domain ResNet with ' + str(self.domains) + '\n')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #elif isinstance(m, nn.BatchNorm2d):
             #   nn.init.constant_(m.weight, 1)
              #  nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DomainSequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                GraphBN(planes * block.expansion, domains=self.domains),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, domains=self.domains))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, domains=self.domains))

        return DomainSequential(*layers)

    def forward(self, x,t):
        x = self.conv1(x)
        x = self.bn1(x,t)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x,t)
        x = self.layer2(x,t)
        x = self.layer3(x,t)
        x = self.layer4(x,t)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    def set_bn_from_edge1d(self,idx, ew=None):
	for m in self.modules():
            if isinstance(m, GraphBN):
		m.set_bn_from_edge1d(idx,ew=ew)


    def copy_source(self,idx):
	for m in self.modules():
            if isinstance(m, GraphBN):
		m.copy_source(idx)


    def init_edges(self,edges):
	for m in self.modules():
            if isinstance(m, GraphBN):
		m.init_edges(edges)

    def print_grads(self):
	for m in self.modules():
            if isinstance(m, GraphBN):
		print(m.grads())



class GraphBN(nn.Module):
    def __init__(self, features, domains=30,dim=2, sw=1.0):
        super(GraphBN, self).__init__()

	self.domains=domains
	self.features=features
	self.dims=dim
	self.sw=1.0

	if dim==1:
		self.bns=nn.ModuleList([nn.BatchNorm1d(features,affine=False) for i in range(domains)])
	else:
		self.bns=nn.ModuleList([nn.BatchNorm2d(features,affine=False) for i in range(domains)])

	self.scale = nn.Parameter(torch.FloatTensor(domains,features).fill_(1.))
	self.bias = nn.Parameter(torch.FloatTensor(domains,features).fill_(0.))
	self.edges=torch.FloatTensor(domains,domains).fill_(0.)
	self.edges.requires_grad=False

    def combined_sb2d(self,d):
	full_scale=self.edges.data[d].view(-1,1)*self.scale
	full_bias=self.edges.data[d].view(-1,1)*self.bias
	return full_scale.sum(0).view(1,self.features,1,1), full_bias.sum(0).view(1,self.features,1,1)
	
    def forward(self, x, d):
        x=self.bns[d](x)
	scale,bias=self.combined_sb2d(d)
        return scale*x+bias

    def forward_node(self, x, d):
        x=self.bns[d](x)
        return self.scale[d]*x+self.bias[d]

    def init_edges(self, edges):
	self.edges=edges.to(self.scale.device)
	self.edges.requires_grad=False

    def get_weighted_stats1d(self,d):
	means=torch.FloatTensor(self.features).fill_(0.).to(self.scale.device)
	stds=torch.FloatTensor(self.features).fill_(0.).to(self.scale.device)
	for i,bn in enumerate(self.bns):
		means=means+self.edges[d,i]*bn.running_mean
		stds=stds+self.edges[d,i]*bn.running_var
	return means, stds

    def set_edge(self,d,ws):
	self.edges.data[d]=ws.to(self.edges.device) 
	self.edges.requires_grad=False

    def get_edges(self):
	return self.edges.data[:] 
 
    def get_sbs(self):
	return self.scale.data[:],self.bias.data[:]   

    def combined_sb1d(self,d):
	full_scale=self.edges[d].view(-1,1)*self.scale
	full_bias=self.edges[d].view(-1,1)*self.bias
	return full_scale.sum(0), full_bias.sum(0)

    def set_bn_from_edge1d(self,d, ew=None):
	if ew is not None:
		self.set_edge(d,ew)
	means, stds=self.get_weighted_stats1d(d)
	self.bns[d].running_mean.data[:]=means
	self.bns[d].running_var.data[:]=stds

	scale,bias=self.combined_sb1d(d)
	self.scale.data[d]=scale
	self.bias.data[d]=bias


    def grads(self):
	return self.scale.grad.data, self.bias.grad.data


    def copy_source(self, idx):
	
	self.scale.data[:]=self.scale.data[:]*0.+self.scale[idx,:].data[:].view(1,-1)
	self.bias.data[:]=self.bias.data[:]*0.+self.bias[idx,:].data[:].view(1,-1)
	for d in range(self.domains):
		self.bns[d].running_mean.data[:]=self.bns[idx].running_mean.data[:]
		self.bns[d].running_var.data[:]=self.bns[idx].running_var.data[:]
		

    def check_source(self, idx):
	
	self.scale.data[:]=self.scale.data[:]*0.+self.scale[idx,:].data[:].view(1,-1)
	self.bias.data[:]=self.scale.data[:]*0.+self.bias[idx,:].data[:].view(1,-1)
	for d in range(self.domains):
		self.bns[d].running_mean.data[:]=self.bns[idx].running_mean.data[:]
		self.bns[d].running_var.data[:]=self.bns[idx].running_var.data[:]

    def add_new(self,ws,wa):
	self.append()
	self.set_edge(-1,ws)
	self.set_bn_from_edge1d(-1)
	self.set_edge(-1,wa)	



	

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18_domain(num_classes=4, domains=30, **kwargs):
    	"""Constructs a ResNet-18 model.
    	Args:
        	pretrained (bool): If True, returns a model pre-trained on ImageNet
    	"""
    	model_origin = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    	model_origin.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    	model = DomainResNet(DomainBlock, [2, 2, 2, 2], domains=domains, **kwargs)
	for m in model_origin.named_modules():
	    if 'bn' in m[0] or 'downsample.1' in m[0]:
		for m2 in model.named_modules():
		    if m2[0]==m[0]:
		        #print(m2[0],m[0])
		        for i in range(domains):
		            m2[1].bns[i].running_var[:]=m[1].running_var.data[:]
		            m2[1].bns[i].running_mean[:]=m[1].running_mean.data[:]          
		            m2[1].scale.data[i,:]=m[1].weight.data[:]                                   
		            m2[1].bias.data[i,:]=m[1].bias.data[:]


	for m in model_origin.named_modules():
	    if 'conv' in m[0] or 'downsample.0' in m[0]:
		for m2 in model.named_modules():
		    if m2[0]==m[0]:
			#print(m2[0],m[0])
		        m2[1].weight.data[:]=m[1].weight.data[:]
			    
	d1=d2=d3=d4=0      
	'''for m in model_origin.named_modules():
	    if 'bn' in m[0]:
		for m2 in model.named_modules():
		    if m2[0]==m[0]:
		        for i in range(30):                                     
		            d1+=(m2[1].bns[i].running_var[:]-m[1].running_var.data[:]).sum()
		            d2+=(m2[1].bns[i].running_mean[:]-m[1].running_mean.data[:]).sum()
		            d3+=(m2[1].scale[i,:]-m[1].weight[:]).sum()
		            d4+=(m2[1].bias[i,:]-m[1].bias[:]).sum()
	if(d1+d2+d3+d4)==0:
		print('Test passed, correct init')'''

	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, num_classes)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.0001)
	

    	return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
