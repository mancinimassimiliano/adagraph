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

from models.layers import GraphBN


########################################
##### Define ResNet bulding blocks #####
########################################

class DomainBlock(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, domains=30):
        super(DomainBlock, self).__init__(inplanes, planes, stride, downsample)
        self.bn1 = GraphBN(planes, domains=domains)
        self.bn2 = GraphBN(planes, domains=domains)

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


class DomainBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, domains=30):
        super(DomainBottleneck, self).__init__(inplanes, planes, stride, downsample)

        self.bn1 = GraphBN(planes, domains=domains)
        self.bn2 = GraphBN(planes, domains=domains)
        self.bn3 = GraphBN(planes * self.expansion, domains=domains)

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





##################################
##### Define AdaGraph ResNet #####
##################################
class DomainResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, domains=30):
        super(DomainResNet, self).__init__()
        self.inplanes = 64
        self.domains=domains

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.bn1 = GraphBN(64, domains=domains)


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


    def set_bn_from_edges(self,idx, ew=None):
        for m in self.modules():
            if isinstance(m, GraphBN):
                m.set_bn_from_edges(idx,ew=ew)


    def copy_source(self,idx):
        for m in self.modules():
            if isinstance(m, GraphBN):
                m.copy_source(idx)

    def reset_edges(self):
        for m in self.modules():
            if isinstance(m, GraphBN):
                m.reset_edges()


    def init_edges(self,edges):
        for m in self.modules():
            if isinstance(m, GraphBN):
                m.init_edges(edges)



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
def resnet18_domain(classes=4, domains=30, url=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
    	num_classes (int): the number of classes of the classification model
    	domains (int): the number of domains included in the model (#source + #auxuliary)
    """
    # Instantiate original ResNet
    model_origin = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model_origin.load_state_dict(model_zoo.load_url(url))

    # Instatiate Domain ResNet
    model = DomainResNet(DomainBlock, [2, 2, 2, 2], domains=domains, **kwargs)

    # Copy BN stats and params from the original to the domain-based
    c=0
    for m_orig in model_origin.named_modules():
        if 'bn' in m_orig[0] or 'downsample.1' in m_orig[0]:
            for m_doms in model.named_modules():
                if m_doms[0]==m_orig[0]:
                    c+=1
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
    model.fc = nn.Linear(num_ftrs, classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0,0.0001)

    return model
