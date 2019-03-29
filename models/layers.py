import torch
import torch.nn as nn
import torch.nn.functional as F

#### Definition of the Graph-based BN layer ####
class GraphBN(nn.Module):

    # Init layer
    def __init__(self, features, domains=30,dim=2):
        super(GraphBN, self).__init__()
        # Number of domains
        self.domains=domains

        # Size of input features
        self.features=features

        # Init BN considering 2 spatial dims
        self.bns=nn.ModuleList([nn.BatchNorm2d(features,affine=False) for i in range(domains)])

        # Init scale and bias parameters
        self.scale = nn.Parameter(torch.FloatTensor(domains,features).fill_(1.))
        self.bias = nn.Parameter(torch.FloatTensor(domains,features).fill_(0.))

        # Init edge values
        self.edges=torch.FloatTensor(domains,domains,1).fill_(0.).to('cuda')


    # Forward pass with scale and bias obtained from the graph
    def forward(self, x, d):
        x=self.bns[d](x)
        scale,bias=self.combine_sb(d)
        return scale*x+bias


    # Copy stats from source
    def copy_source(self, idx):
        self.scale.data[:]=self.scale.data[:]*0.+self.scale[idx,:].data[:].unsqueeze(0)
        self.bias.data[:]=self.bias.data[:]*0.+self.bias[idx,:].data[:].unsqueeze(0)
        for d in range(self.domains):
            self.bns[d].running_mean.data[:]=self.bns[idx].running_mean.data[:]
            self.bns[d].running_var.data[:]=self.bns[idx].running_var.data[:]


    ###########################
    #### GRAPH PROPAGATION ####
    ###########################

    # Combine scale and bias with respect to graph edges
    def combine_sb(self,d):
        full_scale = self.edges.data[d]*self.scale
        full_bias = self.edges.data[d]*self.bias
        final_scale = full_scale.sum(0).view(1,self.features,1,1)
        final_bias = full_bias.sum(0).view(1,self.features,1,1)
        return final_scale, final_bias


    # Get stats from nearby nodes
    def combine_stats(self,d):
        means=torch.FloatTensor(self.features).fill_(0.).to(self.scale.device)
        stds=torch.FloatTensor(self.features).fill_(0.).to(self.scale.device)
        for i,bn in enumerate(self.bns):
            means=means+self.edges[d,i]*bn.running_mean
            stds=stds+self.edges[d,i]*bn.running_var
            return means, stds


    # Initialize BN of a node from an edge
    def set_bn_from_edges(self, d, ew=None):
        if ew is not None:
            self.set_edge(d,ew)
        means, stds=self.combine_stats(d)
        self.bns[d].running_mean.data[:]=means
        self.bns[d].running_var.data[:]=stds
        scale,bias=self.combine_sb(d)
        self.scale.data[d]=scale.view(-1)
        self.bias.data[d]=bias.view(-1)


    # Initialize BN of a node from probabilties on domains
    def set_bn_from_probs(self, x, w):
        means=torch.FloatTensor(w.size(0),self.features).fill_(0.).to(self.scale.device)
        stds=torch.FloatTensor(w.size(0),self.features).fill_(0.).to(self.scale.device)
        scale=torch.FloatTensor(w.size(0),self.features).fill_(0.).to(self.scale.device)
        bias=torch.FloatTensor(w.size(0),self.features).fill_(0.).to(self.scale.device)

        probs = w.unsqueeze(2)
        means = (probs*self.collected_means).sum(1)
        stds = (probs*self.collected_stds).sum(1)
        scale = (probs*self.scale.unsqueeze(0)).sum(1)
        bias = (probs*self.bias.unsqueeze(0)).sum(1)
        x=(x-means)/(torch.pow(stds+self.bns[0].eps,0.5))
        return scale*x+bias

    ############################
    #### EDGES MANIPULATION ####
    ############################


    # Edges initialization
    def init_edges(self, edges):
        self.edges=edges.to(self.scale.device).unsqueeze(-1)
        self.edges.requires_grad=False


    # Set a single edge
    def set_edge(self,d,ws):
        self.edges.data[d]=ws.to(self.edges.device).unsqueeze(-1)
        self.edges.requires_grad=False

    # Get edges
    def get_edges(self):
        return self.edges.data[:]


    # Get scale and bias
    def get_sb(self):
        return self.scale.data[:],self.bias.data[:]

    # Have all the edges with just the self connection on
    def reset_edges(self):
        self.edges.data.fill_(0.)
        eye = torch.eye(self.domains)
        self.edges.data[:]=eye.unsqueeze(-1)





# Entropy loss definition
class EntropyLoss(nn.Module):
    ''' Module to compute entropy loss '''
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(-1).mean()
        return b
