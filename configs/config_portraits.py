import torch
import torchvision.transforms as transforms
from dataloaders.compcars import Compcars, CompcarsSampler

# OPTS FOR COMPCARS
BATCH_SIZE = 16
TEST_BATCH_SIZE = 100

EPOCHS= 2
STEP=1
LR=0.0001
DECAY=0.000001
MOMENTUM=0.9
BANDWIDTH=0.1

REGION_TO_IDX={'MA': 1,'NE': 2,'South': 3, 'Pacific': 4, 'MW': 0 }
IDX_TO_REGION={ 1:'MA',2:'NE',3:'South',4: 'Pacific', 0:'MW'}
REGION_TO_VEC={'MA': [0,.1],'NE': [0.,0.],'South': [1.,1.], 'Pacific': [0.,3.], 'MW': [0.,2.]}

REGIONS=['MA','NE','South', 'Pacific', 'MW']
DATES=[1934,1994,1944,1954,1964,1974,1984,2004]

CLASSES = 2

DOMAINS = [REGIONS, DATES]

DATALIST='.'

def domain_converter(meta):
	year, region = meta
	region = REGION_TO_IDX[region]
	return (year-DATES[0])//10*len(REGIONS) + region

TODO DATALOADER


def compute_edge(x,dt,idx, self_connection = 1.):
    edge_w=torch.exp(-torch.pow(torch.norm(x.view(1,-1)-dt,dim=1),2)/(2.*BANDWIDTH))
    edge_w[idx]=edge_w[idx]*self_connection
    return edge_w/edge_w.sum()

		dts[domain_converter(d, REGION_TO_IDX[r])]=torch.FloatTensor([float((d-START)//10)/7.,REGION_TO_VEC[r][0]/1.,REGION_TO_VEC[r][1]/3.])

def get_meta_vector(meta):
	year, region = meta
	latitude, longitude =  REGION_TO_VEC[region]
	return torch.FloatTensor([float((year-DATES[0])//10)/7.,latitude,longitude/3.])
