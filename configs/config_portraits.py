import torch
import torchvision.transforms as transforms
from dataloaders.portraits import *

# OPTS FOR COMPCARS
BATCH_SIZE = 16
TEST_BATCH_SIZE = 100

EPOCHS= 1
STEP=2
LR=0.0001
DECAY=0.000001
MOMENTUM=0.9
BANDWIDTH=0.1

REGIONS=['MA','NE','South', 'Pacific', 'MW']
DATES=[1934,1994,1944,1954,1964,1974,1984,2004]

CLASSES = 2

DOMAINS = [DATES, REGIONS]
NUM_META = 3

DATAROOT='./data/faces/'

REGION_TO_VEC={'MA': [0,.1],'NE': [0.,0.],'South': [1.,1.], 'Pacific': [0.,3.], 'MW': [0.,2.]}

def domain_converter(meta):
	year, region = meta
	region = REGIONS_TO_IDX[region]
	return (year-DATES[0])//10*len(REGIONS) + region



def init_loader(bs, domains=[], shuffle=False, auxiliar= False, size=224, std=[0.229, 0.224, 0.225]):
    data_transform=transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std)
         ])


    dataset = Portraits(DATAROOT,transform=data_transform,domains=domains)

    if not auxiliar:
        return torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=False,num_workers=4, shuffle=shuffle)

    else:
        return torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=False,num_workers=4, sampler=PortraitsSampler(dataset,bs))




def compute_edge(x,dt,idx, self_connection = 1.):
    edge_w=torch.exp(-torch.pow(torch.norm(x.view(1,-1)-dt,dim=1),2)/(2.*BANDWIDTH))
    edge_w[idx]=edge_w[idx]*self_connection
    return edge_w/edge_w.sum()


def get_meta_vector(meta):
	year, region = meta
	latitude, longitude =  REGION_TO_VEC[region]
	return torch.FloatTensor([float((year-DATES[0])//10)/7.,latitude,longitude/3.])
