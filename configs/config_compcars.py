import torch
import torchvision.transforms as transforms
from dataloaders.compcars import Compcars, CompcarsSampler


# OPTS FOR COMPCARS
BATCH_SIZE = 16
TEST_BATCH_SIZE = 100

EPOCHS= 6
STEP=4
LR=0.0001
DECAY=0.000001
MOMENTUM=0.9
BANDWIDTH=0.1


DATES=['2009','2010','2011','2012','2013','2014']
VIEWS=['1','2','3','4','5']

CLASSES = 4

DOMAINS = [DATES, VIEWS]
NUM_META = 2

DATALIST='.' #TO CHANGE


def domain_converter(meta):
	year, viewpoint = meta
	year = int(year)-int(DATES[0])
	viewpoint = int(viewpoint)-int(VIEWS[0])
	return viewpoint*len(DATES)+year


def init_loader(bs, domains=[], shuffle=False, auxiliar= False, size=224, std=[0.229, 0.224, 0.225]):
    data_transform=transforms.Compose([
            transforms.Resize(256),
                   transforms.CenterCrop(size),
                      transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std)
         ])


    dataset = Compcars(DATALIST,['1','2','3','4'],transform=data_transform,domains=domains)

    if not auxiliar:
        return torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=False,num_workers=4, shuffle=shuffle)

    else:
        return torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=False,num_workers=4, sampler=CompcarsSampler(dataset,bs))


def compute_edge(x,dt,idx, self_connection = 1.):
    edge_w=torch.exp(-torch.pow(torch.norm(x.view(1,-1)-dt,dim=1),2)/(2.*BANDWIDTH))
    edge_w[idx]=edge_w[idx]*self_connection
    return edge_w/edge_w.sum()


def get_meta_vector(meta):
	return torch.FloatTensor([float(i) for i in meta])
