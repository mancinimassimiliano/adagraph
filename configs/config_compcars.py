# OPTS FOR COMPCARS
BATCH_SIZE = 16
TEST_BATCH_SIZE = 100
EPOCHS= 10
STEP=8
LR=0.0001
DECAY=0.000001
MOMENTUM=0.9
BANDWIDTH=0.1

DATES=['2009','2010','2011','2012','2013','2014']
VIEWS=['1','2','3','4','5']

DOMAINS = [DATES, VIEWS]


def domain_converter(meta):
	year, viewpoint = meta
	return viewpoint*len(DATES)+year

def init_loader(bs, domains=[], shuffle=False, auxiliar= False):
	if auxiliar:
		return load_settings.compcars(bs=bs, shuffle=shuffle, domains=domains)
	else:
		return ....

def compute_edge(x,dt,idx, self_connection = 1.):
	edge_w=torch.exp(-torch.pow(torch.norm(x.view(1,-1)-dt,dim=1),2)/(2.*BANDWIDTH))
	edge_w[idx]=edge_w[idx]*self_connection
	return edge_w/edge_w.sum()
