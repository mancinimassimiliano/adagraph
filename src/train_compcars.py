# Manager of the training procedure
from __future__ import print_function
from opts import options
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import load_settings
import residual_networks
import copy
import time
from torchvision.models import resnet


args = options().parse()

# Training settings
from config_compcars import BATCH_SIZE, TEST_BATCH_SIZE, EPOCHS, STEP, LR, DECAY, MOMENTUM, BANDWIDTH, DOMAINS, domain_converter, init_loader
from utils import compute_edge

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_net(classes=3, pretrained=None):
	net=residual_networks.resnet18_domain(classes)
	return net

NUM_DOMS=0
for i in DOMAINS:
	NUM_DOMS+=len(i)

NUM_META = len(DOMAINS)

meta_vectors = torch.FloatTensor(NUM_DOMS,NUM_META).fill_(0)
edge_vals=torch.FloatTensor(NUM_DOMS,NUM_DOMS).fill_(0)
edge_vals_no_self=torch.FloatTensor(NUM_DOMS,NUM_DOMS).fill_(0)
full_list=[]

for meta in itertools.product(*DOMAINS)
		full_list.append(meta)
		meta_vectors[domain_converter(meta)]=torch.FloatTensor([float(i) for i in meta])




for i,vector in enumerate(meta_vectors):
	edge_vals[i,:]=compute_edge(vector,meta_vectors,i,1.)
	edge_vals_no_self[i,:]=compute_edge(vector,meta_vectors,i,0.)

EXP=NUM_DOMS*(NUM_DOMS-1)



res_source=[]
res_upperbound=[]
res_adagraph=[]
res_adagraph_refinement_stats=[]
res_adagraph_refinement=[]

idx_d=0


def safe_print(x):
	try:
		print(x)
	except:
		return




suffix=args.suffix



net= get_net(classes=args.classes, pretrained=None)


upperbound_loader=init_ordered_loader(BATCH_SIZE, domains=full_list)

for meta_source in itertools.product(*DOMAINS):
			source_domain=meta_source

			net_std=copy.deepcopy(net).to(DEVICE)
			source_loader = init_loader(BATCH_SIZE,domains=[source_domain], shuffle=True)
			idx_source=domain_converter(source_domain)

			net_std.reset_edges()

			training_loop(net_std, source_loader, idx_source, epochs=EPOCHS, training_group=[''], store=None, auxiliar=False)
			net_std.copy_source(idx_source)

			net_upperbound=copy.deepcopy(net_std)
			net_upperbound.init_edges(edge_vals)   ##### CHOICE
			training_loop(net_upperbound,upperbound_loader, idx_source, epochs=1, training_group=['bn','downsample.1'], store=None, auxiliar=True)

			for meta_target in itertools.product(*DOMAINS):
					target_domain=meta_target
					idx_target=domain_converter(meta_target)

					if idx_target == idx_source:
						continue


					current_edges=copy.deepcopy(edge_vals)
					current_edges[:,idx_target]=0.0

					test_edges=copy.deepcopy(edge_vals_no_self)

					current_edges=current_edges/current_edges.sum(-1).unsqueeze(-1)
					current_edges[idx_target,:]=0.0

					domain_regressor=full_list[:]
					domain_regressor.remove(target_domain)

					regression_loader=init_ordered_loader(BATCH_SIZE, domains=domain_regressor)

					net_adagraph=copy.deepcopy(net_std)
					net_adagraph.init_edges(current_edges)

					training_loop(net_adagraph,regression_loader, idx_source, epochs=1, training_group=['bn','downsample.1'], store=None, auxiliar=True)

					target_loader = init_loader(BATCH_SIZE, domains=[target_domain], shuffle=False)
					test_loader = init_loader(TEST_BATCH_SIZE, domains=[target_domain], shuffle=False)
			

				    	current_res_source = test(net_std, test_loader, idx_source)


				    	current_adagraph_res = test(net_updated, test_loader, idx_source)
				    	res_bn_entropy[idx_d]= test(net_updated_entropy, test_loader, idx_source)

					net_adagraph.set_bn_from_edge1d(idx_target, ew=edge_vals[idx_target,:])
					net_adagraph.init_edges(edge_vals)

					current_res_adagraph = test(net_adagraph, test_loader, idx_target)
					current_res_refinement_stats, current_res_refinement = online_test(net_adagraph,idx_target,target_loader, device=DEVICE)

					
					current_res_upperbound = test(net_upperbound, test_loader, idx_target)


					res_source.append(current_res_source)
					res_adagraph.append(current_res_adagraph)
					res_adagraph_refinement_stats.append(current_res_adagraph_refinement)
					res_adagraph_refinement.append(current_res_adagraph_refinement)
					res_upperbound.append(current_res_upperbound)

					safe_print('-------------------------res after ' + str(idx_d)+'--------------------------')
					safe_print('RES STD    '+str(np.mean(np.array(res_source))))
					safe_print('RES ADAGRAPH    '+str(np.mean(np.array(res_adagraph))))
					safe_print('RES ADAGRAPH REF. STATS   '+str(np.mean(np.array(res_adagraph_refinement_stats))))
					safe_print('RES ADAGRAPH REF.    '+str(np.mean(np.array(res_adagraph_refinement))))
					safe_print('RES UPPER BOUND    '+str(np.mean(np.array(res_upperbound))))
					safe_print('')

np.save('./results/source'+suffix+'.npy', np.array(res_source))
np.save('./results/adagraph'+suffix+'.npy', np.array(res_adagraph))
np.save('./results/adagraph_refined_stats'+suffix+'.npy', np.array(res_adagraph_refinement_stats))
np.save('./results/adagraph_refined'+suffix+'.npy', np.array(res_adagraph_refinement))
np.save('./results/upper_bound'+suffix+'.npy', np.array(res_upperbound))



				
				
		
