import torch
import itertools

from configs.opts import *
from src.train import *
from src.test import *
from models.networks import get_network

import copy
import numpy as np

def safe_print(x):
	try:
		print(x)
	except:
		return

# INSTANTIATE TRAINING

NUM_DOMS=1
for i in DOMAINS:
	NUM_DOMS*=len(i)

# LOAD NETWORK
net = get_network(CLASSES, NUM_DOMS, residual=RESIDUAL)
net = net.to(DEVICE)

meta_vectors = torch.FloatTensor(NUM_DOMS,NUM_META).fill_(0)
edge_vals=torch.FloatTensor(NUM_DOMS,NUM_DOMS).fill_(0)
edge_vals_no_self=torch.FloatTensor(NUM_DOMS,NUM_DOMS).fill_(0)
full_list=[]

for meta in itertools.product(*DOMAINS):
		full_list.append(meta)
		meta_vectors[domain_converter(meta)]=get_meta_vector(meta)

for i,vector in enumerate(meta_vectors):
	edge_vals[i,:]=compute_edge(vector,meta_vectors,i,1.)
	edge_vals_no_self[i,:]=compute_edge(vector,meta_vectors,i,0.)

EXP=NUM_DOMS*(NUM_DOMS-1)

res_source=[]
res_upperbound=[]
res_adagraph=[]
res_adagraph_refinement_stats=[]
res_adagraph_refinement=[]

for meta_source in itertools.product(*DOMAINS):
			source_domain=meta_source

			net_std=copy.deepcopy(net).to(DEVICE)
			source_loader = init_loader(BATCH_SIZE, domains=[source_domain], shuffle=True, auxiliar=False, size=SIZE, std=STD)
			idx_source=domain_converter(source_domain)

			training_loop(net_std, source_loader, idx_source, epochs=EPOCHS, training_group=['bn','final'], store=None, auxiliar=False)


			for meta_target in itertools.product(*DOMAINS):
					target_domain=meta_target
					idx_target=domain_converter(meta_target)

					if idx_target == idx_source or skip_rule(meta_source,meta_target):
						continue



					test_loader = init_loader(BATCH_SIZE, domains=[target_domain], shuffle=False, auxiliar=False, size=SIZE, std=STD)

					current_res_source = test(net_std, test_loader, idx_source)


					res_source.append(current_res_source)
					safe_print(str(len(res_source))+' '+str(np.mean(np.array(res_source))))

np.save('./results/source'+SUFFIX+'.npy', np.array(res_source))
np.save('./results/adagraph'+SUFFIX+'.npy', np.array(res_adagraph))
np.save('./results/adagraph_refined_stats'+SUFFIX+'.npy', np.array(res_adagraph_refinement_stats))
np.save('./results/adagraph_refined'+SUFFIX+'.npy', np.array(res_adagraph_refinement))
np.save('./results/upper_bound'+SUFFIX+'.npy', np.array(res_upperbound))
