import torch
import itertools

from configs.opts import *
from src.train import *
from src.test import *
from models.networks import get_network
import random

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
res_refined=[]
res_upperbound=[]
res_upperbound_ref=[]
res_adagraph=[]
res_adagraph_refinement=[]

upperbound_loader=init_loader(BATCH_SIZE, domains=full_list, auxiliar= True, size=SIZE, std=STD)

for meta_source in itertools.product(*DOMAINS):
			source_domain=meta_source

			net_std=copy.deepcopy(net).to(DEVICE)
			source_loader = init_loader(BATCH_SIZE, domains=[source_domain], shuffle=True, auxiliar=False, size=SIZE, std=STD)
			idx_source=domain_converter(source_domain)

			net_std.reset_edges()

			training_loop(net_std, source_loader, idx_source, epochs=EPOCHS, training_group=SOURCE_GROUP, store=None, auxiliar=False)
			net_std.copy_source(idx_source)

			net_upperbound=copy.deepcopy(net_std)
<<<<<<< HEAD
			net_upperbound.init_edges(edge_vals)
=======
			net_upperbound.init_edges(edge_vals)   ##### CHOICE
>>>>>>> 70cace8b93e035ca74281f5768ab58fce0b2e520

			training_loop(net_upperbound,upperbound_loader, idx_source,epochs=1, training_group=TRAINING_GROUP, store=None, auxiliar=True)

			for meta_target in itertools.product(*DOMAINS):
					target_domain=meta_target
					idx_target=domain_converter(meta_target)

					if idx_target == idx_source or skip_rule(meta_source,meta_target):
						continue


					safe_print(str(meta_source) + ' vs ' + str(meta_target))

					current_edges=copy.deepcopy(edge_vals)
					current_edges[:,idx_target]=0.0

					current_edges=current_edges/current_edges.sum(-1).unsqueeze(-1)
					current_edges[idx_target,:]=0.0

					available_domains=full_list[:]
					available_domains.remove(target_domain)

					auxiliar_loader=init_loader(BATCH_SIZE, domains=available_domains, auxiliar=True, size=SIZE, std=STD)

					net_adagraph=copy.deepcopy(net_std)
					net_adagraph.init_edges(current_edges)

					training_loop(net_adagraph,auxiliar_loader, idx_source, target_idx=idx_target, epochs=1, training_group=TRAINING_GROUP, store=None, auxiliar=True)

					target_loader = init_loader(BATCH_SIZE, domains=[target_domain], shuffle=True, auxiliar=False, size=SIZE, std=STD)
					test_loader = init_loader(TEST_BATCH_SIZE, domains=[target_domain], shuffle=False, auxiliar=False, size=SIZE, std=STD)

					current_res_source = test(net_std, test_loader, idx_source)

					net_adagraph.set_bn_from_edges(idx_target, ew=edge_vals_no_self[idx_target,:])
					net_adagraph.init_edges(edge_vals)

					net_refined = copy.deepcopy(net_std)

					current_res_adagraph = test(net_adagraph, test_loader, idx_target)
					current_res_refined = online_test(net_refined,idx_target,target_loader, training_group=TRAINING_GROUP, device=DEVICE, bs=BATCH_SIZE)
					current_res_ag_refinement = online_test(net_adagraph,idx_target,target_loader, training_group=TRAINING_GROUP, device=DEVICE, bs=BATCH_SIZE)
					current_res_upperbound = test(net_upperbound, test_loader, idx_target)
					current_res_upperbound_refined = online_test(net_upperbound,idx_target,target_loader, training_group=TRAINING_GROUP, device=DEVICE, bs=BATCH_SIZE)


					res_source.append(current_res_source)
					res_refined.append(current_res_refined)
					res_adagraph.append(current_res_adagraph)
					res_adagraph_refinement.append(current_res_ag_refinement)
					res_upperbound.append(current_res_upperbound)
					res_upperbound_ref.append(current_res_upperbound_refined)



					safe_print('-------------------------res after ' + str(len(res_source))+'--------------------------')
<<<<<<< HEAD
					safe_print('BASELINE    '+str(np.mean(np.array(res_source))))
					safe_print('BASELINE + REFINEMENT    '+str(np.mean(np.array(res_refined))))
					safe_print('ADAGRAPH    '+str(np.mean(np.array(res_adagraph))))
					safe_print('ADAGRAPH + REFINEMENT    '+str(np.mean(np.array(res_adagraph_refinement))))
					safe_print('UPPER BOUND    '+str(np.mean(np.array(res_upperbound))))
					safe_print('UPPER BOUND + REFINEMENT  '+str(np.mean(np.array(res_upperbound_ref))))
=======
					safe_print('RES STD    '+str(np.mean(np.array(res_source))))
					safe_print('RES REFINED    '+str(np.mean(np.array(res_refined))))
					safe_print('RES ADAGRAPH    '+str(np.mean(np.array(res_adagraph))))
					safe_print('RES ADAGRAPH + REF.    '+str(np.mean(np.array(res_adagraph_refinement))))
					safe_print('RES UPPER BOUND    '+str(np.mean(np.array(res_upperbound))))
					safe_print('RES UPPER BOUND + REF.  '+str(np.mean(np.array(res_upperbound_ref))))
>>>>>>> 70cace8b93e035ca74281f5768ab58fce0b2e520
					safe_print('')

np.save('./results/source'+SUFFIX+'.npy', np.array(res_source))
np.save('./results/refined'+SUFFIX+'.npy', np.array(res_refined))
np.save('./results/adagraph'+SUFFIX+'.npy', np.array(res_adagraph))
np.save('./results/adagraph_refined'+SUFFIX+'.npy', np.array(res_adagraph_refinement))
np.save('./results/upper_bound'+SUFFIX+'.npy', np.array(res_upperbound))
np.save('./results/upper_bound_refined'+SUFFIX+'.npy', np.array(res_upperbound_ref))
