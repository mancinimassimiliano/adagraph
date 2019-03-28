from configs.opts import *
from models.networks import *
from src.train import *
from src.test import *

def safe_print(x):
	try:
		print(x)
	except:
		return


if not RESIDUAL:
    STD =


# INSTANTIATE TRAINING

NUM_DOMS=0
for i in DOMAINS:
	NUM_DOMS*=len(i)

# LOAD NETWORK
net = get_network(CLASSES, NUM_DOMS, residual=RESIDUAL)

NUM_META = len(DOMAINS)

meta_vectors = torch.FloatTensor(NUM_DOMS,NUM_META).fill_(0)
edge_vals=torch.FloatTensor(NUM_DOMS,NUM_DOMS).fill_(0)
edge_vals_no_self=torch.FloatTensor(NUM_DOMS,NUM_DOMS).fill_(0)
full_list=[]

for meta in itertools.product(*DOMAINS)
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

					if idx_target == idx_source or skip_rule(meta_source,meta_target):
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


					net_adagraph.set_bn_from_edge1d(idx_target, ew=edge_vals[idx_target,:])
					net_adagraph.init_edges(edge_vals)

					current_res_adagraph = test(net_adagraph, test_loader, idx_target)
					current_res_refinement_stats, current_res_refinement = online_test(net_adagraph,idx_target,target_loader, device=DEVICE)

					current_res_upperbound = test(net_upperbound, test_loader, idx_target)


					res_source.append(current_res_source)
					res_adagraph.append(current_res_adagraph)
					res_adagraph_refinement_stats.append(current_res_refinement_stats)
					res_adagraph_refinement.append(current_res_refinement)
					res_upperbound.append(current_res_upperbound)



					safe_print('-------------------------res after ' + str(len(res_source))+'--------------------------')
					safe_print('RES STD    '+str(np.mean(np.array(res_source))))
					safe_print('RES ADAGRAPH    '+str(np.mean(np.array(res_adagraph))))
					safe_print('RES ADAGRAPH REF. STATS   '+str(np.mean(np.array(res_adagraph_refinement_stats))))
					safe_print('RES ADAGRAPH REF.    '+str(np.mean(np.array(res_adagraph_refinement))))
					safe_print('RES UPPER BOUND    '+str(np.mean(np.array(res_upperbound))))
					safe_print('')

np.save('./results/source'+SUFFIX+'.npy', np.array(res_source))
np.save('./results/adagraph'+SUFFIX+'.npy', np.array(res_adagraph))
np.save('./results/adagraph_refined_stats'+SUFFIX+'.npy', np.array(res_adagraph_refinement_stats))
np.save('./results/adagraph_refined'+SUFFIX+'.npy', np.array(res_adagraph_refinement))
np.save('./results/upper_bound'+SUFFIX+'.npy', np.array(res_upperbound))
