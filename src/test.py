# SET UP EVALUATION
import torch
from models.layers import EntropyLoss
from configs.opts import *
from src.train import set_up_optim, filter_params
import copy


# Test
def test(net, loader, domain, device='cuda'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, meta, targets) in enumerate(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            if domain is None:
                current_domain = domain_converter(meta[0])
            else:
                current_domain = domain
            outputs= net(inputs,current_domain)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    return acc



def single_eval(net,inputs,domain,targets):
	net.eval()
	output = net(inputs,domain)
	output.detach()
	_, predicted = output.max(1)
	correct = predicted.eq(targets).sum().item()
	return correct


def single_update(net,inputs,domain,optimizer=None, criterion=None):
	if optimizer is not None and criterion is not None:
		optimizer.zero_grad()
		net.train()
		prediction=net(inputs,domain)
		loss=criterion(prediction)
		loss.backward()
		optimizer.step()
		net.eval()
	else:
		with torch.no_grad():
			net.train()
			net(inputs,domain)
			net.eval()



def online_test(net, domain, loader_online,training_group=['bn','downsample.1'], device='cuda', bs=16):
    net_entropy = copy.deepcopy(net)

    correct_entropy=0.
    totals=0.

    criterion=EntropyLoss()

    filter_params(net_entropy, training_group)
    optimizer = set_up_optim(net_entropy, LR*0.1, auxiliar=True)

    for batch_idx, (inputs, meta, targets) in enumerate(loader_online):
        if domain is None:
            current_domain = domain_converter(meta[0])
        else:
            current_domain = domain

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        totals += targets.size(0)

        correct_entropy += single_eval(net_entropy, inputs, current_domain, targets)

        if targets.size(0)==bs:
            single_update(net_entropy, inputs, current_domain, optimizer, criterion)

    return 100.*correct_entropy/totals
