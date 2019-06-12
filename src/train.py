# SET UP OPTIMIZERS AND TRAININGS
import torch
import torch.optim as optim
import torch.nn as nn
from models.layers import EntropyLoss
from configs.opts import *




# Full training procedure
def training_loop(net, loader, domain, target_idx=-1, epochs=10, training_group=["bn"], store=None, auxiliar=False):
    lr=LR
    if auxiliar:
        lr=lr*0.1

    filter_params(net, training_group)
    optimizer = set_up_optim(net, lr, auxiliar, RESIDUAL)

    for epoch in range(1, 1+epochs):
        # Perform 1 training epoch
        train(net, domain, target_idx, loader, optimizer)
        if epoch==STEP:
            lr=lr*0.1
            optimizer = set_up_optim(net, lr, auxiliar, RESIDUAL)

    if store is not None:
        state={
        'state_dict': net.state_dict()
        }
        torch.save(state, store)



# Training, single epoch
def train(net, source, idx_target, loader, optimizer):
    net.train()
    train_loss = 0

    criterion=nn.CrossEntropyLoss()
    entropy=EntropyLoss()

    optimizer.zero_grad()


    for batch_idx, (inputs, meta, targets) in enumerate(loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        if targets.size(0)==1:
            continue

        current_domain=domain_converter((meta[0][0],meta[1][0]))

        assert current_domain != idx_target, "The target is used while training AdaGraph. This is wrong."

        # Produce features
        prediction = net(inputs, current_domain)

        if current_domain==source:
            loss=criterion(prediction, targets)
        else:
            loss=entropy(prediction)

        # Backward + update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    return (train_loss/(batch_idx+1))




def filter_params(net, training_group):
	for name,param in net.named_parameters():
			include=False
			for l in training_group:
				if l in name:
					include=True
					break
			if include:
				param.requires_grad=True
			else:
				param.requires_grad=False




def set_up_optim(net, lr, auxiliar=False, residual=True):
    if auxiliar or (not residual):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=DECAY)
    else:
        optimizer = optim.Adam([
				{'params': net.conv1.parameters()},
				{'params': net.bn1.parameters()},
				{'params': net.layer1.parameters()},
				{'params': net.layer2.parameters()},
				{'params': net.layer3.parameters()},
				{'params': net.layer4.parameters()},
				{'params': net.fc.parameters(), 'lr': lr*10}
		    ], lr=lr, weight_decay=DECAY)

    return optimizer
