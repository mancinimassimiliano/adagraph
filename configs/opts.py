# OPTS FOR COMPCARS
import argparse

parser = argparse.ArgumentParser(description='AdaGraph')
parser.add_argument('--dataset', default = 'compcars', help='Dataset to test (compcars, faces)')
parser.add_argument('--network', default='resnet', type=str, help='Network to use (resnet, decaf)')
parser.add_argument('--skip', default=None, type=str, help='Skip some settings (required only for faces, eventually). Options are: regions,years.')
parser.add_argument('--suffix', default='./logs/adagraph_test', type=str, help='Suffix to give for storing the experiments')

args = parser.parse_args()

assert args.network in ['resnet','decaf'], 'Please specify a valid network in [resnet,decaf]'

DEVICE='cuda'
DATASET = args.dataset
RESIDUAL = args.network == 'resnet'
SKIP = args.skip
SUFFIX = args.suffix

if DATASET == 'compcars':
    from configs.config_compcars import *

elif DATASET == 'faces':
    from configs.config_faces import *

else:
    print("Please specify a valid dataset in [compcars,faces]")
    exit(1)
