# OPTS FOR COMPCARS
import argparse

parser = argparse.ArgumentParser(description='AdaGraph')
parser.add_argument('--dataset', default = 'compcars', help='Dataset to test (compcars, portraits)')
parser.add_argument('--network', default='resnet', type=str, help='Network to use (resnet, decaf)')
parser.add_argument('--skip', default=None, type=str, help='Skip some settings (required only for portraits, eventually). Options are: regions,years.')
parser.add_argument('--suffix', default='./logs/adagraph_test', type=str, help='Suffix to give for storing the experiments')

args = parser.parse_args()

assert args.network in ['resnet','decaf'], 'Please specify a valid network in [resnet,decaf]'

DEVICE='cuda'
DATASET = args.dataset
RESIDUAL = args.network == 'resnet'
SKIP = args.skip
SUFFIX = args.suffix
SOURCE_GROUP = ['']
STD = [0.229, 0.224, 0.225]
SIZE = 224

if DATASET == 'compcars':
    from configs.config_compcars import *

elif DATASET == 'portraits':
    from configs.config_portraits import *

else:
    print("Please specify a valid dataset in [compcars,faces]")
    exit(1)

TRAINING_GROUP = ['bn','downsample.1']

if not RESIDUAL:
    EPOCHS = 10
    STEP = 7
    STD = [1./256, 1./256, 1./256]
    SIZE = 227
    LR = 0.001
    SOURCE_GROUP = ['bn','final']

if SKIP == 'regions':
    def skip_rule(meta_source, meta_target):
        source_year, source_region = meta_source
        target_year, target_region = meta_target
        return source_year != target_year
elif SKIP== 'years':
    def skip_rule(meta_source, meta_target):
        source_year, source_region = meta_source
        target_year, target_region = meta_target
        return source_region != target_region
else:
    def skip_rule(meta_source, meta_target):
        return False
