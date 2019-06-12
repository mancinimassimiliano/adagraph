# OPTS FOR COMPCARS
import argparse

class options():
  def __init__(self):
    self.parser = argparse.Argument.parser(description='3D Keypoint')
  
  def init(self):
	self.parser = argparse.Argument.parser(description='DA model for Continuos/Predictive DA')
	self.parser.add_argument('--store', type=str, default='./models/test_regressor',
		            help='Where to store the checkpoints')
	self.parser.add_argument('--pretrained', type=str, default='./models/resnet_compcars_',
		            help='Which model to use for test')
	self.parser.add_argument('--lr', type=float, default=0.001,
		            help='Learning rate.')
	self.parser.add_argument('--epochs', type=int, default=10,
		            help='Number of epochs')
	self.parser.add_argument('--step', type=int, default=8,
		            help='Step point during training')
	self.parser.add_argument('--bs', type=int, default=16,
		            help='Batch size')
	self.parser.add_argument('--bn_decay', type=int, default=0.1,
		            help='Decay of BN statistics')
	self.parser.add_argument('--test_bs', type=int, default=100,
		            help='Test batch size')
	self.parser.add_argument('--bn', type=int, default=1,
		            help='BN usage')
	self.parser.add_argument('--entropy', type=float, default=0.0,
		            help='Entropy loss weight.')
	self.parser.add_argument('--classes', type=int, default=4,
		            help='Output classes')
	self.parser.add_argument('--decay', type=float, default=0.00001,
		            help='Weight decay.')
	self.parser.add_argument('--bandwidth', type=float, default=0.1,
		            help='Bandwidth.')

  def parse(self):
    self.init()
    self.args = self.parser.parse_args()
    return self.args
