import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

import argparse
# parse args first and set gpu id
parser = argparse.ArgumentParser()
parser.add_argument('--points_batch', type=int, default=16384, help='point batch size') #from 16384 to 8192
# parser.add_argument('--nepoch', type=int, default=100000, help='number of epochs to train for') #default
parser.add_argument('--nepoch', type=int, default=20000, help='number of epochs to train for')
parser.add_argument('--conf', type=str, default='setup.conf')
parser.add_argument('--expname', type=str, default='single_shape')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU auto]')
parser.add_argument('--is_continue', default=False, action="store_true", help='continue')
parser.add_argument('--timestamp', default='latest', type=str) #not used anymore
parser.add_argument('--checkpoint', default='latest', type=str)
parser.add_argument('--eval', default=False, action="store_true")
parser.add_argument('--splitpos', default = -1, type = int, help='for testing the best position for spliting the network')
parser.add_argument('--summary', default = True, help = 'write summary')
parser.add_argument('--laplace', default = False, help = 'use laplacian term')
parser.add_argument('--baseline', default = False, action="store_true", help = 'run baseline')
parser.add_argument('--onehot', default = False, action="store_true", help = 'set onehot flag as true')
parser.add_argument('--cpu', default = False, action="store_true", help = 'save for cpu device')
# parser.add_argument('--branch', default = -1, type = int, help='number of branches')
args = parser.parse_args()


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(args.gpu)

from datetime import datetime
from pyhocon import ConfigFactory
import numpy as np
import GPUtil
import torch
import torch.nn as nn
import utils.general as utils
from model.sample import Sampler
from model.network import gradient
from scipy.spatial import cKDTree
from utils.plots import plot_surface, plot_cuts, plot_masks, plot_masks_maxsdf
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


model_prefix_list = '/mnt/sdf1/haog/code/IGR/code/reconstruction/tmp.txt'
model_path = '/mnt/sdf1/haog/code/IGR/CADdata'
output_path = '/mnt/sdf1/haog/code/IGR'

def convert_sdf(model_prefix):
	conf_filename = args.conf
	conf = ConfigFactory.parse_file('./reconstruction/' + conf_filename)
	# feature_mask = utils.load_feature_mask()
	feature_mask_file = os.path.join(model_path, model_prefix + '_mask.txt')
	feature_mask = torch.tensor(np.loadtxt(feature_mask_file)).float()

	# input_file = conf.get_string('train.input_path')
	input_file = os.path.join(model_path, model_prefix + '.xyz')
	csg_tree = ConfigFactory.parse_file(input_file[:-4]+'_csg.conf').get_list('csg.list')
	csg_flag_convex = ConfigFactory.parse_file(input_file[:-4]+'_csg.conf').get_int('csg.flag_convex')
	device = torch.device('cuda')
	if args.cpu:
		device = torch.device('cpu')

	if args.baseline:
	    feature_mask = torch.ones(feature_mask.shape).float()

	nb = int(torch.max(feature_mask).item())
	onehot = False
	if args.onehot:
	    onehot = True

	network = utils.get_class(conf.get_string('train.network_class'))(d_in=3, split_pos = -1, flag_output = 1,
	                                                                                    n_branch = nb,
	                                                                                    csg_tree = csg_tree,
	                                                                                    flag_convex = csg_flag_convex,
	                                                                                    flag_onehot = onehot,
	                                                                                    **conf.get_config(
	                                                                                    'network.inputs'))
	network.to(device)

	# old_checkpnts_dir = os.path.join(self.expdir, self.timestamp, 'checkpoints')

	# ori version

	foldername = conf.get_string('train.folderprefix').strip() + model_prefix
	if args.cpu:
		saved_model_state = torch.load('/mnt/sdf1/haog/code/IGR/exps/single_shape/' + foldername + '/checkpoints/ModelParameters/latest.pth', map_location=device)
		network.load_state_dict(saved_model_state["model_state_dict"])
	else:
		saved_model_state = torch.load('/mnt/sdf1/haog/code/IGR/exps/single_shape/' + foldername + '/checkpoints/ModelParameters/latest.pth')
		network.load_state_dict(saved_model_state["model_state_dict"])
	print('loading finish')
	#trace
	example = torch.rand(224,3).to(device)
	traced_script_module = torch.jit.trace(network, example)
	# traced_script_module = torch.jit.script(network)
	# if args.onehot:
	# 	traced_script_module.save(output_path + foldername + "_model_h_oh.pt")
	# else:
	traced_script_module.save(os.path.join(output_path, foldername + "_latest_min1379_model_h.pt"))

if __name__ == '__main__':
	f = open(model_prefix_list, 'r')
	lines = f.readlines()
	f.close()
	for l in lines:
		l = l.strip()[:-1]
		convert_sdf(l)


# # modified version
# ckpt_sdf = torch.load('/mnt/sdf1/haog/code/IGR/exps/single_shape/nofea_fandisk_cot_rev_nocolor/checkpoints/ModelParameters/latest.pth')
# ckpt_mask = torch.load('/mnt/sdf1/haog/code/IGR/exps/single_shape/nofea_fandisk_cot_rev_nocolor_svm_maskonly/checkpoints/ModelParameters/latest.pth')

# for k in list(ckpt_sdf["model_state_dict"].keys()):
# # 	# print('key: ', k)
# 	if k.startswith('sdf'):
# 		# print('mask key: ', k)
# 		ckpt_mask["model_state_dict"][k] = ckpt_sdf["model_state_dict"][k]

# # for k in list(ckpt_mask["model_state_dict"].keys()):
# # 	if k.startswith('sdf'):
# # 		print('sdf key: ', k)
# # 		del ckpt_mask["model_state_dict"][k]

# network.load_state_dict(ckpt_mask["model_state_dict"])
# # network.load_state_dict(ckpt_mask, strict=False)

# example = torch.rand(224,3).to(device)
# traced_script_module = torch.jit.trace(network, example)
# # traced_script_module = torch.jit.script(network)
# traced_script_module.save("/mnt/sdf1/haog/code/IGR/merge_svm.pt")
