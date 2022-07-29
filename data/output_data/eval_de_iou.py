import os
import glob
import trimesh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--skip', action='store_true', help = 'skip existing one')
parser.add_argument('--gpu', default=0, type=int, help = 'gpu')
parser.add_argument('--s', default=0, type=int, help = 'suffix id')
args = parser.parse_args()

cuda_device = args.gpu

max_edge_length = 1.8 / 32

files = glob.iglob('./*_model_h.pt')

gt_folder = '../gt_mesh'
allname = ''
suffix_id = args.s
for f in files:
	output_file = f.replace('.pt', '_eval.txt')
	flag_process = True
	if args.skip and os.path.exists(output_file):
		flag_process = False
	if flag_process:
		gt_file = os.path.join(gt_folder, f.replace('_50k_model_h.pt', '_normalized.obj'))
		print(f)
		os.system('CUDA_VISIBLE_DEVICES={} ../../code/IsoSurfaceGen/build/App/evaluation/evaluation -i {} -g {}'.format(cuda_device, f, gt_file))