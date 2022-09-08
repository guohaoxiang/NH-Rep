import os
from multiprocessing import Pool
import platform

in_path = 'raw_input'
out_path = 'training_data'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-r', action = 'store_true', help = 'generating training data for repaired data')
args = parser.parse_args()


sys_info = platform.system()

if sys_info == 'Linux':
    #linux version
    feature_sampling_path = './Bin/FeatureSample'
    para_sample_path = './Bin/ParametricSample'
else:
    #windows version
    feature_sampling_path = r'.\Bin\FeatureSample.exe'
    para_sample_path = r'.\Bin\ParametricSample.exe'


def gen_one_para_sample(yamlfile):
    yaml_path = os.path.join(in_path, yamlfile)
    obj_path = os.path.join(in_path, yamlfile.replace('.yml', '.obj'))
    os.system("{} -y {} -o {}".format(para_sample_path, yaml_path, obj_path))

    fea_path =           obj_path.replace('.obj','.fea')
    xyz_path = os.path.join(out_path, yamlfile.replace('.yml', '_50k.xyz'))
    mask_path = os.path.join(out_path, yamlfile.replace('.yml', '_50k_mask.txt'))

    para_pts_path =      obj_path.replace('.obj', '_50k.xyz')
    para_ptsface_path =  obj_path.replace('.obj', '_50k_tris.txt')

    os.system("{} -i {} -o {} -f {} -k {} -m 1 --ns 50000 --fs 0 -c 1 --csg 1 --repairtree --convex 1 -r 0 --mp 6 -p {} --pf {}".format(feature_sampling_path, obj_path, xyz_path, fea_path, mask_path, para_pts_path, para_ptsface_path))

    #copy points data sampled from points to target directory
    if sys_info == 'Linux':
        os.system('cp {} {}'.format(para_pts_path, xyz_path))
    else:
        os.system('copy {} {}'.format(para_pts_path, xyz_path))
    
    

def gen_para_sample():
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    allfs = os.listdir(in_path)
    tasks = []
    for f in allfs:
        if f.endswith('.yml'):
            tasks.append(f)

    flag_parallel = False
    if not flag_parallel:
        for task in tasks:
            gen_one_para_sample(task)
        return
    
    with Pool(40) as p:
        p.map(gen_one_para_sample, tasks)

def gen_one_para_sample_repair(prefix, in_folder, out_folder):
    obj_normalize = os.path.join(in_folder, prefix + '_50k_fixtree.obj')
    fea_path =           obj_normalize.replace('.obj','.fea')    
    xyz_path =           os.path.join(out_folder, prefix + '_50k.xyz')
    mask_path =          os.path.join(out_folder, prefix + '_50k_mask.txt')

    para_pts_path = os.path.join(in_path, prefix + '_50k.xyz')
    para_ptsface_path = os.path.join(in_path, prefix + '_50k_tris.txt')
    
    os.system("{} -i {} -o {} -f {} -k {} -m 1 --ns 50000 --fs 0 -c 0 --csg 1 --convex 1 -r 0 --repairtree -p {} --pf {}".format(feature_sampling_path, obj_normalize, xyz_path, fea_path, mask_path, para_pts_path, para_ptsface_path))

    #copy points data sampled from points to target directory
    if sys_info == 'Linux':
        os.system('cp {} {}'.format(para_pts_path, xyz_path))
    else:
        os.system('copy {} {}'.format(para_pts_path, xyz_path))

def gen_para_sample_repair():
    out_path_repair = 'training_data_repair'
    failure_path = 'raw_input' #path that containing the *objtreefail files
    in_path_repair = 'training_data' #path that containing the *fixtree.obj/fea files
    
    if not os.path.exists(out_path_repair):
        os.mkdir(out_path_repair)

    allfs = os.listdir(failure_path)
    tasks = []
    for f in allfs:
        if f.endswith('.objtreefail'):
            f_split = f.split('.')[0].split('_')
            name = f_split[0] + '_' + f_split[1]
            tasks.append(name)

    for t in tasks:
        gen_one_para_sample_repair(t, in_path_repair, out_path_repair)

if __name__ == '__main__':
    if not args.r:
        gen_para_sample()
    else:
         gen_para_sample_repair()