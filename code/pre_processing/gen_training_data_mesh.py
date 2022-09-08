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
    feature_sampling_path = './Bin/FeatureSample'
else:
    #windows version
    feature_sampling_path = r'.\Bin\FeatureSample.exe'


def gen_one_mesh_sample(objfile):
    obj_path = os.path.join(in_path, objfile)
    obj_normalize = obj_path.replace('.obj', '_normalized.obj')
    os.system("{} -i {} -o {} -m 0".format(feature_sampling_path, obj_path, obj_normalize))

    fea_path =           obj_path.replace('.obj','.fea')    
    xyz_path =           os.path.join(out_path, objfile.replace('.obj', '_50k.xyz'))
    mask_path =          os.path.join(out_path, objfile.replace('.obj', '_50k_mask.txt'))
    os.system("{} -i {} -o {} -f {} -k {} -m 1 --ns 50000 --fs 0 -c 1 --csg 1 --convex 1 -r 0 --mp 6 --repairtree".format(feature_sampling_path, obj_normalize, xyz_path, fea_path, mask_path))

def gen_mesh_sample():
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    allfs = os.listdir(in_path)
    tasks = []
    for f in allfs:
        if f.endswith('.obj') and not f.endswith('normalized.obj'):
            tasks.append(f)

    flag_parallel = False
    if not flag_parallel:
        for task in tasks:
            gen_one_mesh_sample(task)
        return
    
    with Pool(40) as p:
        p.map(gen_one_mesh_sample, tasks)


def gen_one_mesh_sample(objfile):
    obj_path = os.path.join(in_path, objfile)
    obj_normalize = obj_path.replace('.obj', '_normalized.obj')
    os.system("{} -i {} -o {} -m 0".format(feature_sampling_path, obj_path, obj_normalize))

    fea_path =           obj_path.replace('.obj','.fea')    
    xyz_path =           os.path.join(out_path, objfile.replace('.obj', '_50k.xyz'))
    mask_path =          os.path.join(out_path, objfile.replace('.obj', '_50k_mask.txt'))
    os.system("{} -i {} -o {} -f {} -k {} -m 1 --ns 50000 --fs 0 -c 1 --csg 1 --convex 1 -r 0 --mp 6 --repairtree".format(feature_sampling_path, obj_normalize, xyz_path, fea_path, mask_path))


def gen_one_mesh_sample_repair(prefix, in_folder, out_folder):
    obj_normalize = os.path.join(in_folder, prefix + '_50k_fixtree.obj')
    fea_path =           obj_normalize.replace('.obj','.fea')    
    xyz_path =           os.path.join(out_folder, prefix + '_50k.xyz')
    mask_path =          os.path.join(out_folder, prefix + '_50k_mask.txt')
    os.system("{} -i {} -o {} -f {} -k {} -m 1 --ns 50000 --fs 0 -c 0 --csg 1 --convex 1 -r 0 --repairtree".format(feature_sampling_path, obj_normalize, xyz_path, fea_path, mask_path))

def gen_mesh_sample_repair():
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
        gen_one_mesh_sample_repair(t, in_path_repair, out_path_repair)
    

if __name__ == '__main__':
    if not args.r:
        gen_mesh_sample()
    else:
        gen_mesh_sample_repair()