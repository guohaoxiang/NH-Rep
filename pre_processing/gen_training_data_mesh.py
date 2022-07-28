import os
from multiprocessing import Pool

data_path = 'raw_input'
feature_sampling_path = './Bin/FeatureSample'


def gen_one_mesh_sample(objfile):
    obj_path = os.path.join(data_path, objfile)
    obj_normalize = obj_path.replace('.obj', '_normalized.obj')
    os.system("{} -i {} -o {} -m 0".format(feature_sampling_path, obj_path, obj_normalize))

    fea_path =           obj_path.replace('.obj','.fea')
    xyz_path =           obj_path.replace('.obj', '_50k.xyz')
    mask_path =          obj_path.replace('.obj', '_50k_mask.txt')
    os.system("{} -i {} -o {} -f {} -k {} -m 1 --ns 50000 --fs 0 -c 1 --csg 1 --convex 1 -r 0 --mp 6 --repairtree".format(feature_sampling_path, obj_normalize, xyz_path, fea_path, mask_path))

def gen_mesh_sample():
    allfs = os.listdir(data_path)
    tasks = []
    for f in allfs:
        if f.endswith('.obj'):
            tasks.append(f)

    flag_parallel = False
    if not flag_parallel:
        for task in tasks:
            gen_one_mesh_sample(task)
    
    with Pool(40) as p:
        p.map(gen_one_mesh_sample, tasks)

if __name__ == '__main__':
    gen_mesh_sample()