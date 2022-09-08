import os
from multiprocessing import Pool

data_path = 'raw_input'

feature_sampling_path = './Bin/FeatureSample'
para_sample_path = './Bin/ParametricSample'


def gen_one_para_sample(yamlfile):
    yaml_path = os.path.join(data_path, yamlfile)
    obj_path = os.path.join(data_path, yamlfile.replace('.yml', '.obj'))
    os.system("{} -y {} -o {}".format(para_sample_path, yaml_path, obj_path))

    fea_path =           obj_path.replace('.obj','.fea')
    xyz_path =           obj_path.replace('.obj', '_50k.xyz')
    mask_path =          obj_path.replace('.obj', '_50k_mask.txt')
    para_pts_path =      obj_path.replace('.obj', '_50k.xyz')
    para_ptsface_path =  obj_path.replace('.obj', '_50k_tris.txt')

    os.system("{} -i {} -o {} -f {} -k {} -m 1 --ns 50000 --fs 0 -c 1 --csg 1 --repairtree --convex 1 -r 0 --mp 6 -p {} --pf {}".format(feature_sampling_path, obj_path, xyz_path, fea_path, mask_path, para_pts_path, para_ptsface_path))


def gen_para_sample():
    allfs = os.listdir(data_path)
    tasks = []
    for f in allfs:
        if f.endswith('.yml'):
            tasks.append(f)

    flag_parallel = False
    if not flag_parallel:
        for task in tasks:
            gen_one_para_sample(task)
    
    with Pool(40) as p:
        p.map(gen_one_para_sample, tasks)

if __name__ == '__main__':
    gen_para_sample()