import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
import trimesh
import pandas as pd
import csv
import math
import pickle

import argparse
# parse args first and set gpu id
parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='../../data/eval_data', help='ground truth data path')
parser.add_argument('--pred_path', type=str, default='../../data/output_data', help='converted data path')
parser.add_argument('--name_list', type=str, default='broken_bullet_name.txt', help='names of models to be evaluated, if you want to evaluate the whole dataset, please set it as all_names.txt')
parser.add_argument('--nsample', type=int, default=50000, help='point batch size')
parser.add_argument('--regen', default = False, action="store_true", help = 'regenerate feature curves')
args = parser.parse_args()

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    return dist, normals_dot_product

def distance_feature2mesh(points, mesh):
    prox = trimesh.proximity.ProximityQuery(mesh)
    signed_distance = prox.signed_distance(points)
    return np.abs(signed_distance)

def distance_p2mesh(points_src, normals_src, mesh):
    points_tgt, idx = mesh.sample(args.nsample, return_index=True)
    points_tgt = points_tgt.astype(np.float32)
    normals_tgt = mesh.face_normals[idx]
    cd1, nc1 = distance_p2p(points_src, normals_src, points_tgt, normals_tgt) #pred2gt
    hd1 = cd1.max()
    cd1 = cd1.mean()

    nc1 = np.clip(nc1, -1.0, 1.0)
    angles1 = np.arccos(nc1) / math.pi * 180.0
    angles1_mean = angles1.mean()
    angles1_std = np.std(angles1)

    cd2, nc2 = distance_p2p(points_tgt, normals_tgt, points_src, normals_src) #gt2pred
    hd2 = cd2.max()
    cd2 = cd2.mean()
    nc2 = np.clip(nc2, -1.0, 1.0)

    angles2 = np.arccos(nc2)/ math.pi * 180.0
    angles2_mean = angles2.mean()
    angles2_std = np.std(angles2)


    cd = 0.5 * (cd1 + cd2)
    hd = max(hd1, hd2)
    angles_mean = 0.5 * (angles1_mean + angles2_mean)
    angles_std = 0.5 * (angles1_std + angles2_std)
    return cd, hd, angles_mean, angles_std, hd1, hd2


def distance_fea(gt_pa, pred_pa):
    gt_points = gt_pa[:,:3]
    pred_points = pred_pa[:,:3]
    gt_angle = gt_pa[:,3]
    pred_angle = pred_pa[:,3]
    dfg2p = 0.0 
    dfp2g = 0.0 
    fag2p = 0.0 
    fap2g = 0.0
    pred_kdtree = cKDTree(pred_points)
    dist1, idx1 = pred_kdtree.query(gt_points)
    dfg2p = dist1.mean()
    assert(idx1.shape[0] == gt_points.shape[0])
    fag2p = np.abs(gt_angle - pred_angle[idx1])

    gt_kdtree = cKDTree(gt_points)
    dist2, idx2 = gt_kdtree.query(pred_points)
    dfp2g = dist2.mean()
    fap2g = np.abs(pred_angle - gt_angle[idx2])

    fag2p = fag2p.mean()
    fap2g = fap2g.mean()

    return dfg2p, dfp2g, fag2p, fap2g

def compute_all():
    gt_path = args.gt_path
    pred_mesh_path = args.pred_path
    namelst = args.name_list
    output_path = 'eval_results.csv'

    f = open(namelst, 'r')
    lines = f.readlines()
    f.close()

    d = {'name':[], 'CD':[], 'HD':[], 'HDgt2pred':[], 'HDpred2gt':[], 'AngleDiffMean':[], 'AngleDiffStd':[], 'FeaDfgt2pred':[], 'FeaDfpred2gt':[], 'FeaDf':[], 'FeaAnglegt2pred':[], 'FeaAnglepred2gt':[], 'FeaAngle':[]}

    for line in lines:
        line = line.strip()[:-4]
        print(line)
        test_xyz = os.path.join(gt_path, line+'_50k.xyz')
        ptnormal = np.loadtxt(test_xyz)
        meshfile = os.path.join(pred_mesh_path, '{}_50k.ply'.format(line))

        if not os.path.exists(meshfile):
            print('file not exists: ', meshfile)
            f = open(meshfile + 'noexists', 'w')
            f.close()
            continue
        stat_file = meshfile + "_stat"
        if not args.regen and os.path.exists(stat_file) and os.path.getsize(stat_file) > 0:
            #load compuated ones
            f = open(stat_file, 'rb')
            cur_dict = pickle.load(f)
            for k in cur_dict:
                d[k].append(cur_dict[k])
            f.close()
            continue

        d['name'].append(line)

        mesh = trimesh.load(meshfile)

        cd, hd, adm, ads, hd_pred2gt, hd_gt2pred = distance_p2mesh(ptnormal[:,:3], ptnormal[:,3:], mesh)

        d['CD'].append(cd)
        d['HD'].append(hd)
        d['HDpred2gt'].append(hd_pred2gt)
        d['HDgt2pred'].append(hd_gt2pred)
        d['AngleDiffMean'].append(adm)
        d['AngleDiffStd'].append(ads)

        gt_ptangle = np.loadtxt(os.path.join(gt_path, line + '_detectfea4e-3.ptangle'))
        pred_ptangle_path = meshfile[:-4]+'_4e-3.ptangle'
        if not os.path.exists(pred_ptangle_path) or args.regen:
            os.system('./MeshFeatureSample/build/SimpleSample -i {} -o {} -s 4e-3'.format(meshfile, pred_ptangle_path))
        pred_ptangle = np.loadtxt(pred_ptangle_path).reshape(-1,4)
        
        #for smooth case: if gt fea is empty, or pred fea is empty, then return 0
        if len(gt_ptangle) == 0 or len(pred_ptangle) == 0:
            d['FeaDfgt2pred'].append(0.0)
            d['FeaDfpred2gt'].append(0.0)
            d['FeaAnglegt2pred'].append(0.0)
            d['FeaAnglepred2gt'].append(0.0)
            d['FeaDf'].append(0.0)
            d['FeaAngle'].append(0.0)
        else:
            dfg2p, dfp2g, fag2p, fap2g = distance_fea(gt_ptangle, pred_ptangle)
            d['FeaDfgt2pred'].append(dfg2p)
            d['FeaDfpred2gt'].append(dfp2g)
            d['FeaAnglegt2pred'].append(fag2p)
            d['FeaAnglepred2gt'].append(fap2g)
            d['FeaDf'].append((dfg2p + dfp2g) / 2.0)
            d['FeaAngle'].append((fag2p + fap2g) / 2.0)

        cur_d = {}
        for k in d:
            cur_d[k] = d[k][-1]
        
        f = open(stat_file,"wb")
        pickle.dump(cur_d, f)
        f.close()
        

    d['name'].append('mean')
    for key in d:
        if key != 'name':
            d[key].append(sum(d[key])/len(d[key]))

    df = pd.DataFrame(d, columns=['name', 'CD', 'HD', 'HDpred2gt', 'HDgt2pred', 'AngleDiffMean', 'AngleDiffStd','FeaDfgt2pred', 'FeaDfpred2gt', 'FeaDf', 'FeaAnglegt2pred', 'FeaAnglepred2gt', 'FeaAngle'])

    df.to_csv(output_path, index = False, header=True)

if __name__ == '__main__':
    compute_all()