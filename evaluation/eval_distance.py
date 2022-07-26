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
parser.add_argument('--nsample', type=int, default=50000, help='point batch size') #from 16384 to 8192
parser.add_argument('--ed', default = True, action="store_false", help = 'evaluating edge')
parser.add_argument('-t', type=str, default='other')
parser.add_argument('-s', type=str, default='', help = 'suffix')
parser.add_argument('--gather', default = False, action="store_true", help = 'gather results')
parser.add_argument('--small', default = False, action="store_true", help = 'small dataset(200)')

parser.add_argument('--subtable', default = False, action="store_true", help = 'get sub table')
parser.add_argument('--compone', default = False, action="store_true", help = 'get one component')
parser.add_argument('--regen', default = False, action="store_true", help = 'regenerate feature curves')


args = parser.parse_args()

def save_ply_data_numpy(filename, array):
    f = open(filename, 'w')
    if array.shape[1] == 10:  
        f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float opacity\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'.format(array.shape[0]))
        for i in range(array.shape[0]):
            for j in range(6):
                f.write("{:f} ".format(array[i][j]))
            for j in range(3):
                f.write("{:d} ".format(int(array[i][j+6])))
            f.write('{:f}\n'.format(array[i][9]))
    elif array.shape[1] == 7:
        f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float opacity\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'.format(array.shape[0]))
        for i in range(array.shape[0]):
            for j in range(3):
                f.write("{:f} ".format(array[i][j]))
            for j in range(3):
                f.write("{:d} ".format(int(array[i][j+3])))
            f.write('{:f}\n'.format(array[i][6]))
    f.close()

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

def test_distance_p2p():
    pts_f1 = '/mnt/sdf1/haog/code/IGR/CADdata/00000112_50k_color.xyz'
    pts_f2 = '/mnt/sdf1/haog/code/IGR/CADdata/00000112_50k_color.xyz'

    # pts_f2 = '/mnt/sdf1/haog/code/IGR/CADdata_test/00000112_50k_test.xyz'
    ptnormal1 = np.loadtxt(pts_f1)
    ptnormal2 = np.loadtxt(pts_f2)
    cd1, nd1 = distance_p2p(ptnormal1[:,:3], ptnormal1[:,3:], ptnormal2[:,:3], ptnormal2[:,3:])
    cd2, nd2 = distance_p2p(ptnormal2[:,:3], ptnormal2[:,3:], ptnormal1[:,:3], ptnormal1[:,3:])
    print ("cd1: {} cd2: {} nd1: {} nd2: {}".format(cd1.mean(), cd2.mean(), nd1.mean(), nd2.mean()))
    print ('haussdorff1: {} haussdorff2: {}'.format(cd1.max(), cd2.max()))
    # print ('direct haussdorff1: {} haussdorff2: {}'.format(directed_hausdorff(ptnormal1[:,:3], ptnormal2[:,:3]), directed_hausdorff(ptnormal2[:,:3], ptnormal1[:,:3])))

def test_mesh_samples():
    pts_f1 = '/mnt/sdf1/haog/code/IGR/CADdata/00000112_50k_color.xyz'
    mesh_f= '/mnt/sdf1/haog/code/IGR/CADdata/00000112_normalize.obj'
    mesh = trimesh.load(mesh_f)

    pc, idx = mesh.sample(args.nsample, return_index=True) #uniform sampling here
    pc = pc.astype(np.float32)
    normals = mesh.face_normals[idx]
    np.savetxt('sample.xyz', np.concatenate((pc, normals), 1))

def test_feature2mesh():
    pts_f1 = '/mnt/sdf1/haog/code/IGR/CADdata_test/00000112_10k_fea.xyz'
    # mesh_f= '/mnt/sdf1/haog/code/IGR/CADdata/00000112_normalize.obj'
    mesh_f= '/mnt/sdf1/haog/code/generated_mesh/Poisson_50k/00000112_50k_color_pw50.obj'

    ptnormal = np.loadtxt(pts_f1)
    mesh = trimesh.load(mesh_f)
    dist = distance_feature2mesh(ptnormal[:,:3], mesh)
    print ("feature avg: {} max: {}".format(dist.mean(), dist.max()))


def distance_p2mesh(points_src, normals_src, mesh):
    #return chamferL2, haussdorf L2,chamfer L1, haussdorf L1, normal consistency
    points_tgt, idx = mesh.sample(args.nsample, return_index=True)
    points_tgt = points_tgt.astype(np.float32)
    normals_tgt = mesh.face_normals[idx]
    cd1, nc1 = distance_p2p(points_src, normals_src, points_tgt, normals_tgt) #pred2gt
    # cd1_L1 = np.sqrt(cd1)
    hd1 = cd1.max()
    # hd1_L1 = cd1_L1.max()
    cd1 = cd1.mean()
    # cd1_L1 = cd1_L1.mean()

    nc1 = np.clip(nc1, -1.0, 1.0)
    angles1 = np.arccos(nc1) / math.pi * 180.0
    angles1_mean = angles1.mean()
    angles1_std = np.std(angles1)

    cd2, nc2 = distance_p2p(points_tgt, normals_tgt, points_src, normals_src) #gt2pred
    # cd2_L1 = np.sqrt(cd2)
    hd2 = cd2.max()
    # hd2_L1 = cd2_L1.max()
    cd2 = cd2.mean()
    # cd2_L1 = cd2_L1.mean()
    # print('nc2: ', nc2)
    nc2 = np.clip(nc2, -1.0, 1.0)

    angles2 = np.arccos(nc2)/ math.pi * 180.0
    angles2_mean = angles2.mean()
    angles2_std = np.std(angles2)


    cd = 0.5 * (cd1 + cd2)
    # cd_L1 = 0.5 * (cd1_L1 + cd2_L1)
    hd = max(hd1, hd2)
    # hd_L1 = max(hd1_L1, hd2_L1)
    # nc = 0.5 * (nc1.mean() + nc2.mean())
    angles_mean = 0.5 * (angles1_mean + angles2_mean)
    angles_std = 0.5 * (angles1_std + angles2_std)
    # return cd, hd, angles_mean, angles_std
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

    # #visualize and testing fag2p, fap2g
    # print ('fag2p min: {} max: {}'.format(fag2p.min(), fag2p.max()))
    # print('fap2g min: {} max: {}'.format(fap2g.min(), fap2g.max()))
    # fag2p_n = (fag2p - fag2p.min())/(fag2p.max() - fag2p.min())
    # fap2g_n = (fap2g - fap2g.min())/(fap2g.max() - fap2g.min())

    # print('g2p size: ', fag2p_n.shape)
    # fag2p_n = np.expand_dims(fag2p_n, 1)
    # fap2g_n = np.expand_dims(fap2g_n, 1)

    # g2pcolor = fag2p_n * np.array([255, 0, 0]) + (1 - fag2p_n) * np.array([255, 255, 255])
    # p2gcolor = fap2g_n * np.array([255, 0, 0]) + (1 - fap2g_n) * np.array([255, 255, 255])

    # g2pdata = np.concatenate([gt_points, g2pcolor, np.ones([gt_points.shape[0],1])],1)
    # p2gdata = np.concatenate([pred_points, p2gcolor, np.ones([pred_points.shape[0],1])],1)
    # save_ply_data_numpy('g2p.ply', g2pdata)
    # save_ply_data_numpy('p2g.ply', p2gdata)
    # #visualize end

    fag2p = fag2p.mean()
    fap2g = fap2g.mean()

    return dfg2p, dfp2g, fag2p, fap2g


def compute_all():
    # namelst = '/mnt/sdf1/haog/code/siren_ori/siren/all_models.txt'
    # namelst = '107models.txt'
    # namelst = '/mnt/sdf1/haog/code/siren_ori/siren/5models_tmp.txt'

    # namelst = '10models.txt'

    # namelst = 'ablation20.txt'
    namelst = 'split_ablation.txt'


    gt_xyz_path = '/mnt/sdf1/haog/code/IGR/CADdata_test'

    gt_ptangle_path = '/mnt/sdf1/haog/code/IGR/CADdata'

    # mesh_path = '/mnt/sdf1/haog/code/IGR/Poisson_50k'
    # mesh_path = '/mnt/sdf1/haog/code/generated_mesh/IGR'
    # mesh_path = '/mnt/sdf1/haog/code/generated_mesh/baseline4'
    # mesh_path = '/mnt/sdf1/haog/code/generated_mesh/siren'
    if args.t == 'ours':
        mesh_path = '/mnt/sdf1/haog/code/generated_mesh/linearassign'
        output_path = 'linearassign_fea_eval{}_new.csv'.format(args.s)
    if args.t == 'nocolor':
        mesh_path = '/mnt/sdf1/haog/code/generated_mesh/nolimit'
        output_path = 'nocolor_fea_eval{}.csv'.format(args.s)
    elif args.t == 'siren':
        mesh_path = '/mnt/sdf1/haog/code/generated_mesh/siren'
        output_path = 'siren_fea_eval{}_new.csv'.format(args.s)
    elif args.t == 'igr':
        mesh_path = '/mnt/sdf1/haog/code/generated_mesh/IGR'
        output_path = 'igr_fea_eval{}_new.csv'.format(args.s)
    elif args.t == 'bl':    
        mesh_path = '/mnt/sdf1/haog/code/generated_mesh/baseline4'
        output_path = 'baseline4_fea_eval{}_new617.csv'.format(args.s)
    elif args.t == 'poisson':       
        mesh_path = '/mnt/sdf1/haog/code/generated_mesh/Poisson_50k'
        output_path = 'poisson_fea_eval_new.csv'
        # output_path = 'poisson_fea_eval.csv'
    elif args.t == '227':           
        mesh_path = '/mnt/sdf1/haog/code/generated_mesh/ours0227'
        output_path = '227_fea_eval_nooct.csv'
    elif args.t == 'rimls':           
        mesh_path = '/mnt/sdf1/haog/code/generated_mesh/RIMLS'
        output_path = 'RIMLS_fea_eval_nooct.csv'
    elif args.t == 'other':
        #other mesh
        # mesh_path = '/mnt/sdf1/haog/code/generated_mesh/ablation'
        # output_path = 'ablation_nonormal_318.csv'
        # mesh_path = '/mnt/sdf1/haog/code/generated_mesh/256x3'
        # mesh_path = '/mnt/sdf1/haog/code/generated_mesh/IGRsirenloss'
        # output_path = 'igrloss_all.csv'
        # output_path = 'sirenloss_all.csv'

        # mesh_path = '/mnt/data/haog/code/generate_mesh/nonormal_0110'
        mesh_path = '/mnt/data/haog/code/generate_mesh/shapenet_split/split'

        # output_path = 'ablation_nonormal_0110.csv'
        output_path = 'split_ablation.csv'





    # output_path = 'baseline4_eval.csv'
    # output_path = 'siren_eval.csv'
    


    f = open(namelst, 'r')
    lines = f.readlines()
    f.close()

    # d = {'name':[], 'CD':[], 'CDL1':[], 'HD':[], 'HDL1':[], 'NC':[]}
    # d = {'name':[], 'CD':[], 'HD':[], 'NC':[], 'ED_mean':[], 'ED_max':[]}
    d = {'name':[], 'CD':[], 'HD':[], 'AngleDiffMean':[], 'AngleDiffStd':[], 'FeaDfgt2pred':[], 'FeaDfpred2gt':[], 'FeaDf':[], 'FeaAnglegt2pred':[], 'FeaAnglepred2gt':[], 'FeaAngle':[]}



    for line in lines:
        line = line.strip()[:-1]
        print(line)
        d['name'].append(line[:-6])
        test_xyz = os.path.join(gt_xyz_path, line[:-5]+'test.xyz')
        ptnormal = np.loadtxt(test_xyz)
        fea_xyz = os.path.join(gt_xyz_path, line[:-9]+'1k_fea.xyz')
        fea_ptnormal = np.loadtxt(fea_xyz)

        # meshfile = os.path.join(mesh_path, 'igr_{}.ply'.format(line))
        # meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_baseline_{}.ply'.format(line))
        if args.t == 'ours':
            meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_linassign_{}{}.ply'.format(line, args.s))
        elif args.t == 'nolimit':
            meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_linassign_nl_{}{}.ply'.format(line, args.s))
        elif args.t == 'siren':
            meshfile = os.path.join(mesh_path, '{}{}.ply'.format(line, args.s))         
        elif args.t == 'igr':
            meshfile = os.path.join(mesh_path, 'igr_{}{}.ply'.format(line, args.s))
        elif args.t == 'bl':    
            meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_baseline_{}{}.ply'.format(line, args.s))
        elif args.t == 'poisson':
            meshfile = os.path.join(mesh_path, '{}_pw50.obj'.format(line))
            # meshfile = os.path.join(mesh_path, '{}.obj'.format(line))

        elif args.t == '227':
            meshfile_ori = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_abc_{}.ply'.format(line))
            meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_{}.ply'.format(line))
            if os.path.exists(meshfile_ori):
                os.system('mv {} {}'.format(meshfile_ori, meshfile))
        elif args.t == 'rimls':
            meshfile = os.path.join(mesh_path, '{}_rimls.obj'.format(line))
        elif args.t == 'other':
            # meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_50k_linear_512x7{}_nooct.ply'.format(line))
            # meshfile = os.path.join(mesh_path, 'ablation_nonormal_{}_nooct.ply'.format(line))
            # meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_50k_linear_noise_wde-4_ns_{}_nooct.ply'.format(line))
            # meshfile = os.path.join(mesh_path, '{}{}.ply'.format(line, '_ps_nooct'))
            # meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_linassign_256x3_{}{}.ply'.format(line, '_nooct'))
            # meshfile = os.path.join(mesh_path, 'baseline_igrloss_{}{}.ply'.format(line, '_nooct'))

            # meshfile = os.path.join(mesh_path, 'baseline_igrloss_{}{}.ply'.format(line, '_nooct'))

            # meshfile = os.path.join(mesh_path, 'baseline_sirenloss_{}{}.ply'.format(line, '_nooct'))
            # meshfile = os.path.join(mesh_path, 'ablation_nonormal_{}.ply'.format(line))
            meshfile = os.path.join(mesh_path, 'split_ablation_{}.ply'.format(line))






        mesh = trimesh.load(meshfile)

        cd, hd, adm, ads, hd_pred2gt, hd_gt2pred = distance_p2mesh(ptnormal[:,:3], ptnormal[:,3:], mesh)
        # if args.ed:
        #   fea_dist = distance_feature2mesh(fea_ptnormal[:,:3],mesh)
        #   d['ED_mean'].append(fea_dist.mean())
        #   d['ED_max'].append(fea_dist.max())
        # else:
        #   d['ED_mean'].append(0.0)
        #   d['ED_max'].append(0.0)

        # d['CD'].append(cd)
        # d['CDL1'].append(cd1)
        # d['HD'].append(hd)
        # d['HDL1'].append(hd1)
        # d['NC'].append(nc)

        d['CD'].append(cd)
        # d['CDL1'].append(cd1)
        d['HD'].append(hd)
        # d['HD'].append(hd)
        # d['HDL1'].append(hd1)
        # d['NC'].append(nc)
        d['AngleDiffMean'].append(adm)
        d['AngleDiffStd'].append(ads)

        if not args.ed:
            d['FeaDfgt2pred'].append(0.0)
            d['FeaDfpred2gt'].append(0.0)
            d['FeaAnglegt2pred'].append(0.0)
            d['FeaAnglepred2gt'].append(0.0)
            d['FeaDf'].append(0.0)
            d['FeaAngle'].append(0.0)
        else:
            gt_ptangle = np.loadtxt(os.path.join(gt_ptangle_path, line[:-9] + 'feasample4e-3.ptangle'))
            pred_ptangle_path = meshfile[:-4]+'_4e-3.ptangle'
            if not os.path.exists(pred_ptangle_path) or args.regen:
                os.system('~/SimpleSample -i {} -o {} -s 4e-3'.format(meshfile, pred_ptangle_path))
            pred_ptangle = np.loadtxt(pred_ptangle_path)
            dfg2p, dfp2g, fag2p, fap2g = distance_fea(gt_ptangle, pred_ptangle)
            d['FeaDfgt2pred'].append(dfg2p)
            d['FeaDfpred2gt'].append(dfp2g)
            d['FeaAnglegt2pred'].append(fag2p)
            d['FeaAnglepred2gt'].append(fap2g)
            d['FeaDf'].append((dfg2p + dfp2g) / 2.0)
            d['FeaAngle'].append((fag2p + fap2g) / 2.0)

        #edge part

    d['name'].append('mean')
    for key in d:
        if key != 'name':
            d[key].append(sum(d[key])/len(lines))

    # d['name'].append('mean')
    # d['CD'].append(sum(d['CD'])/len(lines))
    # # d['CDL1'].append(sum(d['CDL1'])/len(lines))
    # d['HD'].append(sum(d['HD'])/len(lines))
    # # d['HDL1'].append(sum(d['HDL1'])/len(lines))
    # d['NC'].append(sum(d['NC'])/len(lines))
    # d['ED_mean'].append(sum(d['ED_mean'])/len(lines))
    # d['ED_max'].append(sum(d['ED_max'])/len(lines))


    # df = pd.DataFrame(d, columns=['name', 'CD', 'CDL1', 'HD', 'HDL1', 'NC'])
    # print(d)
    # print("length of d: ", len(d))
    # print("length of d: ", d.keys())
    # for k in d.keys():
    #     print('length of {} : {}'.format(k, len(d[k])))

    df = pd.DataFrame(d, columns=['name', 'CD', 'HD', 'AngleDiffMean', 'AngleDiffStd','FeaDfgt2pred', 'FeaDfpred2gt', 'FeaDf', 'FeaAnglegt2pred', 'FeaAnglepred2gt', 'FeaAngle'])
    df.to_csv(output_path, index = False, header=True)

def compute_all_1210():
    # namelst = '/mnt/sdf1/haog/code/siren_ori/siren/all_models.txt'
    # namelst = '/mnt/sdf1/haog/data/implicitcad/test_5name.txt'
    # namelst = '/mnt/sdf1/haog/data/implicitcad/1201_200_names.txt'
    # namelst = '/mnt/sdf1/haog/data/implicitcad/1211_200_names.txt'

    # namelst = '/mnt/sdf1/haog/data/implicitcad/1211_200_names_igr.txt'
    # namelst = '/mnt/sdf1/haog/code/siren_ori/siren/5models_tmp.txt'

    # namelst = '10models.txt'

    # namelst = 'ablation20.txt'

    # namelst = '/mnt/sdf1/haog/code/ImplicitCAD_script/fns/200/models_parametric_200_nocolor-cc.txt'
    # gt_xyz_path = '/mnt/data/haog/data/data_0000_200_output_test'
    # gt_ptangle_path = '/mnt/data/haog/data/data_0000_200_output_test'

    #all mesh
    # namelst = 'tmp_fns/all_mesh_color6_part1.txt'
    # namelst = 'tmp_fns/all_para_color6.txt'
    # namelst = 'tmp_fns/all_para_nocolor.txt'


    # gt_xyz_path = '/mnt/data/haog/data/data_0000_200_output_test'
    # gt_ptangle_path = '/mnt/data/haog/data/data_0000_200_output_test'



    #all version
    if args.t == 'ours':
        # mesh_path = '/mnt/data/haog/code/generate_mesh/1206_aml_200_our_color6'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/0101_our_200_color6_para'

        gt_ptangle_path = '/mnt/data/haog/code/gt/mesh_fea_from200'
        # gt_xyz_path = '/mnt/data/haog/code/gt/mesh_pts_from200'
        gt_xyz_path = '/mnt/data/haog/code/gt/para_pts_from200'

        mesh_path = '/mnt/data/haog/code/generate_mesh/0101_our_all_para_color6_mesh'
        namelst = 'tmp_fns/all_para_color6.txt'


        # output_path = 'our_200_color6.csv'
        # output_path = 'our_200_color6_1212.csv'
        # output_path = 'our_para_200_color6_0107.csv'
        # output_path = 'our_mesh_color6_part1.csv'
        output_path = 'all_our_color6_para.csv'



    if args.t == 'nocolor':
        # mesh_path = '/mnt/data/haog/code/generate_mesh/1206_aml_200_our_nocolor'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/0101_our_200_nocolor_para'

        gt_ptangle_path = '/mnt/data/haog/code/gt/mesh_fea_from200'
        # gt_xyz_path = '/mnt/data/haog/code/gt/mesh_pts_from200'
        gt_xyz_path = '/mnt/data/haog/code/gt/para_pts_from200'
        namelst = 'tmp_fns/all_para_nocolor.txt'
        mesh_path = '/mnt/data/haog/code/generate_mesh/0101_our_all_para_nocolor_mesh'


        # output_path = 'our_200_nocolor.csv'
        # output_path = 'our_200_nocolor_1212.csv'
        # output_path = 'our_para_200_nocolor_0107.csv'
        output_path = 'all_our_nocolor_para.csv'

    elif args.t == 'siren':

        #large version
        gt_ptangle_path = '/mnt/data/haog/code/gt/mesh_fea_from200'
        gt_xyz_path = '/mnt/data/haog/code/gt/mesh_pts_from200'
        
        #para version
        # gt_xyz_path = '/mnt/data/haog/code/gt/para_pts_from200'
        # namelst = 'tmp_fns/all_para_siren.txt'
        # namelst = 'tmp_fns/all_para_siren_part2.txt'

        namelst = 'tmp_fns/ablation_our_valid_100.txt'
        # mesh_path = '/mnt/sdf1/haog/generated_mesh_sdf1/0101_siren_all_para_mesh'

        mesh_path = '/mnt/data/haog/code/generate_mesh/siren_fs'

        # mesh_path = '/mnt/data/haog/code/generate_mesh/1206_aml_200_siren'
        
        # output_path = 'all_siren_para.csv'
        # output_path = 'all_siren_para_part2.csv'
        output_path = 'siren_fs.csv'


        # mesh_path = '/mnt/data/haog/code/generate_mesh/1206_aml_200_siren'
        # output_path = '1210_siren_200.csv'

        if args.small:
            namelst = 'tmp_fns/200_para_siren.txt'
            mesh_path = '/mnt/data/haog/code/generate_mesh/0101_siren_200_para'

            # mesh_path = '/mnt/data/haog/code/generate_mesh/1206_aml_200_siren'
            
            output_path = '200_siren_para.csv'

    elif args.t == 'igr':
        # mesh_path = '/mnt/data/haog/code/generate_mesh/1206_aml_igr_200'
        # output_path = '1210_igr_200.csv'

        gt_ptangle_path = '/mnt/data/haog/code/gt/mesh_fea_from200'
        # gt_xyz_path = '/mnt/data/haog/code/gt/mesh_pts_from200'
        gt_xyz_path = '/mnt/data/haog/code/gt/para_pts_from200'
        namelst = 'tmp_fns/all_para_igr.txt'
        mesh_path = '/mnt/sdf1/haog/generated_mesh_sdf1/0101_igr_all_para_mesh'

        output_path = 'all_igr_para.csv'

        if args.small:
            namelst = 'tmp_fns/200_para_igr.txt'
            mesh_path = '/mnt/data/haog/code/generate_mesh/0101_igr_200_para'

            output_path = '200_igr_para.csv'

        # mesh_path = '/mnt/data/haog/code/generate_mesh/0101_igr_200_para'
        # output_path = '0101_igr_200_para.csv'
    # elif args.t == 'bl':    
    #     mesh_path = '/mnt/sdf1/haog/code/generated_mesh/baseline4'
    #     output_path = 'baseline4_fea_eval{}_new617.csv'.format(args.s)
    elif args.t == 'poisson':       
        # mesh_path = '/mnt/data/haog/code/generate_mesh/data_0000_200_output_1201_poisson_rimls'
        # output_path = '1210_poisson_200.csv'

        
        # mesh_path = '/mnt/data/haog/code/generate_mesh/200_parametric_train_pts_poisson_rimls'
        # output_path = '0101_poisson_200_para.csv'
        
        gt_ptangle_path = '/mnt/data/haog/code/gt/mesh_fea_from200'
        # gt_xyz_path = '/mnt/data/haog/code/gt/mesh_pts_from200'
        gt_xyz_path = '/mnt/data/haog/code/gt/para_pts_from200'
        namelst = '/mnt/sdf1/haog/code/ImplicitCAD_script/fns/all/models_parametric_all_nocolor-cc-sp-200_0107.txt'
        # namelst = 'tmp_fns/tmp.txt'

        mesh_path = '/mnt/data/haog/code/generate_mesh/all_parametric_poisson_rimls'
        output_path = 'all_poisson_para.csv'


        # mesh_path = '/mnt/data/haog/code/generate_mesh/200_parametric_train_pts_poisson_rimls'
        # output_path = '0101_poisson_200_para.csv'


        if args.small:
            namelst = '/mnt/sdf1/haog/code/ImplicitCAD_script/fns/200/models_parametric_200_nocolor-cc-sp-200.txt'
            mesh_path = '/mnt/data/haog/code/generate_mesh/200_parametric_train_pts_poisson_rimls'
            output_path = '200_poisson_para.csv'
        


    elif args.t == 'rimls':       
        # mesh_path = '/mnt/data/haog/code/generate_mesh/200_parametric_train_pts_poisson_rimls'
        # output_path = '0101_rimls_200_para.csv'

        gt_ptangle_path = '/mnt/data/haog/code/gt/mesh_fea_from200'
        # gt_xyz_path = '/mnt/data/haog/code/gt/mesh_pts_from200'
        gt_xyz_path = '/mnt/data/haog/code/gt/para_pts_from200'
        namelst = '/mnt/sdf1/haog/code/ImplicitCAD_script/fns/all/models_parametric_all_nocolor-cc-sp-200_0107.txt'
        mesh_path = '/mnt/data/haog/code/generate_mesh/all_parametric_poisson_rimls'
        output_path = 'all_rimls_para.csv'
        if args.small:
            namelst = '/mnt/sdf1/haog/code/ImplicitCAD_script/fns/200/models_parametric_200_nocolor-cc-sp-200.txt'
            # namelst = 'tmp_fns/tmp.txt'

            mesh_path = '/mnt/data/haog/code/generate_mesh/200_parametric_train_pts_poisson_rimls'
            output_path = '200_rimls_para.csv'
            # output_path = '200_rimls_para_test.csv'
    #     # output_path = 'poisson_fea_eval.csv'
    # elif args.t == '227':           
    #     mesh_path = '/mnt/sdf1/haog/code/generated_mesh/ours0227'
    #     output_path = '227_fea_eval_nooct.csv'
    # elif args.t == 'rimls':           
    #     mesh_path = '/mnt/sdf1/haog/code/generated_mesh/RIMLS'
    #     output_path = 'RIMLS_fea_eval_nooct.csv'
    elif args.t == 'other':
        #other mesh, for ablation
        
        gt_ptangle_path = '/mnt/data/haog/code/gt/mesh_fea_from200'
        gt_xyz_path = '/mnt/data/haog/code/gt/mesh_pts_from200'
        # gt_xyz_path = '/mnt/data/haog/code/gt/para_pts_from200'
        # namelst = '/mnt/sdf1/haog/code/ImplicitCAD_script/fns/all/ablation100.txt'

        # namelst = 'tmp_fns/all_valid_norimls_append200.txt'
        # namelst = 'tmp_fns/all_valid_norimls_append200_from6000.txt'
        # namelst = 'tmp_fns/ablation_our_valid_100.txt'
        # namelst = 'tmp_fns/ablation_our_valid_100.txt'
        namelst = 'tmp_fns/115.txt'






        # namelst = '/mnt/sdf1/haog/code/ImplicitCAD_script/fns/all/tmp.txt'
        # namelst = 'tmp_fns/tmp.txt'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/ablation_offsurface'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/ablation_64x3'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/ablation_bl'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/0217_ab_igr_fs'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/0308_ab_cons'

        # mesh_path = '/mnt/data/haog/code/generate_mesh/neural_splines_fitone_256'

        # mesh_path = '/mnt/data/haog/code/generate_mesh/neural_spline_grid512'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/ns_output_512_8'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/ablation_correction'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/0101_our_all_color6_mesh/0101_our_all_color6_mesh'
        # mesh_path = '/mnt/data/haog/code/generate_mesh/all_mesh_color6_part2/part2' #second part of colored mesh
        # mesh_path = '/mnt/data/haog/code/generate_mesh/0308_ab_cons_corres' #second part of colored mesh

        # mesh_path = '/mnt/data/haog/code/generate_mesh/115_poisson' #second part of colored mesh
        # mesh_path = '/mnt/data/haog/code/generate_mesh/115_poisson_5000k' #second part of colored mesh
        # mesh_path = '/mnt/data/haog/code/generate_mesh/115_ns_5000k' #second part of colored mesh
        # mesh_path = '/mnt/data/haog/code/generate_mesh/115_poisson_depth10' #second part of colored mesh
        mesh_path = '/mnt/data/haog/code/generate_mesh/115_ns_depth10' #second part of colored mesh





        




        # mesh_path = '/mnt/data/haog/code/generate_mesh/all_mesh_color6_part2/part2'

        # output_path = 'ablation100_normal.csv'
        # output_path = 'ablation100_offsurface.csv'
        # output_path = 'ourmesh_part2.csv'
        # output_path = 'ourmesh_test.csv'
        
        # output_path = 'ablation128x3.csv'
        # output_path = 'ablationbl.csv'
        # output_path = 'ablation_igr_with_feature.csv'

        # output_path = 'ablation_0308_cons.csv'
        # output_path = 'neural_spline_one_256.csv'

        # output_path = 'neural_spline_grid_512.csv'

        # output_path = 'neural_spline_grid_512_all.csv'
        # output_path = 'neural_spline_grid_512_all_from6000.csv'
        # output_path = 'neural_spline_grid_512_all_from6000.csv'

        # output_path = 'tmp.csv'
        # output_path = 'ablation100_correction_hdtwoway.csv'

        # output_path = 'ablation100_our_hdtwoway200.csv'
        # output_path = 'ablation100_our_hdtwoway200restpart2.csv'


        # output_path = 'ablation100_corres_cor.csv'
        # output_path = '115_ns.csv'
        # output_path = '115_poisson.csv'
        # output_path = '115_poisson_5000k.csv'
        # output_path = '115_ns_5000k.csv'
        # output_path = '115_poisson_depth10.csv'

        output_path = '115_ns_depth10.csv'


        # output_path = '115_ns_5000k.csv'
















        #test:
        


        # output_path = 'sirenloss_all.csv'




    # output_path = 'baseline4_eval.csv'
    # output_path = 'siren_eval.csv'
    


    f = open(namelst, 'r')
    lines = f.readlines()
    f.close()

    # d = {'name':[], 'CD':[], 'CDL1':[], 'HD':[], 'HDL1':[], 'NC':[]}
    # d = {'name':[], 'CD':[], 'HD':[], 'NC':[], 'ED_mean':[], 'ED_max':[]}
    # d = {'name':[], 'CD':[], 'HD':[], 'AngleDiffMean':[], 'AngleDiffStd':[], 'FeaDfgt2pred':[], 'FeaDfpred2gt':[], 'FeaDf':[], 'FeaAnglegt2pred':[], 'FeaAnglepred2gt':[], 'FeaAngle':[]}

    d = {'name':[], 'CD':[], 'HD':[], 'HDgt2pred':[], 'HDpred2gt':[], 'AngleDiffMean':[], 'AngleDiffStd':[], 'FeaDfgt2pred':[], 'FeaDfpred2gt':[], 'FeaDf':[], 'FeaAnglegt2pred':[], 'FeaAnglepred2gt':[], 'FeaAngle':[]}




    for line in lines:
        line = line.strip()[:-4]
        print(line)
        test_xyz = os.path.join(gt_xyz_path, line+'_50k.xyz')
        ptnormal = np.loadtxt(test_xyz)

        # meshfile = os.path.join(mesh_path, 'igr_{}.ply'.format(line))
        # meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_baseline_{}.ply'.format(line))
        if args.t == 'ours':
            # meshfile = os.path.join(mesh_path, '1206_aml_200_256x3_color6_{}_50k.ply'.format(line))

            # meshfile = os.path.join(mesh_path, '0101_our_200_color6_para_{}_50k.ply'.format(line))

            meshfile = os.path.join(mesh_path, '0101_our_all_para_color6_{}_50k.ply'.format(line))


        elif args.t == 'nocolor':
            # meshfile = os.path.join(mesh_path, '1203_aml_200_256x3_nocolor_{}_50k.ply'.format(line))

            # meshfile = os.path.join(mesh_path, '0101_our_200_nocolor_para_{}_50k.ply'.format(line))

            meshfile = os.path.join(mesh_path, '0101_our_all_para_nocolor_{}_50k.ply'.format(line))

            
            
        elif args.t == 'siren':
            meshfile = os.path.join(mesh_path, '{}_50k.ply'.format(line))         
        elif args.t == 'igr':
            # meshfile = os.path.join(mesh_path, '1206_aml_200_igr_{}_50k.ply'.format(line))         
            # meshfile = os.path.join(mesh_path, '0101_igr_200_para_{}_50k.ply'.format(line)) 
            meshfile = os.path.join(mesh_path, '0101_igr_all_para_{}_50k.ply'.format(line))
            if args.small:
                meshfile = os.path.join(mesh_path, '0101_igr_200_para_{}_50k.ply'.format(line)) 

        # elif args.t == 'bl':    
        #     meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_baseline_{}{}.ply'.format(line, args.s))
        elif args.t == 'poisson':
            meshfile = os.path.join(mesh_path, '{}_50k_poisson.obj'.format(line))
            # meshfile = os.path.join(mesh_path, '{}.obj'.format(line))
        elif args.t == 'rimls':
            # meshfile = os.path.join(mesh_path, '{}_50k_rimls.ply'.format(line))
            meshfile = os.path.join(mesh_path, '{}_50k_rimls.obj'.format(line))
            if args.small:
                meshfile = os.path.join(mesh_path, '{}_50k_rimls.ply'.format(line))

            # meshfile = os.path.join(mesh_path, '{}.obj'.format(line))

        # elif args.t == '227':
        #     meshfile_ori = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_abc_{}.ply'.format(line))
        #     meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_{}.ply'.format(line))
        #     if os.path.exists(meshfile_ori):
        #         os.system('mv {} {}'.format(meshfile_ori, meshfile))
            # meshfile = os.path.join(mesh_path, 'baseline_sirenloss_{}{}.ply'.format(line, '_nooct'))

        # elif args.t == 'rimls':
        #     meshfile = os.path.join(mesh_path, '{}_rimls.obj'.format(line))
        elif args.t == 'other':
            # meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_50k_linear_512x7{}_nooct.ply'.format(line))
            # meshfile = os.path.join(mesh_path, 'ablation_nonormal_{}_nooct.ply'.format(line))
            # meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_50k_linear_noise_wde-4_ns_{}_nooct.ply'.format(line))
            # meshfile = os.path.join(mesh_path, '{}{}.ply'.format(line, '_ps_nooct'))
            # meshfile = os.path.join(mesh_path, 'nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_linassign_256x3_{}{}.ply'.format(line, '_nooct'))
            # meshfile = os.path.join(mesh_path, 'ablation_normal_{}{}.ply'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, 'ablation_offsurface_{}{}.ply'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, 'ablation_256x7_{}{}.ply'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, 'ablation_64x3_{}{}.ply'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, 'ablation_baseline_{}{}.ply'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, '0217_ab_our_fs_color6_{}{}.ply'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, '0217_ab_igr_fs_{}{}.ply'.format(line, '_50k'))

            # meshfile = os.path.join(mesh_path, '0308_ab_cons_{}{}.ply'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, '{}{}.ply'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, 'ablation_correction_{}{}.ply'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, '0101_our_all_color6_{}{}.ply'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, '0308_ab_cons_corres_{}{}.ply'.format(line, '_50k'))

            # meshfile = os.path.join(mesh_path, '{}{}_poisson.obj'.format(line, '_50k'))
            # meshfile = os.path.join(mesh_path, '{}{}_recon.ply'.format(line, '_50k'))
            meshfile = os.path.join(mesh_path, '{}_5000k_depth10_recon.ply'.format(line))







            # meshfile = os.path.join(mesh_path, '0101_our_all_color6_{}{}.ply'.format(line, '_50k'))

        if not os.path.exists(meshfile):
            print('file not exists: ', meshfile)
            f = open(meshfile + 'noexists', 'w')
            f.close()
            continue
        stat_file = meshfile + "_stat"
        if not args.regen and os.path.exists(stat_file) and os.path.getsize(stat_file) > 0:
            #load it 
            # d['name'].append(line)
            f = open(stat_file, 'rb')
            cur_dict = pickle.load(f)
            for k in cur_dict:
                d[k].append(cur_dict[k])
            f.close()

            continue

        

        d['name'].append(line)

        mesh = trimesh.load(meshfile)

        if args.t == "siren" or args.compone:
            #save the largest component
            meshfile_max = meshfile.replace('.ply', '_component0.ply')
            # print('mesh info before: {} {}'.format(mesh.vertices.shape, mesh.faces.shape))
            mesh_split = mesh.split(only_watertight=False)
            # print('number of split: ', len(mesh_split))
            if len(mesh_split) > 1:
                max_face = -1
                max_face_id = -1
                for i in range(len(mesh_split)):
                    cur_face = mesh_split[i].faces.shape[0]
                    # print('cur face: ', cur_face)
                    if cur_face > max_face:
                        # print('cur face: ', cur_face)
                        max_face = cur_face
                        max_face_id = i
                mesh = mesh_split[max_face_id]
                # print('mesh info: {} {}'.format(mesh.vertices.shape, mesh.faces.shape))
                mesh.export(meshfile_max)

        cd, hd, adm, ads, hd_pred2gt, hd_gt2pred = distance_p2mesh(ptnormal[:,:3], ptnormal[:,3:], mesh)
        # if args.ed:
        #   fea_dist = distance_feature2mesh(fea_ptnormal[:,:3],mesh)
        #   d['ED_mean'].append(fea_dist.mean())
        #   d['ED_max'].append(fea_dist.max())
        # else:
        #   d['ED_mean'].append(0.0)
        #   d['ED_max'].append(0.0)

        # d['CD'].append(cd)
        # d['CDL1'].append(cd1)
        # d['HD'].append(hd)
        # d['HDL1'].append(hd1)
        # d['NC'].append(nc)

        d['CD'].append(cd)
        # d['CDL1'].append(cd1)
        d['HD'].append(hd)
        d['HDpred2gt'].append(hd_pred2gt)
        d['HDgt2pred'].append(hd_gt2pred)

        # d['HDL1'].append(hd1)
        # d['NC'].append(nc)

        # print('adm {} ads {} !!!'.format(adm, ads))
        d['AngleDiffMean'].append(adm)
        d['AngleDiffStd'].append(ads)

        if not args.ed:
            d['FeaDfgt2pred'].append(0.0)
            d['FeaDfpred2gt'].append(0.0)
            d['FeaAnglegt2pred'].append(0.0)
            d['FeaAnglepred2gt'].append(0.0)
            d['FeaDf'].append(0.0)
            d['FeaAngle'].append(0.0)
        else:
            gt_ptangle = np.loadtxt(os.path.join(gt_ptangle_path, line + '_detectfea4e-3.ptangle'))
            pred_ptangle_path = meshfile[:-4]+'_4e-3.ptangle'
            if not os.path.exists(pred_ptangle_path) or args.regen:
                os.system('~/SimpleSample -i {} -o {} -s 4e-3'.format(meshfile, pred_ptangle_path))
            pred_ptangle = np.loadtxt(pred_ptangle_path).reshape(-1,4)
            
            #update 1210, for smooth case: if gt fea is empty, or pred fea is empty, then return 0, be aware that other method may benifit when all pred fea is empty
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
            # print('key: ', key)
            # print('len key: ', len(d[key]))
            d[key].append(sum(d[key])/len(d[key]))

    # d['name'].append('mean')
    # d['CD'].append(sum(d['CD'])/len(lines))
    # # d['CDL1'].append(sum(d['CDL1'])/len(lines))
    # d['HD'].append(sum(d['HD'])/len(lines))
    # # d['HDL1'].append(sum(d['HDL1'])/len(lines))
    # d['NC'].append(sum(d['NC'])/len(lines))
    # d['ED_mean'].append(sum(d['ED_mean'])/len(lines))
    # d['ED_max'].append(sum(d['ED_max'])/len(lines))


    # df = pd.DataFrame(d, columns=['name', 'CD', 'CDL1', 'HD', 'HDL1', 'NC'])
    # df = pd.DataFrame(d, columns=['name', 'CD', 'HD', 'AngleDiffMean', 'AngleDiffStd','FeaDfgt2pred', 'FeaDfpred2gt', 'FeaDf', 'FeaAnglegt2pred', 'FeaAnglepred2gt', 'FeaAngle'])
    df = pd.DataFrame(d, columns=['name', 'CD', 'HD', 'HDpred2gt', 'HDgt2pred', 'AngleDiffMean', 'AngleDiffStd','FeaDfgt2pred', 'FeaDfpred2gt', 'FeaDf', 'FeaAnglegt2pred', 'FeaAnglepred2gt', 'FeaAngle'])

    df.to_csv(output_path, index = False, header=True)

def gather_results():
    # model_list = 'selected.txt'
    # model_list = 'ourgood_new.txt'
    model_list = 'ablation20.txt'
    # model_list = 'ourgood.txt'
    # model_list = '/mnt/sdf1/haog/data/implicitcad/1211_200_names.txt'
    # model_list = '/mnt/sdf1/haog/data/implicitcad/1213_filter.txt'
    # model_list = 'ourgood617.txt'
    # model_list = '/mnt/sdf1/haog/code/siren_ori/siren/5models_tmp.txt'


    # output_path = 'gather{}_ourgood_new.csv'.format(args.s)
    # output_path = 'gather_rimls_ourgood_new.csv'.format(args.s)

    # output_path = 'gather_256x3_ablation_netsize.csv'.format(args.s)

    # output_path = 'gather_igrsiren_loss.csv'
    # output_path = 'gather_1211_200model.csv'

    output_path = 'gather_ablation_0110.csv'




    f2c = {}
    # f2c['poisson'] = '1210_poisson_200.csv'
    # f2c['rimls'] = '1210_rimls_200.csv'
    # f2c['siren'] = '1210_siren_200.csv'
    # f2c['igr'] = '1210_igr_200.csv'
    # # f2c['our-color']='256.csv'

    # f2c['our-nocolor']='our_200_nocolor_1212.csv'
    # f2c['poisson']='poisson_fea_eval.csv'

    # f2c['baseline4'] = 'baseline4_fea_eval.csv'

    # #origin version
    # f2c['poisson']='poisson_fea_eval_new.csv'
    # # f2c['poisson']='poisson_fea_eval_ori.csv'
    # f2c['rimls']='RIMLS_fea_eval_nooct.csv'
    # f2c['siren'] = 'siren_fea_eval{}_new.csv'.format(args.s)
    # f2c['igr'] = 'igr_fea_eval{}_new.csv'.format(args.s)
    # f2c['igr-small'] = 'baseline4_fea_eval{}_new.csv'.format(args.s)
    # f2c['ours']='linearassign_fea_eval{}_new.csv'.format('_nooct')


    # f2c['rimls'] = 'RIMLS_fea_eval_nooct.csv'

    #new version
    # f2c['64x3'] = 'ablation_64x3_318.csv'
    # f2c['128x3'] = 'ablation_128x3_318.csv'
    # f2c['256x3'] = '256x3.csv'
    # f2c['256x7'] = 'ablation_256x7_318.csv'
    # # f2c['512x3'] = 'ablation_512x3.csv'
    # f2c['512x7'] = 'ablation_512x7_318.csv'
    # f2c['512x7_skipin'] = 'ablation_512x7_skipin.csv'
    # f2c['igr'] = 'igr_fea_eval_nooct.csv'

    f2c['nonormal'] = 'ablation_nonormal_318.csv'
    f2c['nooff'] = 'ablation_nooff_318.csv'
    f2c['nocorrect'] = 'ablation_nocor_318.csv'
    f2c['ours'] = '256x3.csv'


    # f2c['siren_ori'] = 'siren_fea_eval{}_new.csv'.format('_nooct')
    # f2c['siren_ps'] = 'siren_ps_all.csv'
    # f2c['siren_samecoeff'] = 'siren_pssamecoeff320.csv'
    # f2c['igr'] = 'igr_fea_eval{}_new.csv'.format('_nooct')
    # f2c['256x4'] = 'linearassign_fea_eval{}_new.csv'.format('_nooct')
    # f2c['256x3'] = '256x3.csv'

    # f2c['igrloss'] = 'igrloss_all.csv'
    # f2c['sirenloss'] = 'sirenloss_all.csv'
    # f2c['ourbaseline'] = 'igrloss_all.csv'
    # f2c['single'] = 'baseline4_fea_eval{}_new.csv'.format('_nooct')
    # f2c['multiple'] = 'linearassign_fea_eval_nooct_new.csv'


    # d = {'name':[], 'CD':[], 'HD':[], 'AngleDiffMean':[], 'AngleDiffStd':[], 'FeaDfgt2pred':[], 'FeaDfpred2gt':[], 'FeaDf':[], 'FeaAnglegt2pred':[], 'FeaAnglepred2gt':[], 'FeaAngle':[]}
    d = {'name':[], 'CD':[], 'HD':[], 'AngleDiffMean':[], 'FeaDf':[], 'FeaAngle':[]}


    multiply_term = {'CD', 'HD', 'FeaDf'}


    # d = {'name':[], 'CD':[], 'HD':[], 'NC':[], 'ED_mean':[], 'ED_max':[]}

    f = open(model_list, 'r')
    models = f.readlines()
    f.close()

    model_set = set()
    for m in models:
        m = m.strip()
        # m = m[:-7]
        m = m[:-4]
        model_set.add(m)

    print('model size: ', len(model_set))

    for k, v in f2c.items():
        # metrics = {'CD':0.0, 'HD':0.0, 'NC':0.0, 'ED_mean':0.0, 'ED_max':0.0}
        # metrics = {'CD':0.0, 'HD':0.0, 'AngleDiffMean':0.0, 'AngleDiffStd':0.0, 'FeaDfgt2pred':0.0, 'FeaDfpred2gt':0.0, 'FeaDf':0.0, 'FeaAnglegt2pred':0.0, 'FeaAnglepred2gt':0.0, 'FeaAngle':0.0}
        
        metrics = {'CD':0.0, 'HD':0.0, 'AngleDiffMean':0.0, 'FeaDf':0.0, 'FeaAngle':0.0}
        
        with open(v, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # print ("row key: ", row.keys())
                if row['name'] in model_set:
                    # cd = cd + row['CD']
                    # hd = hd + row['HD']
                    # nc = nc + row['NC']
                    # ed_mean = ed_mean + row['ED_mean']
                    # ed_max = ed_max + row['ED_max']
                    for kk in metrics.keys():
                        if kk in multiply_term:
                            metrics[kk] = metrics[kk] + 1000.0 * float(row[kk])
                        else:
                            metrics[kk] = metrics[kk] + float(row[kk])
        d['name'].append(k)
        for kk in metrics.keys():
            d[kk].append(metrics[kk] / len(model_set))

    # df = pd.DataFrame(d, columns=['name', 'CD', 'HD', 'AngleDiffMean', 'AngleDiffStd', 'FeaDfgt2pred', 'FeaDfpred2gt', 'FeaDf', 'FeaAnglegt2pred', 'FeaAnglepred2gt', 'FeaAngle'])
    df = pd.DataFrame(d, columns=['name', 'CD', 'HD', 'AngleDiffMean', 'FeaDf', 'FeaAngle'])
    
    df.to_csv(output_path, index = False, header=True)

def gather_results_0117():
    # model_list = '/mnt/sdf1/haog/code/ImplicitCAD_script/fns/all/ablation100.txt'
    model_list = 'tmp_fns/ablation_our_valid_100.txt'

    # model_list = 'tmp_fns/all_valid_norimls_append200.txt'

    # model_list = 'tmp_fns/200_valid_norimls.txt'


    # output_path = 'gather_ablation_loss_0119.csv'

    # output_path = '200_comparison.csv'
    # output_path = 'gather_ablation_netsize_0120.csv'
    # output_path = 'gather_ablation_0217.csv'
    # output_path = 'gather_ablation_0309.csv'
    # output_path = 'gather_neural_splines_all.csv'
    # output_path = 'gather_ns_all_0411.csv'
    
    output_path = 'gather_ablation_100_0514.csv'

    # output_path = 'ablation100_corres_cor.csv'
    





    f2c = {}
    f2c['normal'] = 'ablation100_normal.csv'
    f2c['offsurface'] = 'ablation100_offsurface.csv'
    f2c['correction'] = 'ablation100_correction.csv'
    f2c['consistent'] = 'ablation_0308_cons.csv'
    f2c['cc'] = 'ablation100_corres_cor.csv'
    f2c['ours'] = 'our_mesh_color6_part1_merge200_merge_part2valid.csv'

    # f2c['ns_one'] = 'neural_spline_one_256.csv'
    # f2c['ns_grid'] = 'neural_spline_grid_512.csv'
    # f2c['ns_grid'] = 'neural_spline_grid_512.csv'
    # f2c['ours'] = 'our_mesh_color6_part1_merge200_merge_part2valid.csv'
    # f2c['nocolor'] = 'our_mesh_nocolor_part1_merge200.csv'

    # f2c['ns'] = 'neural_spline_grid_512_all.csv'
    # f2c['nocolor'] = 'our_mesh_nocolor_part1_merge200.csv'
    # f2c['ours'] = 'all_our_color6_para_append200.csv'
    # f2c['ours'] = 'our_mesh_color6_part1_merge200_merge_part2valid.csv'


    # f2c['poisson'] = '0101_poisson_200_para.csv'
    # f2c['rimls'] = '200_rimls_para.csv'
    # f2c['siren'] = '200_siren_para.csv'
    # f2c['igr'] = '0101_igr_200_para.csv'
    # f2c['color'] = 'our_para_200_color6_0107.csv'
    # f2c['nocolor'] = 'our_para_200_nocolor_0107.csv'

    # f2c['64x3'] = 'ablation64x3.csv'
    # f2c['128x3'] = 'ablation128x3.csv'
    # f2c['ours'] = 'our_mesh_color6_part1_merge200_merge_part2valid.csv'
    # f2c['256x7'] = 'ablation256x7.csv'
    # f2c['512x3'] = 'ablation512x3.csv'
    # f2c['512x7'] = 'ablation512x7.csv'

    # f2c['baseline'] = 'ablationbl.csv'
    # f2c['ours-fea'] = 'ablation_our_with_feature.csv'
    # f2c['igr-fea'] = 'ablation_igr_with_feature.csv'
    # f2c['ours'] = 'our_mesh_color6_part1_merge200_merge_part2valid.csv'




    d = {'name':[], 'CD':[], 'HD':[], 'AngleDiffMean':[], 'FeaDf':[], 'FeaAngle':[]}


    multiply_term = {'CD', 'HD', 'FeaDf'}


    # d = {'name':[], 'CD':[], 'HD':[], 'NC':[], 'ED_mean':[], 'ED_max':[]}

    f = open(model_list, 'r')
    models = f.readlines()
    f.close()

    model_set = set()
    for m in models:
        m = m.strip()
        # m = m[:-7]
        m = m[:-4]
        model_set.add(m)

    print('model size: ', len(model_set))
    for k, v in f2c.items():
        # metrics = {'CD':0.0, 'HD':0.0, 'NC':0.0, 'ED_mean':0.0, 'ED_max':0.0}
        # metrics = {'CD':0.0, 'HD':0.0, 'AngleDiffMean':0.0, 'AngleDiffStd':0.0, 'FeaDfgt2pred':0.0, 'FeaDfpred2gt':0.0, 'FeaDf':0.0, 'FeaAnglegt2pred':0.0, 'FeaAnglepred2gt':0.0, 'FeaAngle':0.0}
        print('k:', k)
        metrics = {'CD':0.0, 'HD':0.0, 'AngleDiffMean':0.0, 'FeaDf':0.0, 'FeaAngle':0.0}
        valid_count = 0
        with open(v, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            valid_set = set()
            for row in csv_reader:
                # print ("row key: ", row.keys())
                if row['name'] in model_set:
                    # cd = cd + row['CD']
                    # hd = hd + row['HD']
                    # nc = nc + row['NC']
                    # ed_mean = ed_mean + row['ED_mean']
                    # ed_max = ed_max + row['ED_max']
                    valid_count += 1
                    valid_set.add(row['name'])
                    for kk in metrics.keys():
                        if kk in multiply_term:
                            metrics[kk] = metrics[kk] + 1000.0 * float(row[kk])
                        else:
                            # print('name {} key {} rowkk: {}'.format(row['name'], kk, row[kk]))
                            metrics[kk] = metrics[kk] + float(row[kk])
            
            # rest_set = model_set - valid_set
            # print('rest set: ', rest_set)

        print('{} valid count: {}'.format(k, valid_count))
        d['name'].append(k)
        for kk in metrics.keys():
            d[kk].append(metrics[kk] / valid_count)

    # df = pd.DataFrame(d, columns=['name', 'CD', 'HD', 'AngleDiffMean', 'AngleDiffStd', 'FeaDfgt2pred', 'FeaDfpred2gt', 'FeaDf', 'FeaAnglegt2pred', 'FeaAnglepred2gt', 'FeaAngle'])
    df = pd.DataFrame(d, columns=['name', 'CD', 'HD', 'AngleDiffMean', 'FeaDf', 'FeaAngle'])
    
    df.to_csv(output_path, index = False, header=True)

def get_sub_table():
    # model_list = 'tmp_fns/ablation_our_valid.txt'
    model_list = 'tmp_fns/ablation_our_valid_100.txt'
    # in_table = 'our_mesh_color6_part1_merge200_merge_part2valid.csv'
    # out_table = 'our_ablation_104.csv'

    # in_table = 'ablation_0308_cons.csv'
    # out_table = 'ablation_0308_cons_100.csv'

    in_table = 'our_mesh_color6_part1_merge200_merge_part2valid.csv'
    out_table = 'our_mesh_color6_ablation100.csv'

    d = {'name':[], 'CD':[], 'HD':[], 'AngleDiffMean':[], 'FeaDf':[], 'FeaAngle':[]}
    metrics = {'CD':0.0, 'HD':0.0, 'AngleDiffMean':0.0, 'FeaDf':0.0, 'FeaAngle':0.0}
    f = open(model_list, 'r')
    models = f.readlines()
    f.close()

    model_set = set()
    for m in models:
        m = m.strip()
        # m = m[:-7]
        m = m[:-4]
        model_set.add(m)

    print('model size: ', len(model_set))
    
    valid_count = 0
    with open(in_table, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # print ("row key: ", row.keys())
            if row['name'] in model_set:
                # cd = cd + row['CD']
                # hd = hd + row['HD']
                # nc = nc + row['NC']
                # ed_mean = ed_mean + row['ED_mean']
                # ed_max = ed_max + row['ED_max']
                valid_count += 1
                for kk in metrics.keys():
                    # if kk in multiply_term:
                    #     metrics[kk] = metrics[kk] + 1000.0 * float(row[kk])
                    # else:
                    #     metrics[kk] = metrics[kk] + float(row[kk])
                    d[kk].append(float(row[kk]))
                    metrics[kk] = metrics[kk] + float(row[kk])
                d['name'].append(row['name'])
    
    d['name'].append('mean')
    for kk in metrics.keys():
        d[kk].append(metrics[kk] / valid_count)
            
    print('valid count: {}'.format(valid_count))
    df = pd.DataFrame(d, columns=['name', 'CD', 'HD', 'AngleDiffMean', 'FeaDf', 'FeaAngle'])
    
    df.to_csv(out_table, index = False, header=True)


if __name__ == '__main__':
    # test_distance_p2p()
    # test_mesh_samples()
    if args.subtable:
        get_sub_table()
    elif args.gather:
        # gather_results()
        gather_results_0117()
    else:
        # compute_all()
        compute_all_1210()