import os
import sys
import time
import trimesh
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

import argparse
# parse args first and set gpu id
parser = argparse.ArgumentParser()
parser.add_argument('--points_batch', type=int, default=16384, help='point batch size') 
parser.add_argument('--nepoch', type=int, default=15001, help='number of epochs to train for')
parser.add_argument('--conf', type=str, default='setup.conf')
parser.add_argument('--expname', type=str, default='single_shape')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU auto]')
parser.add_argument('--is_continue', default=False, action="store_true", help='continue')
parser.add_argument('--timestamp', default='latest', type=str) #not used anymore
parser.add_argument('--checkpoint', default='latest', type=str)
parser.add_argument('--eval', default=False, action="store_true")
parser.add_argument('--evaldist', default=False, action="store_true")
parser.add_argument('--vismask', default=False, action="store_true")
parser.add_argument('--test', default=False, action="store_true", help='use test data')
parser.add_argument('--splitpos', default = -1, type = int, help='for testing the best position for spliting the network')
parser.add_argument('--summary', default = True, action="store_false", help = 'write summary')
parser.add_argument('--laplace', default = False, help = 'use laplacian term')
parser.add_argument('--baseline', default = False, action="store_true", help = 'run baseline')
parser.add_argument('--csvm',type=float, default = 1, help = 'c for svm')
parser.add_argument('--th_closeness',type=float, default = 1e-5, help = 'threshold deciding whether two points are the same')
parser.add_argument('--onehot', default = False, action="store_true", help = 'set onehot flag as true')
parser.add_argument('--cpu', default = False, action="store_true", help = 'save for cpu device')
parser.add_argument('--square', default = False, action="store_true", help = 'use quadratic manifold loss')
parser.add_argument('--offsurface', default = True, action="store_false", help = 'use assignment loss')
parser.add_argument('--ab', default='none', type=str, help = 'ablation')
parser.add_argument('--siren', default = False, action="store_true", help = 'siren normal loss')
parser.add_argument('--pt', default='ptfile path', type=str) 
parser.add_argument('--feature_sample', action="store_true", help = 'use feature curve samples')
parser.add_argument('--lossgrad2', action="store_true", help = 'use feature samples')
parser.add_argument('--num_feature_sample', type=int, default=2048, help ='number of bs feature samples')
parser.add_argument('--all_feature_sample', type=int, default=10000, help ='number of all feature samples')

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
from utils.plots import plot_surface, plot_cuts, plot_masks, plot_masks_maxsdf, plot_cuts_axis
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from plywrite import save_vert_color_ply, save_vertnormal_color_ply
import matplotlib.cm as cm
from numpy import linalg as LA
import torch.nn.functional as F

def generate_random_color_palette(n_color, flag_partnet = False):
    if flag_partnet:
        np.random.seed(1)
    else:
        np.random.seed(0)
    return np.random.rand(n_color, 3)

def save_mesh_off(filename, vertices, faces, f_mask):
    n_color = np.max(f_mask) + 1
    print('n color: ', n_color)
    colormap = np.round(255 * generate_random_color_palette(n_color)).astype('int')
    f = open(filename, 'w')
    f.write('COFF\n{} {} 0\n'.format(vertices.shape[0], faces.shape[0]))
    for i in range(vertices.shape[0]):
        f.write('{} {} {}\n'.format(vertices[i][0],vertices[i][1],vertices[i][2]))
    for i in range(faces.shape[0]):
        f.write('3 {} {} {} '.format(faces[i][0], faces[i][1], faces[i][2]))
        if f_mask[i] == -1:
            f.write('255 255 255\n')
        else:
            f.write('{} {} {}\n'.format(colormap[f_mask[i]][0], colormap[f_mask[i]][1], colormap[f_mask[i]][2]))
        # f.write('3 {} {} {}\n'.format(faces[i][0], faces[i][1], faces[i][2]))


def save_ply_data_numpy(filename, array):
    f = open(filename, 'w')
    f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float opacity\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'.format(array.shape[0]))
    for i in range(array.shape[0]):
        for j in range(6):
            f.write("{:f} ".format(array[i][j]))
        for j in range(3):
            f.write("{:d} ".format(int(array[i][j+6])))
        f.write('{:f}\n'.format(array[i][9]))
    f.close()

def visualize_ptnormal_loss(filename, ptnormal, loss, flag_exp = False):
    loss = (loss - np.min(loss))/(np.max(loss) - np.min(loss) + 1e-6)
    loss = loss.reshape(-1,1)
    # print ('loss shape ', loss.shape)
    if flag_exp:
        loss = np.exp(loss)
        loss = (loss - np.min(loss))/(np.max(loss) - np.min(loss))
    pt_color = loss * np.array([255, 0, 0]) + (1.0 - loss) * np.array([255, 255, 255])
    plydata = np.concatenate([ptnormal, pt_color, np.ones([ptnormal.shape[0], 1])], 1)
    save_ply_data_numpy(filename, plydata)

class ReconstructionRunner:
    def run_nhrepnet_training(self):
        print("running")
        self.data = self.data.cuda()
        self.data.requires_grad_()
        feature_mask_cpu = self.feature_mask.numpy()
        self.feature_mask = self.feature_mask.cuda()
        # n_patch_batch = 2048
        # n_feature_batch = 2048
        # n_batchsize = n_feature_batch + n_branch * n_patch_batch
        n_branch = int(torch.max(self.feature_mask).item())
        n_batchsize = self.points_batch
        n_patch_batch = n_batchsize // n_branch
        # n_feature_batch = n_batchsize - n_patch_batch * n_branch
        n_patch_last = n_batchsize - n_patch_batch * (n_branch - 1)
        print("patch batch & last patch batch: ", n_patch_batch, n_patch_last)
        print('number of branches: ', n_branch)
        

        #first non feature then feature elements
        omega_1 = 1.0
        a = 2
        patch_sup = True
        # offset_omega = 1.0
        offset_omega = 0.0
        offset_sigma = 0.01
        weight_mnfld_h = 1
        weight_mnfld_cs = 1
        weight_correction = 1
        a_correction = 100
        flag_eta = False
        eta = 0.0


        # print ("feature mask shape: ", feature_mask_cpu.shape)
        patch_id = []
        patch_id_n = []
        for i in range(n_branch):
            patch_id = patch_id + [np.where(feature_mask_cpu == i + 1)[0]]
            patch_id_n = patch_id_n + [patch_id[i].shape[0]]
        if self.eval:
            #to be changed
            print("evaluating epoch: {0}".format(self.startepoch))
            my_path = os.path.join(self.cur_exp_dir, 'evaluation', str(self.startepoch))

            utils.mkdir_ifnotexists(os.path.join(self.cur_exp_dir, 'evaluation'))
            utils.mkdir_ifnotexists(my_path)
            self.network.flag_softmax = True
            # for i in range(n_branch + 1):            
            # for i in range(n_branch + 2):  #for testing
            for i in range(1):            
                self.network.flag_output = i + 1
                self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            # self.plot_masks_maxsdf(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_mask" + str(i))

            # self.network.flag_onehot = True
            # for i in range(n_branch + 1):
            #     self.network.flag_output = i + 1
            #     self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i) + "_onehot", with_cuts = True)
            # self.network.flag_output = 0
            # # self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_onehotmask" + str(i))
            # self.network.flag_onehot = False
            return

        print("training")

        if args.summary == True:
            writer = SummaryWriter(os.path.join("summary", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1

        print('summary status: ', args.summary)
        # branch mask is predefined
        branch_mask = torch.zeros(n_branch, n_batchsize).cuda()
        single_branch_mask_gt = torch.zeros(n_batchsize, n_branch).cuda()
        #for cross entropy loss, id start from 0
        single_branch_mask_id = torch.zeros([n_batchsize], dtype = torch.long).cuda()
        for i in range(n_branch - 1):
            branch_mask[i, i * n_patch_batch : (i + 1) * n_patch_batch] = 1.0
            single_branch_mask_gt[i * n_patch_batch : (i + 1) * n_patch_batch, i] = 1.0
            single_branch_mask_id[i * n_patch_batch : (i + 1) * n_patch_batch] = i

        #last patch
        branch_mask[n_branch - 1, (n_branch - 1) * n_patch_batch:] = 1.0
        single_branch_mask_gt[(n_branch - 1) * n_patch_batch:, (n_branch - 1)] = 1.0
        single_branch_mask_id[(n_branch - 1) * n_patch_batch:] = (n_branch - 1)

        for epoch in range(self.startepoch, self.nepochs + 1):
            #indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            #first feature part then non-feature part
            # indices = torch.tensor(feature_id[0][np.random.choice(n_feature, n_feature_batch, True)]).cuda()
            indices = torch.empty(0,dtype=torch.int64).cuda()
            #current feature mask
            # cur_feature_mask = torch.zeros(n_branch * n_patch_batch + n_feature_batch).cuda()
            # cur_feature_mask[0: n_feature_batch] = 1.0
            cur_feature_mask = torch.zeros(n_batchsize).cuda()
            # print("patch id num: ", patch_id_n)
            for i in range(n_branch - 1):
                indices_nonfeature = torch.tensor(patch_id[i][np.random.choice(patch_id_n[i], n_patch_batch, True)]).cuda()
                indices = torch.cat((indices, indices_nonfeature), 0)
            #last patch
            indices_nonfeature = torch.tensor(patch_id[n_branch - 1][np.random.choice(patch_id_n[n_branch - 1], n_patch_last, True)]).cuda()
            indices = torch.cat((indices, indices_nonfeature), 0)            

            cur_data = self.data[indices]
            mnfld_pnts = cur_data[:, :self.d_in] #n_indices x 3
            mnfld_sigma = self.local_sigma[indices] #noise points

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                self.save_checkpoints(epoch)

            # if epoch != self.startepoch and epoch % self.conf.get_int('train.plot_frequency') == 0:
            # if epoch == self.nepochs:
            if epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                for i in range(n_branch + 1):
                # for i in range(1):
                # for i in range(n_branch + 3): #for testing
                    self.network.flag_output = i + 1
                    self.plot_shapes(epoch, file_suffix = "_" + str(i), with_cuts = False)
                self.network.flag_output = 0
                # adjust coeff
                # if omega_1 < omega_max:
                #     omega_1 = omega_1 * 5
                #     omega_2 = omega_2 * 5
                #     omega_3 = omega_3 * 5

            # change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)
            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()
            # print ("nonmnfld_pnts shape: ", nonmnfld_pnts.shape)

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]
            #print shape
            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss

            mnfld_loss = torch.zeros(1).cuda()
            if not args.ab == 'overall':
                if flag_eta:
                    if epoch < 10000:
                        mnfld_loss = (mnfld_pred.abs()).mean()
                    else:
                        #eta version
                        mnfld_pred_avg = (mnfld_pred.abs()).mean()
                        mnfld_loss_weight = (mnfld_pred.abs() > mnfld_pred_avg).type(torch.float32)
                        mnfld_loss = (mnfld_loss_weight * mnfld_pred.abs()).sum() / mnfld_loss_weight.sum()
                else:
                    if args.square:
                        mnfld_loss = (mnfld_pred*mnfld_pred).mean()
                    else:
                        mnfld_loss = (mnfld_pred.abs()).mean()

            #eta version
            # mask_mfd = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            # sm_loss = nn.Softmax(dim=1)
            # mask_mfd_sm = sm_loss(mask_mfd)

            # mask_bool = single_branch_mask_gt.type(torch.BoolTensor)
            # mask_weight = (mask_mfd_sm[mask_bool] < eta).type(torch.float32)
            # mask_mfd_logsm = F.log_softmax(mask_mfd, dim=1)
            # # mask_loss = (mask_weight * (torch.sum(single_branch_mask_gt * mask_mfd_logsm, dim=1))).mean()
            
            # print ("mnfld loss device: ", mnfld_loss.device)
            loss = loss + weight_mnfld_h *  mnfld_loss
            # manifold loss for patches

            #feature sample
            if args.feature_sample:
                feature_indices = torch.randperm(args.all_feature_sample)[:args.num_feature_sample].cuda()
                feature_pnts = self.feature_data[feature_indices]
                feature_mask_pair = self.feature_data_mask_pair[feature_indices]
                feature_pred_all = self.network(feature_pnts)
                feature_pred = feature_pred_all[:,0]
                feature_mnfld_loss = feature_pred.abs().mean()
                loss = loss + weight_mnfld_h * feature_mnfld_loss  #|h|
                
                #patch loss:
                feature_id_left = [list(range(args.num_feature_sample)), feature_mask_pair[:,0].tolist()]
                feature_id_right = [list(range(args.num_feature_sample)), feature_mask_pair[:,1].tolist()]
                feature_fis_left = feature_pred_all[feature_id_left]
                feature_fis_right = feature_pred_all[feature_id_right]
                feature_loss_patch = feature_fis_left.abs().mean() + feature_fis_right.abs().mean()
                loss += feature_loss_patch

                #consistency loss:
                feature_loss_cons = (feature_fis_left - feature_pred).abs().mean() + (feature_fis_right - feature_pred).abs().mean()
                loss += weight_mnfld_cs *  feature_loss_cons
            offset_loss = torch.zeros(1).cuda()

            offset_patch_loss = torch.zeros(1).cuda()


            if offset_omega > 0.0:

                #for h
                offset_sigma_array = torch.rand(cur_data.shape[0], 1).cuda() * 2 * offset_sigma - offset_sigma
                offset_pnts_pos = mnfld_pnts + cur_data[:,self.d_in:] * offset_sigma_array
                offset_pred_pos = self.network(offset_pnts_pos)

                #mnfld loss
                offset_loss = offset_loss +  (offset_pred_pos[:,0] - offset_sigma_array).abs().mean()
                
                #offset normal loss
                normals = cur_data[:, -self.d_in:]
                offset_grad = gradient(offset_pnts_pos, offset_pred_pos[:, 0])
                offset_loss = offset_loss + (((offset_grad - normals).abs()).norm(2, dim=1)).mean()

                loss = loss + offset_omega * offset_loss

                # offset_grad_neg = gradient(offset_pnts_neg, offset_pred_neg)
                # offset_loss = offset_loss + (((offset_grad_neg - normals).abs()).norm(2, dim=1)).mean()

                #for all patch
                
                offset_all_fi = torch.zeros(n_batchsize, 1).cuda()
                for i in range(n_branch - 1):
                    offset_all_fi[i * n_patch_batch : (i + 1) * n_patch_batch, 0] = offset_pred_pos[i * n_patch_batch : (i + 1) * n_patch_batch, i + 1]
                #last patch
                offset_all_fi[(n_branch - 1) * n_patch_batch:, 0] = offset_pred_pos[(n_branch - 1) * n_patch_batch:, n_branch]

                #manifold loss
                offset_patch_loss = offset_patch_loss +  (offset_all_fi - offset_sigma_array).abs().mean()
                
                #offset normal loss
                normals = cur_data[:, -self.d_in:]
                offset_grad_all = gradient(offset_pnts_pos, offset_all_fi)
                offset_patch_loss = offset_patch_loss + (((offset_grad_all - normals).abs()).norm(2, dim=1)).mean()
                loss = loss + offset_omega * offset_patch_loss



            # all_fi = torch.zeros(n_batchsize, 1).cuda()
            all_fi = torch.zeros([n_batchsize, 1], device = 'cuda')
            for i in range(n_branch - 1):
                all_fi[i * n_patch_batch : (i + 1) * n_patch_batch, 0] = mnfld_pred_all[i * n_patch_batch : (i + 1) * n_patch_batch, i + 1]
            #last patch
            all_fi[(n_branch - 1) * n_patch_batch:, 0] = mnfld_pred_all[(n_branch - 1) * n_patch_batch:, n_branch]

            # mnfld_loss_patch = 0.0
            # for i in range(n_branch):
                # mnfld_loss_patch = mnfld_loss_patch + (mnfld_pred_all[:,i + 1].abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            mnfld_loss_patch = torch.zeros(1).cuda()
            if not args.ab == 'patch':
                if patch_sup:
                    if args.square:
                        mnfld_loss_patch = (all_fi[:,0] * all_fi[:,0]).mean()
                    else:
                        mnfld_loss_patch = all_fi[:,0].abs().mean()
                    # mnfld_loss_patch = (all_fi[:,0]*all_fi[:,0]).mean()
            loss = loss + mnfld_loss_patch

            #correction loss
            correction_loss = torch.zeros(1).cuda()
            if  not (args.ab == 'cor' or args.ab == 'cc') and epoch > 10000 and not args.baseline:
                mismatch_id = torch.abs(mnfld_pred - all_fi[:,0]) > args.th_closeness
                if mismatch_id.sum() != 0:
                    correction_loss = (a_correction * torch.abs(mnfld_pred - all_fi[:,0])[mismatch_id]).mean()
                loss = loss + weight_correction * correction_loss


            #off surface_loss
            offsurface_loss = torch.zeros(1).cuda()
            if not args.ab == 'off' and  args.offsurface:
                # nonmnfld_pred = nonmnfld_pred_all[:,0]
                offsurface_loss = torch.exp(-100.0 * torch.abs(nonmnfld_pred[n_batchsize:])).mean()
                loss = loss + offsurface_loss

            mnfld_consistency_loss = torch.zeros(1).cuda()
            if not (args.ab == 'cons' or args.ab == 'cc'):
                if  args.square:
                    mnfld_consistency_loss = ((mnfld_pred - all_fi[:,0]) * (mnfld_pred - all_fi[:,0])).mean()
                else:
                    mnfld_consistency_loss = (mnfld_pred - all_fi[:,0]).abs().mean()
            loss = loss + weight_mnfld_cs *  mnfld_consistency_loss


            # eikonal loss for all
            grad_loss = torch.zeros(1).cuda()
            #all gradient loss: no good for reconstruction
            # for i in range(n_branch):
            #     single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,i + 1])
            #     grad_loss = grad_loss + ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            # loss = loss + self.grad_lambda * grad_loss
            # eikonal loss for h
            grad_loss_h = torch.zeros(1).cuda()
            single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,0])
            grad_loss_h = ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss_h

            # normals loss
            normals_loss = torch.zeros(1).cuda()
            normals_loss_h = torch.zeros(1).cuda()
            normal_consistency_loss = torch.zeros(1).cuda()
            if not args.siren:
                if not args.ab == 'normal' and self.with_normals:
                    #all normals
                    normals = cur_data[:, -self.d_in:]
                    #defined for h
                    # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                    if patch_sup:
                        branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                        # normals_loss = ((torch.cross(branch_grad, normals, dim=1).abs()).norm(2, dim=1)).mean()
                        normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1)).mean()
                    loss = loss + self.normals_lambda * normals_loss

                    # if epoch > 5000:
                    #     mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                    #     normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()
                    loss = loss + self.normals_lambda * normals_loss_h

                    #only supervised
                    mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                    normal_consistency_loss = (mnfld_grad - branch_grad).abs().norm(2, dim=1).mean()

                else:
                    #ori version
                    # normals_loss = torch.zeros(1)
                    # normals_loss_h = torch.zeros(1)

                    #update 0109, eikonal equation for those points
                    # grad_loss = torch.zeros(1).cuda()
                    # eikonal loss for patch points
                    # print('normal ablation')
                    # grad_loss_h = torch.zeros(1, device = 'cuda')
                    single_nonmnfld_grad = gradient(mnfld_pnts, all_fi[:,0])
                    normals_loss_h = ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
                    loss = loss + self.normals_lambda * normals_loss_h
            else:
                #compute consine normal
                # normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()
                normals = cur_data[:, -self.d_in:]
                normals_loss_h = (1 - F.cosine_similarity(mnfld_grad, normals, dim=-1)).mean()
                loss = loss + self.normals_lambda * normals_loss_h


            #mask only version:
            # loss = mask_loss

            # if epoch % 1000 == 0:
            #     k = 2 * k
            #     if k > k_max:
            #         k = k_max

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            #tensorboard
            if args.summary == True and epoch % 100 == 0:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss h', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Manifold cons loss', mnfld_consistency_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',self.grad_lambda * grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',self.grad_lambda * grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss all', self.normals_lambda * normals_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss h', self.normals_lambda * normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/Normal cs loss', self.normals_lambda * normal_consistency_loss.item(), epoch)
                writer.add_scalar('Loss/Offset loss h', offset_omega * offset_loss.item(), epoch)
                writer.add_scalar('Loss/Offset loss patch', offset_omega * offset_patch_loss.item(), epoch)
                writer.add_scalar('Loss/Assignment loss', correction_loss.item(), epoch)
                writer.add_scalar('Loss/Offsurface loss', offsurface_loss.item(), epoch)


                # writer.add_scalar('Loss/mask loss', omega_1 * mask_loss.item(), epoch)
                # writer.add_scalar('Loss/mask expectation', omega_3 * mask_expectation.item(), epoch)
                # writer.add_scalar('Loss/assignment_loss', assignment_loss.item(), epoch)
                # writer.add_scalar('Loss/G dist', g_dist.item(), epoch)
                # writer.add_scalar('Loss/FG diff', fg_diff_dist, epoch)
                # writer.add_scalar('Loss/Non-feature repulsion', non_feature_repulsion_loss, epoch)
                # writer.add_scalar('Loss/vertical loss', verticle_loss, epoch)
                # if args.laplace:
                #     writer.add_scalar('Loss/laplacian f', f_laplacian, epoch)
                #     writer.add_scalar('Loss/laplacian g', g_laplacian, epoch)

            if epoch % self.conf.get_int('train.status_frequency') == 0:
                # print('Train Epoch: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tManifold loss: {:.6f}'
                #     '\tGrad loss: {:.6f}\tNormals Loss: {:.6f}\t f loss: {:.6f}\t g loss: {:.6f}'.format(
                #     epoch, self.nepochs, 100. * epoch / self.nepochs,
                #     loss.item(), mnfld_loss.item(), grad_loss.item(), normals_loss.item(), f_dist.item(), g_dist.item()))
                # print('fg diff loss: {:.6f}'.format(fg_diff_dist.item()))
                # print('Repulsion Loss: {:.6f}'.format(non_feature_repulsion_loss.item()))
                # print('verticle_loss: {:.6f}'.format(verticle_loss.item()))
                # if args.laplace:
                #     print('laplacian f & g: {:.6f}\t {:.6f}'.format(f_laplacian, g_laplacian))
                # print('a: {:.6f}\t beta_3: {:.6f}'.format(a, beta_3_init))
                print('Train Epoch: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\t Manifold loss: {:.6f}'
                    '\tManifold patch loss: {:.6f}\t grad loss all: {:.6f}\t grad loss h: {:.6f}\t normals loss all: {:.6f}\t normals loss h: {:.6f}'.format(
                    epoch, self.nepochs, 100. * epoch / self.nepochs,
                    loss.item(), mnfld_loss.item(), mnfld_loss_patch.item(), grad_loss.item(), grad_loss_h.item(), normals_loss.item(), normals_loss_h.item()))
                print ("offset loss: ", offset_loss.item())
                print ("offset patch loss: ", offset_patch_loss.item())
                print ("mnfld consistency loss: ", mnfld_consistency_loss.item())
                print ('correction loss: ', correction_loss.item())
                print ('offsurface_loss loss: ', offsurface_loss.item())
                
                if args.feature_sample:
                    print('feature mnfld loss: {} patch loss: {} cons loss: {}'.format(feature_mnfld_loss.item(), feature_loss_patch.item(), feature_loss_cons.item()))

        self.tracing()

    def tracing(self):
        #network definition
        device = torch.device('cuda')
        if args.cpu:
            device = torch.device('cpu')
        network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=3, flag_output = 1,
                                                                                        n_branch = int(torch.max(self.feature_mask).item()),
                                                                                        csg_tree = self.csg_tree,
                                                                                        flag_convex = self.csg_flag_convex,
                                                                                        **self.conf.get_config(
                                                                                        'network.inputs'))
        network.to(device)
        ckpt_prefix = 'exps/single_shape/'
        save_prefix = '{}/'.format(args.pt)
        if not os.path.exists(save_prefix):
            os.mkdir(save_prefix)

        if args.cpu:
            saved_model_state = torch.load(ckpt_prefix + self.foldername + '/checkpoints/ModelParameters/latest.pth', map_location=device)
            network.load_state_dict(saved_model_state["model_state_dict"])
        else:
            saved_model_state = torch.load(ckpt_prefix + self.foldername + '/checkpoints/ModelParameters/latest.pth')
            network.load_state_dict(saved_model_state["model_state_dict"])
        #trace
        example = torch.rand(224,3).to(device)
        traced_script_module = torch.jit.trace(network, example)
        traced_script_module.save(save_prefix + self.foldername + "_model_h.pt")
        print('converting to pt finished')

    def plot_shapes(self, epoch, path=None, with_cuts=False, file_suffix="all"):
        # plot network validation shapes
        with torch.no_grad():

            self.network.eval()

            if not path:
                path = self.plots_dir

            # indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, True)) #modified 0107, with replace


            pnts = self.data[indices, :3]

            #draw nonmnfld pts
            mnfld_sigma = self.local_sigma[indices]
            nonmnfld_pnts = self.sampler.get_points(pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            pnts = nonmnfld_pnts

            plot_surface(with_points=True,
                         points=pnts,
                         decoder=self.network,
                         path=path,
                         epoch=epoch,
                         shapename=self.expname,
                         suffix = file_suffix,
                         **self.conf.get_config('plot'))

            if with_cuts:
                # plot_cuts_axis(points,decoder,latent,path,epoch,near_zero,axis,file_name_sep='/')
                
                plot_cuts_axis(points=pnts,
                          decoder=self.network,
                          latent = None,
                          path=path,
                          epoch=epoch,
                          near_zero=False,
                          axis = 2)

    def __init__(self, **kwargs):
        self.home_dir = os.path.abspath(os.getcwd())
        flag_list = False
        if 'flag_list' in kwargs:
            flag_list = True

        # config setting
        if type(kwargs['conf']) == str:
            self.conf_filename = './conversion/' + kwargs['conf']
            self.conf = ConfigFactory.parse_file(self.conf_filename)
        else:
            self.conf = kwargs['conf']

        self.expname = kwargs['expname']

        # GPU settings, currently we only support single-gpu training
        self.GPU_INDEX = kwargs['gpu_index']
        self.num_of_gpus = torch.cuda.device_count()
        self.eval = kwargs['eval']
        self.evaldist = kwargs['evaldist']

        self.exps_folder_name = 'exps'
        utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name)))
        self.expdir = utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name, self.expname))
        utils.mkdir_ifnotexists(self.expdir)

        if not flag_list:
            if args.test:
                self.input_file = self.conf.get_string('train.input_path')
                self.input_file = self.input_file[:-4] + "_test.xyz"
                self.data = utils.load_point_cloud_by_file_extension(self.input_file)
                self.feature_mask_file = self.conf.get_string('train.feature_mask_path')
                self.feature_mask_file = self.feature_mask_file[:-4] + "_test.txt"
                self.feature_mask = utils.load_feature_mask(self.feature_mask_file)
            else:
                self.input_file = self.conf.get_string('train.input_path')
                self.data = utils.load_point_cloud_by_file_extension(self.input_file)
                self.feature_mask_file = self.conf.get_string('train.feature_mask_path')
                self.feature_mask = utils.load_feature_mask(self.feature_mask_file)
        else:
            self.input_file = os.path.join(self.conf.get_string('train.input_path'), kwargs['file_prefix']+'.xyz')
            if not os.path.exists(self.input_file):
                self.flag_data_load = False
                return
            self.flag_data_load = True
            self.data = utils.load_point_cloud_by_file_extension(self.input_file)
            self.feature_mask_file = os.path.join(self.conf.get_string('train.input_path'), kwargs['file_prefix']+'_mask.txt')
            if not args.baseline:
                self.feature_mask = utils.load_feature_mask(self.feature_mask_file)

            if args.feature_sample:
                input_fs_file = os.path.join(self.conf.get_string('train.input_path'), kwargs['file_prefix']+'_feature.xyz')
                self.feature_data = np.loadtxt(input_fs_file)
                self.feature_data = torch.tensor(self.feature_data, dtype = torch.float32, device = 'cuda')
                fs_mask_file = os.path.join(self.conf.get_string('train.input_path'), kwargs['file_prefix']+'_feature_mask.txt')
                self.feature_data_mask_pair = torch.tensor(np.loadtxt(fs_mask_file), dtype = torch.int64, device = 'cuda')
        if args.baseline:
            self.csg_tree = [0]
            self.csg_flag_convex = True
        else:
            self.csg_tree = []
            self.csg_tree = ConfigFactory.parse_file(self.input_file[:-4]+'_csg.conf').get_list('csg.list')
            self.csg_flag_convex = ConfigFactory.parse_file(self.input_file[:-4]+'_csg.conf').get_int('csg.flag_convex')
        print ("csg tree: ", self.csg_tree)
        print ("csg convex flag: ", self.csg_flag_convex)
        
        if not flag_list:
            self.foldername = self.conf.get_string('train.foldername')
        else:
            self.foldername = kwargs['folder_prefix'] + kwargs['file_prefix']

        if args.baseline:
            self.feature_mask = torch.ones(self.data.shape[0]).float()

        print ("loading finished")
        print ("data shape: ", self.data.shape)

        sigma_set = []
        ptree = cKDTree(self.data)
        print ("kd tree constructed")

        for p in np.array_split(self.data, 100, axis=0):
            d = ptree.query(p, 50 + 1)
            sigma_set.append(d[0][:, -1])

        sigmas = np.concatenate(sigma_set)
        self.local_sigma = torch.from_numpy(sigmas).float().cuda()

        self.cur_exp_dir = os.path.join(self.expdir, self.foldername)
        utils.mkdir_ifnotexists(self.cur_exp_dir)

        self.plots_dir = os.path.join(self.cur_exp_dir, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.checkpoints_path = os.path.join(self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        model_params_path = os.path.join(self.checkpoints_path, self.model_params_subdir)
        ckpts = os.listdir(model_params_path)
        #if ckpts exists, then continue
        is_continue = False
        if (len(ckpts)) != 0:
            is_continue = True

        self.nepochs = kwargs['nepochs']

        self.points_batch = kwargs['points_batch']

        self.global_sigma = self.conf.get_float('network.sampler.properties.global_sigma')
        self.sampler = Sampler.get_sampler(self.conf.get_string('network.sampler.sampler_type'))(self.global_sigma,
                                                                                                 self.local_sigma)
        self.grad_lambda = self.conf.get_float('network.loss.lambda')
        self.normals_lambda = self.conf.get_float('network.loss.normals_lambda')

        self.with_normals = self.normals_lambda > 0

        self.d_in = self.conf.get_int('train.d_in')

        self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=self.d_in, 
                                                                                n_branch = int(torch.max(self.feature_mask).item()),
                                                                                csg_tree = self.csg_tree,
                                                                                flag_convex = self.csg_flag_convex,
                                                                                **self.conf.get_config(
                                                                                    'network.inputs'))


        print (self.network)

        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))
        self.weight_decay = self.conf.get_float('train.weight_decay')

        self.startepoch = 0
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0),
                    "weight_decay": self.weight_decay
                },
            ])

        # if continue load checkpoints
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, self.foldername, 'checkpoints')
            print('loading checkpoint from: ', old_checkpnts_dir)
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.startepoch = saved_model_state['epoch']

    def get_learning_rate_schedules(self, schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self, epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

if __name__ == '__main__':

    if args.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = args.gpu

    nepoch = args.nepoch
    conf = ConfigFactory.parse_file('./conversion/' + args.conf)
    folderprefix = conf.get_string('train.folderprefix')
    fileprefix_list = conf.get_list('train.fileprefix_list')

    trainrunners = []
    for i in range(len(fileprefix_list)):
        fp = fileprefix_list[i]
        print ('cur model: ', fp)

        begin = time.time()
        trainrunners.append(ReconstructionRunner(
            conf=args.conf,
            folder_prefix = folderprefix,
            file_prefix = fp,
            points_batch=args.points_batch,
            nepochs=nepoch,
            expname=args.expname,
            gpu_index=gpu,
            is_continue=args.is_continue,
            timestamp=args.timestamp,
            checkpoint=args.checkpoint,
            eval=args.eval,
            splitpos = args.splitpos,
            evaldist = args.evaldist, 
            flag_list = True
            )
            )

        if trainrunners[i].flag_data_load:
            trainrunners[i].run_nhrepnet_training()
            end = time.time()
            dur = end - begin
            if args.baseline:
                fp = fp+"_bl"