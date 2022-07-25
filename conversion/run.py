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
parser.add_argument('--points_batch', type=int, default=16384, help='point batch size') #from 16384 to 8192
# parser.add_argument('--nepoch', type=int, default=100000, help='number of epochs to train for') #default
# parser.add_argument('--nepoch', type=int, default=15000, help='number of epochs to train for')
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
parser.add_argument('--visloss', default = False, action="store_true", help = 'visualizating loss')
parser.add_argument('--aml', default = False, action="store_true", help = 'training on aml')
parser.add_argument('--ori', default = False, action="store_true", help = 'run origin version of igr')
parser.add_argument('--assign', default = True, action="store_false", help = 'use assignment loss, default true')
parser.add_argument('--linearassign', default = True, action="store_false", help = 'use linear assignment loss, default true')
parser.add_argument('--square', default = False, action="store_true", help = 'use quadratic manifold loss')
parser.add_argument('--offsurface', default = True, action="store_false", help = 'use assignment loss')
parser.add_argument('--getmask', default = False, action="store_true", help = 'get mask from point cloud')
parser.add_argument('--ab', default='none', type=str, help = 'ablation')

parser.add_argument('--siren', default = False, action="store_true", help = 'siren normal loss')

parser.add_argument('--pt', default='ptfile', type=str) #not used anymore

# parser.add_argument('--branch', default = -1, type = int, help='number of branches')
#training stage
parser.add_argument('--stage', type=int, default=0, help='training stage: 0 for normal training, 1 for mask only training, 2 for sdf only training')

parser.add_argument('--feature_sample', action="store_true", help = 'use feature samples')
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

from focalloss import *
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


    def run_ori(self):

        print("running")

        self.data = self.data.cuda()
        self.data.requires_grad_()

        if self.eval:

            print("evaluating epoch: {0}".format(self.startepoch))
            my_path = os.path.join(self.cur_exp_dir, 'evaluation', str(self.startepoch))

            utils.mkdir_ifnotexists(os.path.join(self.cur_exp_dir, 'evaluation'))
            utils.mkdir_ifnotexists(my_path)
            self.plot_shapes(epoch=self.startepoch, path=my_path, with_cuts=True)
            return

        print("training")

        for epoch in range(self.startepoch, self.nepochs + 1):

            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))

            cur_data = self.data[indices]

            mnfld_pnts = cur_data[:, :self.d_in]
            mnfld_sigma = self.local_sigma[indices]

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0 and epoch:
                print('saving checkpoint: ', epoch)
                self.save_checkpoints(epoch)
                print('plot validation epoch: ', epoch)
                self.plot_shapes(epoch, with_cuts=True)

            # change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)

            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()

            # forward pass

            mnfld_pred = self.network(mnfld_pnts)
            nonmnfld_pred = self.network(nonmnfld_pnts)

            # compute grad

            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

            # manifold loss

            mnfld_loss = (mnfld_pred.abs()).mean()

            
            # eikonal loss

            grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()

            loss = mnfld_loss + self.grad_lambda * grad_loss

            #feature sample:
            if args.feature_sample:
                feature_indices = torch.randperm(args.all_feature_sample)[:args.num_feature_sample].cuda()
                feature_pnts = self.feature_data[feature_indices]
                # feature_mask_pair = self.feature_data_mask_pair[feature_indices]
                feature_pred = self.network(feature_pnts)
                feature_mnfld_loss = feature_pred.abs().mean()
                loss += feature_mnfld_loss  #|h|

                #no  patch loss and close loss

            
            if args.lossgrad2:
                
                # w_grad2 = 10
                w_grad2 = 0.1

                gradx_grad = gradient(nonmnfld_pnts, nonmnfld_grad[:, 0])
                grady_grad = gradient(nonmnfld_pnts, nonmnfld_grad[:, 1])
                gradz_grad = gradient(nonmnfld_pnts, nonmnfld_grad[:, 2])

                latterx = (gradx_grad * nonmnfld_grad).sum(-1).view(-1,1)
                lattery = (grady_grad * nonmnfld_grad).sum(-1).view(-1,1)
                latterz = (gradz_grad * nonmnfld_grad).sum(-1).view(-1,1)
                
                latter = torch.cat([latterx, lattery, latterz], axis = -1)
                # print('latterx size: ', latterx.shape)
                # print('latter size: ', latter.shape)
                # print('grad size: ', nonmnfld_grad.shape)

                lossgrad2 = (nonmnfld_grad * latter).sum(-1).abs().mean()
                
                loss += w_grad2 * lossgrad2
                


                # f_laplacian = (torch.abs(f_gradx_grad[:, 0] + f_grady_grad[:, 1] + f_gradz_grad[:, 2])).mean()
                # g_laplacian = (torch.abs(g_gradx_grad[:, 0] + g_grady_grad[:, 1] + g_gradz_grad[:, 2])).mean()
                # # f_laplacian = 0
                # # g_laplacian = 0
                # loss = loss + beta_3_init * (f_laplacian + g_laplacian)

                # #manifold pts:
                # gradx_grad = gradient(mnfld_pnts, mnfld_grad[:, 0])
                # grady_grad = gradient(mnfld_pnts, mnfld_grad[:, 1])
                # gradz_grad = gradient(mnfld_pnts, mnfld_grad[:, 2])

                # latterx = (gradx_grad * mnfld_grad).sum(-1).view(-1,1)
                # lattery = (grady_grad * mnfld_grad).sum(-1).view(-1,1)
                # latterz = (gradz_grad * mnfld_grad).sum(-1).view(-1,1)
                
                # latter = torch.cat([latterx, lattery, latterz], axis = -1)
                # # print('latterx size: ', latterx.shape)
                # # print('latter size: ', latter.shape)
                # # print('grad size: ', nonmnfld_grad.shape)

                # lossgrad2_mnfld = (mnfld_grad * latter).sum(-1).abs().mean()

                # lossgrad2 = lossgrad2_mnfld
                # loss += w_grad2 * lossgrad2
                
                # print('second order loss of manifold points: {}'.format(lossgrad2_mnfld))                
                


            # normals loss

            if self.with_normals:
                normals = cur_data[:, -self.d_in:]
                normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
                loss = loss + self.normals_lambda * normals_loss
            else:
                normals_loss = torch.zeros(1)

            # back propagation

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            if epoch % self.conf.get_int('train.status_frequency') == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tManifold loss: {:.6f}'
                    '\tGrad loss: {:.6f}\tNormals Loss: {:.6f}'.format(
                    epoch, self.nepochs, 100. * epoch / self.nepochs,
                    loss.item(), mnfld_loss.item(), grad_loss.item(), normals_loss.item()))
                if args.feature_sample:
                    print('feature loss: {}'.format(feature_mnfld_loss))

                if args.lossgrad2:
                    print('loss grad2: {}'.format(lossgrad2.item()))
        self.tracing()

    def run(self):

        print("running")

        self.data = self.data.cuda()
        self.data.requires_grad_()
        self.feature_mask = self.feature_mask.cuda()

        n_feature_split = 2
        n_feature = torch.nonzero(self.feature_mask).shape[0]
        n_nonfeature = self.data.shape[0] - n_feature
        n_points_batch_feature = self.points_batch // n_feature_split
        n_points_batch_nonfeature = self.points_batch - n_points_batch_feature
        print(n_feature, n_nonfeature, n_points_batch_feature, n_points_batch_nonfeature)
        #first non feature then feature elements
        omega_1 = 10.0
        omega_2 = 10.0
        omega_3 = 10.0
        omega_max = 1000.0
        a = 0.01
        a_max = 10.0
        beta_1 = 10
        beta_2 = 1
        eps = 0.01
        # beta_3_init = 0.02
        beta_3_init = 0.0
        beta_3_max = 0.2


        if self.eval:
            print("evaluating epoch: {0}".format(self.startepoch))
            my_path = os.path.join(self.cur_exp_dir, 'evaluation', str(self.startepoch))

            utils.mkdir_ifnotexists(os.path.join(self.cur_exp_dir, 'evaluation'))
            utils.mkdir_ifnotexists(my_path)
            self.network.flag_output = 1
            self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_h", with_cuts = True)
            self.network.flag_output = 2
            # self.plot_shapes(epoch, file_suffix = "_f", with_cuts = True)
            self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_f", with_cuts = True)
            self.network.flag_output = 3
            # self.plot_shapes(epoch, file_suffix = "_g", with_cuts = True)
            self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_g", with_cuts = True)
            return

        print("training")

        if args.summary == True:
            writer = SummaryWriter(os.path.join("runs", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1
        a = pow(10, pow_index) * a
        beta_3_init = pow(10, pow_index) * beta_3_init
        if a > a_max:
            a = a_max
        if beta_3_init > beta_3_max:
            beta_3_init = beta_3_max
        print('summary status: ', args.summary)

        for epoch in range(self.startepoch, self.nepochs + 1):

            #indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            #two parts indices
            indices_nonfeature = torch.tensor(np.random.choice(n_nonfeature, n_points_batch_nonfeature, False))
            indices_feature = n_nonfeature + torch.tensor(np.random.choice(n_feature, n_points_batch_feature, True)) #with replacement
            indices = torch.cat((indices_nonfeature, indices_feature), 0)

            cur_data = self.data[indices]
            cur_feature_mask = self.feature_mask[indices]

            mnfld_pnts = cur_data[:, :self.d_in]
            mnfld_sigma = self.local_sigma[indices] #noise points

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                self.save_checkpoints(epoch)
                if a < a_max:
                    a = a * 10
                if beta_3_init < beta_3_max:
                    beta_3_init = beta_3_init * 10


            # init result saving
            if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                self.network.flag_output = 1
                self.plot_shapes(epoch, file_suffix = "_h", with_cuts = True)
                self.network.flag_output = 2
                self.plot_shapes(epoch, file_suffix = "_f", with_cuts = True)
                self.network.flag_output = 3
                self.plot_shapes(epoch, file_suffix = "_g", with_cuts = True)
                self.network.flag_output = 0
                # adjust coeff
                # if omega_1 < omega_max:
                #     omega_1 = omega_1 * 5
                #     omega_2 = omega_2 * 5
                #     omega_3 = omega_3 * 5

            # change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)

            # use non feature points
            non_feature_pnts = self.data[indices_nonfeature][:, :self.d_in]
            non_feature_sigma = self.local_sigma[indices_nonfeature]
            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pred_all.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]

            # mnfld_pred = self.network(mnfld_pnts)
            # nonmnfld_pred = self.network(nonmnfld_pnts)

            #print shape
            # print("mnfld_pnts size: ")
            # print(mnfld_pnts.size())
            # print("mnfld_pred size: ")
            # print(mnfld_pred.size())

            # compute grad
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss

            mnfld_loss = (mnfld_pred.abs()).mean()
            # eikonal loss

            #grad loss for f & g
            #commented on 20200924
            # grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()

            f_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,1])
            g_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,2])
            f_grad_loss = ((f_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            g_grad_loss = ((g_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            # modified on 20200930, no g_grad
            grad_loss = f_grad_loss + g_grad_loss
            # grad_loss = f_grad_loss

            loss = mnfld_loss + self.grad_lambda * grad_loss

            # normals loss

            if self.with_normals:
                normals = cur_data[:, -self.d_in:]
                #modified below
                #normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
                normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
                loss = loss + self.normals_lambda * normals_loss
            else:
                normals_loss = torch.zeros(1)

            # feature loss
            f_dist = (mnfld_pred_all[:, 1].abs() * cur_feature_mask).mean() * n_feature_split
            g_dist = (mnfld_pred_all[:, 2].abs() * cur_feature_mask).mean() * n_feature_split
            fg_diff_dist = ((mnfld_pred_all[:, 1] - mnfld_pred_all[:, 2]).abs() * cur_feature_mask).mean() * n_feature_split
            #exp version
            # f_loss = (torch.exp(a * mnfld_pred_all[:, 1].abs()) * cur_feature_mask).mean() * n_feature_split
            # g_loss = (torch.exp(a * mnfld_pred_all[:, 2].abs()) * cur_feature_mask).mean() * n_feature_split
            # fg_diff_loss = (torch.exp(a * (mnfld_pred_all[:, 1] - mnfld_pred_all[:, 2]).abs()) * cur_feature_mask).mean() * n_feature_split
            # loss = loss + omega_1 * f_loss + omega_2 * g_loss + omega_3 * fg_diff_loss

            #linear version
            loss = loss + omega_1 * f_dist + omega_2 * g_dist + omega_3 * fg_diff_dist

            # non-feature point penalty loss
            # non_feature_point_penalty = omega_3 * (torch.exp( -a * torch.abs(mnfld_pred_all[:, 2])) * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
            # loss = loss + non_feature_point_penalty
            # back propagation

            #non feature repulstion loss
            # non_feature_repulsion_loss_1 = ((beta_1 * mnfld_pred_all[:, 1].abs() * mnfld_pred_all[:, 2].abs()) * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
            # non_feature_repulsion_loss_2 = ((- beta_2 * torch.log(eps + mnfld_pred_all[:, 1].abs() + mnfld_pred_all[:, 2].abs())) * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
            non_feature_repulsion_loss = (torch.min(mnfld_pred_all[:, 1].abs(), mnfld_pred_all[:, 2].abs())/(mnfld_pred_all[:, 1].abs() + mnfld_pred_all[:, 2].abs() + eps) * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
            # non_feature_repulsion_loss_2 = (1.0 / ((mnfld_pred_all[:, 1] - mnfld_pred_all[:, 2]).abs() + eps)  * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
            loss = loss + beta_1 * non_feature_repulsion_loss

            #verticle loss
            # feature points
            # f_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 1])
            # g_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 2])
            # verticle_loss = (torch.abs(g_grad[:, 0] * f_grad[:, 0] + g_grad[:, 1] * f_grad[:, 1] + g_grad[:, 2] * f_grad[:, 2]) * cur_feature_mask).mean() * n_feature_split
            # all points
            f_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:, 1])
            g_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:, 2])
            verticle_loss = (torch.abs(g_grad[:, 0] * f_grad[:, 0] + g_grad[:, 1] * f_grad[:, 1] + g_grad[:, 2] * f_grad[:, 2])).mean()
            loss = loss + beta_2 * verticle_loss

            #smoothness loss
            if args.laplace: 
                f_gradx_grad = gradient(nonmnfld_pnts, f_grad[:, 0])
                f_grady_grad = gradient(nonmnfld_pnts, f_grad[:, 1])
                f_gradz_grad = gradient(nonmnfld_pnts, f_grad[:, 2])

                g_gradx_grad = gradient(nonmnfld_pnts, g_grad[:, 0])
                g_grady_grad = gradient(nonmnfld_pnts, g_grad[:, 1])
                g_gradz_grad = gradient(nonmnfld_pnts, g_grad[:, 2])

                f_laplacian = (torch.abs(f_gradx_grad[:, 0] + f_grady_grad[:, 1] + f_gradz_grad[:, 2])).mean()
                g_laplacian = (torch.abs(g_gradx_grad[:, 0] + g_grady_grad[:, 1] + g_gradz_grad[:, 2])).mean()
                # f_laplacian = 0
                # g_laplacian = 0
                loss = loss + beta_3_init * (f_laplacian + g_laplacian)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            #tensorboard
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss F',f_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss', normals_loss.item(), epoch)
                writer.add_scalar('Loss/F dist', f_dist.item(), epoch)
                writer.add_scalar('Loss/G dist', g_dist.item(), epoch)
                writer.add_scalar('Loss/FG diff', fg_diff_dist, epoch)
                writer.add_scalar('Loss/Non-feature repulsion', non_feature_repulsion_loss, epoch)
                writer.add_scalar('Loss/vertical loss', verticle_loss, epoch)
                if args.laplace:
                    writer.add_scalar('Loss/laplacian f', f_laplacian, epoch)
                    writer.add_scalar('Loss/laplacian g', g_laplacian, epoch)

            if epoch % self.conf.get_int('train.status_frequency') == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tManifold loss: {:.6f}'
                    '\tGrad loss: {:.6f}\tNormals Loss: {:.6f}\t f loss: {:.6f}\t g loss: {:.6f}'.format(
                    epoch, self.nepochs, 100. * epoch / self.nepochs,
                    loss.item(), mnfld_loss.item(), grad_loss.item(), normals_loss.item(), f_dist.item(), g_dist.item()))
                print('fg diff loss: {:.6f}'.format(fg_diff_dist.item()))
                print('Repulsion Loss: {:.6f}'.format(non_feature_repulsion_loss.item()))
                print('verticle_loss: {:.6f}'.format(verticle_loss.item()))
                if args.laplace:
                    print('laplacian f & g: {:.6f}\t {:.6f}'.format(f_laplacian, g_laplacian))
                print('a: {:.6f}\t beta_3: {:.6f}'.format(a, beta_3_init))

    def run_multi_branch(self):

        print("running")

        self.data = self.data.cuda()
        self.data.requires_grad_()
        feature_mask_cpu = self.feature_mask.numpy()
        self.feature_mask = self.feature_mask.cuda()

        n_patch_batch = 2048
        n_feature_batch = 2048
        print("patch batch & feature batch: ", n_patch_batch, n_feature_batch)
        n_branch = int(torch.max(self.feature_mask).item())
        print('number of branches: ', n_branch)
        n_batchsize = n_feature_batch + n_branch * n_patch_batch
        #first non feature then feature elements
        assignment_weight = 1.0
        omega_1 = 10.0
        omega_2 = 10.0
        omega_3 = 10.0
        omega_max = 1000.0
        a = 0.01
        a_max = 10.0
        beta_1 = 10
        beta_2 = 1
        eps = 0.01
        # beta_3_init = 0.02
        beta_3_init = 0.0
        beta_3_max = 0.2

        
        # print ("feature mask shape: ", feature_mask_cpu.shape)
        feature_id = np.where(feature_mask_cpu == 0)
        print ("feature id shape: ", feature_id[0].shape)
        n_feature = feature_id[0].shape[0]
        print ("n feature : ", n_feature)
        print ("n feature tyep: ", type(n_feature))
        print ("feature batch type: ", type(n_feature_batch))
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
            for i in range(n_branch + 1):
                self.network.flag_output = i + 1
                self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            return

        print("training")

        if args.summary == True:
            writer = SummaryWriter(os.path.join("runs", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1
        a = pow(10, pow_index) * a
        beta_3_init = pow(10, pow_index) * beta_3_init
        if a > a_max:
            a = a_max
        if beta_3_init > beta_3_max:
            beta_3_init = beta_3_max

        print('summary status: ', args.summary)


        for epoch in range(self.startepoch, self.nepochs + 1):
            #indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            #first feature part then non-feature part
            indices = torch.tensor(feature_id[0][np.random.choice(n_feature, n_feature_batch, True)]).cuda()
            #current feature mask
            cur_feature_mask = torch.zeros(n_branch * n_patch_batch + n_feature_batch).cuda()
            cur_feature_mask[0: n_feature_batch] = 1.0
            branch_mask = torch.zeros(n_branch, n_branch * n_patch_batch + n_feature_batch).cuda()
            for i in range(n_branch):
                indices_nonfeature = torch.tensor(patch_id[i][np.random.choice(patch_id_n[i], n_patch_batch, True)]).cuda()
                branch_mask[i, n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch] = 1.0
                indices = torch.cat((indices, indices_nonfeature), 0)

            cur_data = self.data[indices]
            mnfld_pnts = cur_data[:, :self.d_in] #n_indices x 3
            mnfld_sigma = self.local_sigma[indices] #noise points

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                self.save_checkpoints(epoch)
                if a < a_max:
                    a = a * 10
                if beta_3_init < beta_3_max:
                    beta_3_init = beta_3_init * 10

            if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                for i in range(n_branch + 1):
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

            # use non feature points
            indices_nonfeature = indices[n_feature_batch:]
            non_feature_pnts = self.data[indices_nonfeature][:, :self.d_in]
            non_feature_sigma = self.local_sigma[indices_nonfeature]
            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pred_all.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]
            #print shape
            # print("mnfld_pnts size: ")
            # print(mnfld_pnts.size())
            # print("mnfld_pred size: ")
            # print(mnfld_pred.size())

            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss
            mnfld_loss = (mnfld_pred.abs()).mean()
            # print ("mnfld loss device: ", mnfld_loss.device)
            loss = loss + mnfld_loss
            # manifold loss for patches
            mnfld_loss_patch = 0.0
            for i in range(n_branch):
                mnfld_loss_patch = mnfld_loss_patch + (mnfld_pred_all[:,i + 1].abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            loss = loss + mnfld_loss_patch

            assignment_loss = torch.zeros(1)
            if assignment_weight > 0.0: 
                for i in range(n_branch):
                    assignment_loss = assignment_loss + ((mnfld_pred - mnfld_pred_all[:,i + 1]).abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            loss = loss + assignment_weight * assignment_loss

            # eikonal loss
            grad_loss = 0.0
            for i in range(n_branch):
                single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,i + 1])
                grad_loss = grad_loss + ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss
            # normals loss
            if self.with_normals:
                #all normals
                normals = cur_data[:, -self.d_in:]
                #defined for h
                # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                normals_loss = 0.0
                all_fi = torch.zeros(n_batchsize, 1).cuda()
                for i in range(n_branch):
                    all_fi[n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch, 0] = mnfld_pred_all[n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch, i + 1]

                branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

                # for i in range(n_branch):
                    # branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                    # normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                loss = loss + self.normals_lambda * normals_loss
            else:
                normals_loss = torch.zeros(1)

            # feature loss
            feature_loss = (mnfld_pred.abs() * cur_feature_mask).mean() * n_batchsize / n_feature_batch
            #exp version
            # f_loss = (torch.exp(a * mnfld_pred_all[:, 1].abs()) * cur_feature_mask).mean() * n_feature_split
            # g_loss = (torch.exp(a * mnfld_pred_all[:, 2].abs()) * cur_feature_mask).mean() * n_feature_split
            # fg_diff_loss = (torch.exp(a * (mnfld_pred_all[:, 1] - mnfld_pred_all[:, 2]).abs()) * cur_feature_mask).mean() * n_feature_split
            # loss = loss + omega_1 * f_loss + omega_2 * g_loss + omega_3 * fg_diff_loss

            #linear version
            loss = loss + omega_1 * feature_loss

            # # non-feature point penalty loss
            # # non_feature_point_penalty = omega_3 * (torch.exp( -a * torch.abs(mnfld_pred_all[:, 2])) * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
            # # loss = loss + non_feature_point_penalty
            # # back propagation

            # #non feature repulstion loss
            # # non_feature_repulsion_loss_1 = ((beta_1 * mnfld_pred_all[:, 1].abs() * mnfld_pred_all[:, 2].abs()) * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
            # # non_feature_repulsion_loss_2 = ((- beta_2 * torch.log(eps + mnfld_pred_all[:, 1].abs() + mnfld_pred_all[:, 2].abs())) * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
            # non_feature_repulsion_loss = (torch.min(mnfld_pred_all[:, 1].abs(), mnfld_pred_all[:, 2].abs())/(mnfld_pred_all[:, 1].abs() + mnfld_pred_all[:, 2].abs() + eps) * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
            # # non_feature_repulsion_loss_2 = (1.0 / ((mnfld_pred_all[:, 1] - mnfld_pred_all[:, 2]).abs() + eps)  * (1 - cur_feature_mask)).mean() * (n_feature_split / (n_feature_split - 1))
            # loss = loss + beta_1 * non_feature_repulsion_loss

            # #verticle loss
            # # feature points
            # # f_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 1])
            # # g_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 2])
            # # verticle_loss = (torch.abs(g_grad[:, 0] * f_grad[:, 0] + g_grad[:, 1] * f_grad[:, 1] + g_grad[:, 2] * f_grad[:, 2]) * cur_feature_mask).mean() * n_feature_split
            # # all points
            # f_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:, 1])
            # g_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:, 2])
            # verticle_loss = (torch.abs(g_grad[:, 0] * f_grad[:, 0] + g_grad[:, 1] * f_grad[:, 1] + g_grad[:, 2] * f_grad[:, 2])).mean()
            # loss = loss + beta_2 * verticle_loss

            # #smoothness loss
            # if args.laplace: 
            #     f_gradx_grad = gradient(nonmnfld_pnts, f_grad[:, 0])
            #     f_grady_grad = gradient(nonmnfld_pnts, f_grad[:, 1])
            #     f_gradz_grad = gradient(nonmnfld_pnts, f_grad[:, 2])

            #     g_gradx_grad = gradient(nonmnfld_pnts, g_grad[:, 0])
            #     g_grady_grad = gradient(nonmnfld_pnts, g_grad[:, 1])
            #     g_gradz_grad = gradient(nonmnfld_pnts, g_grad[:, 2])

            #     f_laplacian = (torch.abs(f_gradx_grad[:, 0] + f_grady_grad[:, 1] + f_gradz_grad[:, 2])).mean()
            #     g_laplacian = (torch.abs(g_gradx_grad[:, 0] + g_grady_grad[:, 1] + g_gradz_grad[:, 2])).mean()
            #     # f_laplacian = 0
            #     # g_laplacian = 0
            #     loss = loss + beta_3_init * (f_laplacian + g_laplacian)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            #tensorboard
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Grad loss F',grad_loss.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss', normals_loss.item(), epoch)
                writer.add_scalar('Loss/feature loss', feature_loss.item(), epoch)
                writer.add_scalar('Loss/assignment_loss', assignment_loss.item(), epoch)
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
                    '\tManifold patch loss: {:.6f}\t grad loss: {:.6f}\t normals loss: {:.6f}'.format(
                    epoch, self.nepochs, 100. * epoch / self.nepochs,
                    loss.item(), mnfld_loss.item(), mnfld_loss_patch.item(), grad_loss.item(), normals_loss.item()))
                print ('feature loss: {:.6f}\t assignment_loss {:.6f}\t'.format(feature_loss.item(), assignment_loss.item()))

    def run_multi_branch_mlp(self):
        print("running")
        self.data = self.data.cuda()
        self.data.requires_grad_()
        feature_mask_cpu = self.feature_mask.numpy()
        self.feature_mask = self.feature_mask.cuda()
        n_patch_batch = 2048
        n_feature_batch = 2048
        print("patch batch & feature batch: ", n_patch_batch, n_feature_batch)
        n_branch = int(torch.max(self.feature_mask).item())
        print('number of branches: ', n_branch)
        n_batchsize = n_feature_batch + n_branch * n_patch_batch
        #first non feature then feature elements
        assignment_weight = 0.0
        omega_1 = 10.0
        omega_2 = 10.0
        omega_3 = 10.0
        omega_max = 1000.0
        a = 0.01
        a_max = 10.0
        beta_1 = 10
        beta_2 = 1
        eps = 0.01
        # beta_3_init = 0.02
        beta_3_init = 0.0
        beta_3_max = 0.2

        
        # print ("feature mask shape: ", feature_mask_cpu.shape)
        feature_id = np.where(feature_mask_cpu == 0)
        print ("feature id shape: ", feature_id[0].shape)
        n_feature = feature_id[0].shape[0]
        print ("n feature : ", n_feature)
        print ("n feature tyep: ", type(n_feature))
        print ("feature batch type: ", type(n_feature_batch))
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
            for i in range(n_branch + 1):
                self.network.flag_output = i + 1
                self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            return

        print("training")

        if args.summary == True:
            writer = SummaryWriter(os.path.join("runs", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1
        a = pow(10, pow_index) * a
        beta_3_init = pow(10, pow_index) * beta_3_init
        if a > a_max:
            a = a_max
        if beta_3_init > beta_3_max:
            beta_3_init = beta_3_max

        print('summary status: ', args.summary)
        for epoch in range(self.startepoch, self.nepochs + 1):
            #indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            #first feature part then non-feature part
            indices = torch.tensor(feature_id[0][np.random.choice(n_feature, n_feature_batch, True)]).cuda()
            #current feature mask
            cur_feature_mask = torch.zeros(n_branch * n_patch_batch + n_feature_batch).cuda()
            cur_feature_mask[0: n_feature_batch] = 1.0
            branch_mask = torch.zeros(n_branch, n_branch * n_patch_batch + n_feature_batch).cuda()
            for i in range(n_branch):
                indices_nonfeature = torch.tensor(patch_id[i][np.random.choice(patch_id_n[i], n_patch_batch, True)]).cuda()
                branch_mask[i, n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch] = 1.0
                indices = torch.cat((indices, indices_nonfeature), 0)

            cur_data = self.data[indices]
            mnfld_pnts = cur_data[:, :self.d_in] #n_indices x 3
            mnfld_sigma = self.local_sigma[indices] #noise points

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                self.save_checkpoints(epoch)
                if a < a_max:
                    a = a * 10
                if beta_3_init < beta_3_max:
                    beta_3_init = beta_3_init * 10

            if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                for i in range(2 * n_branch + 1):
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

            # use non feature points
            indices_nonfeature = indices[n_feature_batch:]
            non_feature_pnts = self.data[indices_nonfeature][:, :self.d_in]
            non_feature_sigma = self.local_sigma[indices_nonfeature]
            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]
            #print shape
            # print("mnfld_pnts size: ")
            # print(mnfld_pnts.size())
            # print("mnfld_pred size: ")
            # print(mnfld_pred.size())

            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss
            mnfld_loss = (mnfld_pred.abs()).mean()
            # print ("mnfld loss device: ", mnfld_loss.device)
            loss = loss + mnfld_loss
            # manifold loss for patches
            mnfld_loss_patch = 0.0
            for i in range(n_branch):
                mnfld_loss_patch = mnfld_loss_patch + (mnfld_pred_all[:,i + 1].abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            loss = loss + mnfld_loss_patch

            assignment_loss = torch.zeros(1).cuda()
            if assignment_weight > 0.0: 
                for i in range(n_branch):
                    assignment_loss = assignment_loss + ((mnfld_pred - mnfld_pred_all[:,i + 1]).abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            loss = loss + assignment_weight * assignment_loss

            # eikonal loss
            grad_loss = 0.0
            for i in range(n_branch):
                single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,i + 1])
                grad_loss = grad_loss + ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss
            # eikonal loss for h
            single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,0])
            grad_loss_h = ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss_h

            # normals loss
            if self.with_normals:
                #all normals
                normals = cur_data[:, -self.d_in:]
                #defined for h
                # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                normals_loss = 0.0
                for i in range(n_branch):
                    branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                    normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                loss = loss + self.normals_lambda * normals_loss

                mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

                loss = loss + self.normals_lambda * normals_loss_h

            else:
                normals_loss = torch.zeros(1)

            # feature loss
            feature_loss = (mnfld_pred.abs() * cur_feature_mask).mean() * n_batchsize / n_feature_batch

            #linear version
            loss = loss + omega_1 * feature_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            #tensorboard
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss all', normals_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss h', normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/feature loss', feature_loss.item(), epoch)
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
                print ('feature loss: {:.6f}\t assignment_loss {:.6f}\t'.format(feature_loss.item(), assignment_loss.item()))               

    def run_multi_branch_mask(self):
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
        n_patch_batch = n_batchsize // (n_branch + 1)
        n_feature_batch = n_batchsize - n_patch_batch * n_branch
        print("patch batch & feature batch: ", n_patch_batch, n_feature_batch)
        print('number of branches: ', n_branch)
        

        #first non feature then feature elements
        assignment_weight = 0.0
        omega_1 = 0.0
        omega_2 = 1.0
        omega_3 = 0.0
        omega_max = 1000.0
        a = 0.01
        a_max = 10.0
        beta_1 = 10
        beta_2 = 1
        eps = 0.01
        # beta_3_init = 0.02
        beta_3_init = 0.0
        beta_3_max = 0.2
        k = 2 #no breaking
        k_max = 12

        # print ("feature mask shape: ", feature_mask_cpu.shape)
        feature_id = np.where(feature_mask_cpu == 0)
        print ("feature id shape: ", feature_id[0].shape)
        n_feature = feature_id[0].shape[0]
        print ("n feature : ", n_feature)
        print ("n feature tyep: ", type(n_feature))
        print ("feature batch type: ", type(n_feature_batch))
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
            for i in range(n_branch + 1):
                self.network.flag_output = i + 1
                self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_mask" + str(i))

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
            writer = SummaryWriter(os.path.join("runs", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1
        a = pow(10, pow_index) * a
        beta_3_init = pow(10, pow_index) * beta_3_init
        if a > a_max:
            a = a_max
        if beta_3_init > beta_3_max:
            beta_3_init = beta_3_max

        print('summary status: ', args.summary)
        # branch mask is predefined
        branch_mask = torch.zeros(n_branch, n_branch * n_patch_batch + n_feature_batch).cuda()
        single_branch_mask_gt = torch.zeros(n_branch * n_patch_batch + n_feature_batch, n_branch).cuda()
        #for cross entropy loss, id start from 0
        single_branch_mask_id = torch.zeros([n_branch * n_patch_batch], dtype = torch.long).cuda()
        for i in range(n_branch):
            branch_mask[i, n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch] = 1.0
            single_branch_mask_gt[n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch, i] = 1.0
            single_branch_mask_id[i * n_patch_batch : (i + 1) * n_patch_batch] = i

        for epoch in range(self.startepoch, self.nepochs + 1):
            #indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            #first feature part then non-feature part
            indices = torch.tensor(feature_id[0][np.random.choice(n_feature, n_feature_batch, True)]).cuda()
            #current feature mask
            cur_feature_mask = torch.zeros(n_branch * n_patch_batch + n_feature_batch).cuda()
            cur_feature_mask[0: n_feature_batch] = 1.0
            for i in range(n_branch):
                indices_nonfeature = torch.tensor(patch_id[i][np.random.choice(patch_id_n[i], n_patch_batch, True)]).cuda()
                indices = torch.cat((indices, indices_nonfeature), 0)

            cur_data = self.data[indices]
            mnfld_pnts = cur_data[:, :self.d_in] #n_indices x 3
            mnfld_sigma = self.local_sigma[indices] #noise points

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                self.save_checkpoints(epoch)
                if a < a_max:
                    a = a * 10
                if beta_3_init < beta_3_max:
                    beta_3_init = beta_3_init * 10

            if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
            # if epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                for i in range(n_branch + 1):
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

            # use non feature points
            indices_nonfeature = indices[n_feature_batch:]
            non_feature_pnts = self.data[indices_nonfeature][:, :self.d_in]
            non_feature_sigma = self.local_sigma[indices_nonfeature]
            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]
            #print shape
            # print("mnfld_pnts size: ")
            # print(mnfld_pnts.size())
            # print("mnfld_pred size: ")
            # print(mnfld_pred.size())

            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss
            mnfld_loss = (mnfld_pred.abs()).mean()
            # print ("mnfld loss device: ", mnfld_loss.device)
            loss = loss + mnfld_loss
            # manifold loss for patches

            all_fi = torch.zeros(n_batchsize, 1).cuda()
            for i in range(n_branch):
                all_fi[n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch, 0] = mnfld_pred_all[n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch, i + 1]

            # mnfld_loss_patch = 0.0
            # for i in range(n_branch):
                # mnfld_loss_patch = mnfld_loss_patch + (mnfld_pred_all[:,i + 1].abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            mnfld_loss_patch = (all_fi[:,0].abs() * (1 - cur_feature_mask)).mean() * n_batchsize / (n_batchsize - n_feature_batch) 
            loss = loss + mnfld_loss_patch

            assignment_loss = torch.zeros(1).cuda()
            if assignment_weight > 0.0: 
                # for i in range(n_branch):
                    # assignment_loss = assignment_loss + ((mnfld_pred - mnfld_pred_all[:,i + 1]).abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
                assignment_loss = ((mnfld_pred - all_fi).abs() * (1 - cur_feature_mask)).mean() * n_batchsize / (n_batchsize - n_feature_batch)
                
            loss = loss + assignment_weight * assignment_loss

            # eikonal loss for all
            grad_loss = torch.zeros(1).cuda()
            # for i in range(n_branch):
            #     single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,i + 1])
            #     grad_loss = grad_loss + ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            # loss = loss + self.grad_lambda * grad_loss
            # eikonal loss for h
            single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,0])
            grad_loss_h = ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss_h

            # normals loss
            if self.with_normals:
                #all normals
                normals = cur_data[:, -self.d_in:]
                #defined for h
                # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                # normals_loss = 0.0
                # for i in range(n_branch):
                #     branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                #     normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                loss = loss + self.normals_lambda * normals_loss

                mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

                loss = loss + self.normals_lambda * normals_loss_h

            else:
                normals_loss = torch.zeros(1)
                normals_loss_h = torch.zeros(1)

            # feature loss
            feature_loss = (mnfld_pred.abs() * cur_feature_mask).mean() * n_batchsize / n_feature_batch

            #linear version
            loss = loss + omega_1 * feature_loss

            #mask loss
            mask_feature = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            #vanilla version
            # for i in range(k - 1):
            #     mask_feature = mask_feature * mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]

            # mask_loss = torch.abs(torch.sum(mask_feature, dim = 1) - 1).mean()
            # mask_loss = (torch.sum((mask_feature - single_branch_mask_gt).abs(), dim = 1) * (1 - cur_feature_mask) ).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

            #cross entropy form
            ce_loss = nn.CrossEntropyLoss()
            mask_loss = ce_loss(mask_feature[n_feature_batch:,:], single_branch_mask_id)

            #original version:
            loss = loss + omega_2 * mask_loss

            #mask only version:
            # loss = mask_loss


            # mask_loss expectation
            # mask_nonmfd = nonmnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            # mask_expectation = torch.abs(torch.sum(mask_nonmfd * mask_nonmfd, dim = 1) - 1).mean()
            # loss = loss + omega_3 * mask_expectation


            if epoch % 1000 == 0:
                k = 2 * k
                if k > k_max:
                    k = k_max

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            #tensorboard
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',self.grad_lambda * grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',self.grad_lambda * grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss all', self.normals_lambda * normals_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss h', self.normals_lambda * normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/feature loss', omega_1 * feature_loss.item(), epoch)
                writer.add_scalar('Loss/mask loss', omega_2 * mask_loss.item(), epoch)
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
                print ('feature loss: {:.6f}\t assignment_loss {:.6f}\t'.format(feature_loss.item(), assignment_loss.item()))               
                print ('k: ', k, " mask loss: ", mask_loss.item())

    def run_multi_branch_mask_nofea(self):
        #to be changed
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
        assignment_weight = 0.0
        omega_1 = 0.0
        omega_2 = 1.0
        # omega_3 = 0.01
        omega_3 = 1
        omega_4 = 0
        omega3_max = 10.0
        a = 0.01
        a_max = 10.0
        beta_1 = 10
        beta_2 = 1
        eps = 0.01
        # beta_3_init = 0.02
        beta_3_init = 0.0
        beta_3_max = 0.2
        k = 2 
        k_max = 10

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
            #     self.network.flag_output = i + 1
            #     self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_mask" + str(i))

            # self.network.flag_onehot = True
            # for i in range(n_branch + 1):
            #     self.network.flag_output = i + 1
            #     self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i) + "_onehot", with_cuts = True)
            # self.network.flag_output = 0
            # # self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_onehotmask" + str(i))
            # self.network.flag_onehot = False
            return
        if self.evaldist:
            print ("evaluating mesh and normal distance")
            #only for h
            avg_dist = 0.0
            avg_normal_dist = 0.0
            self.network.eval()
            for i, cur_data in enumerate(torch.split(self.data, self.points_batch)):
                mnfld_pnts = cur_data[:, :self.d_in]
                mnfld_pred_all = self.network(mnfld_pnts)
                mnfld_pred = mnfld_pred_all[:,0]
                #print shape
                mnfld_loss = (mnfld_pred.abs()).mean()
                # normals loss
                if self.with_normals:
                    #all normals
                    normals = cur_data[:, -self.d_in:]
                    #defined for h
                    # # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                    # branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                    # normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1)).mean()
                    # # normals_loss = 0.0
                    # # for i in range(n_branch):
                    # #     branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                    # #     normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                    # loss = loss + self.normals_lambda * normals_loss

                    mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                    normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()
                else:
                    # normals_loss = torch.zeros(1)
                    normals_loss_h = torch.zeros(1)
                avg_dist = avg_dist + mnfld_loss * mnfld_pnts.shape[0]
                avg_normal_dist = avg_normal_dist + normals_loss_h * mnfld_pnts.shape[0]
                print ("batch loss: {} mnfld:{:.6f}\t normal:{:.6f}\n".format(i, mnfld_loss.item(), normals_loss_h.item()))
            avg_dist = avg_dist / self.data.shape[0]
            avg_normal_dist = avg_normal_dist / self.data.shape[0]
            print ("Overall loss: mnfld:{:.6f}\t normal:{:.6f}\n".format(avg_dist, avg_normal_dist))
            return
        if args.vismask:
            print("visualize mask for point cloud") 
            branch_color = []
            if n_branch == 1:
                branch_color.append(cm.plasma(0.0))
            else:
                for i in range(n_branch):
                    # branch_color.append(cm.hot(i / n_branch))
                    branch_color.append(cm.plasma(i / (n_branch - 1)))
            branch_color = np.concatenate(branch_color, axis = 0)
            branch_color = branch_color.reshape(-1,4)
            branch_color = branch_color[:,:3] * 255
            vertscolor = np.zeros([self.data.shape[0], 3])
            vertscolor_onehot = np.zeros([self.data.shape[0], 3])
            count = 0
            verts_error = np.zeros([self.data.shape[0], 1])
            verts_error_onehot = np.zeros([self.data.shape[0], 1])
            verts_bgt = np.zeros([self.data.shape[0], n_branch])
            verts_bgt[np.arange(self.data.shape[0]), feature_mask_cpu.astype(int) - 1] = 1.0

            self.network.flag_softmax = True
            for i, cur_data in enumerate(torch.split(self.data, self.points_batch)):
                mnfld_pnts = cur_data[:, :self.d_in]
                mnfld_pred_all = self.network(mnfld_pnts)
                cur_data_length = cur_data.shape[0]
                mask_feature_np = mnfld_pred_all.detach()[:, n_branch + 1: 2 * n_branch + 1].cpu().numpy()
                maxid = np.argmax(mask_feature_np, 1)
                mask_feature_onehot = np.zeros_like(mask_feature_np)
                mask_feature_onehot[np.arange(mask_feature_np.shape[0]), maxid] = 1.0

                # verts_error[count:count + cur_data_length, :] =  np.expand_dims(np.sum(np.abs(mask_feature_np - mask_feature_onehot), 1),1)
                verts_error[count:count + cur_data_length, :] =  np.expand_dims(LA.norm((mask_feature_np - verts_bgt[count:count + cur_data_length, :]), axis = 1),1)
                verts_error_onehot[count:count + cur_data_length, :] =  np.expand_dims(LA.norm((mask_feature_onehot - verts_bgt[count:count + cur_data_length, :]), axis = 1),1)
                vertscolor[count:count + cur_data_length, :] = np.matmul(mask_feature_np, branch_color)
                vertscolor_onehot[count:count + cur_data_length, :] = np.matmul(mask_feature_onehot, branch_color)
                count = count + self.points_batch
            vertscolor[vertscolor > 255] = 255
            vertscolor = vertscolor.astype(int)
            vertscolor_onehot[vertscolor_onehot > 255] = 255
            vertscolor_onehot = vertscolor_onehot.astype(int)

            print ("sum: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error), np.max(verts_error), np.min(verts_error)))
            print ("hot: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error_onehot), np.max(verts_error_onehot), np.min(verts_error_onehot)))
            diff = np.max(verts_error) - np.min(verts_error)
            verts_error = (verts_error - np.min(verts_error)) / diff
            verts_error_color = np.matmul(verts_error, np.array([[255,255,255]])).astype(int)

            diff_onehot = np.max(verts_error_onehot) - np.min(verts_error_onehot)
            verts_error_onehot = (verts_error_onehot - np.min(verts_error_onehot)) / diff_onehot
            verts_error_onehot_color = np.matmul(verts_error_onehot, np.array([[255,255,255]])).astype(int)

            if args.test:
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor, self.foldername +"_mask.ply")
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor_onehot, self.foldername + "_mask_onehot.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_color, self.foldername + "_mask_error.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_onehot_color, self.foldername + "_mask_error_onehot.ply")
            else:
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor, self.foldername +"_mask_train.ply")   
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor_onehot, self.foldername + "_mask_onehot_train.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_color, self.foldername + "_mask_error_train.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_onehot_color, self.foldername + "_mask_error_train_onehot.ply")
            return

        print("training")

        if args.summary == True:
            writer = SummaryWriter(os.path.join("nofea", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1
        a = pow(10, pow_index) * a
        beta_3_init = pow(10, pow_index) * beta_3_init
        if a > a_max:
            a = a_max
        if beta_3_init > beta_3_max:
            beta_3_init = beta_3_max

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
                if a < a_max:
                    a = a * 10
                if beta_3_init < beta_3_max:
                    beta_3_init = beta_3_init * 10

            if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
            # if epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                for i in range(n_branch + 1):
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

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]
            #print shape
            # print("mnfld_pnts size: ")
            # print(mnfld_pnts.size())
            # print("mnfld_pred size: ")
            # print(mnfld_pred.size())

            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss
            mnfld_loss = (mnfld_pred.abs()).mean()
            # print ("mnfld loss device: ", mnfld_loss.device)
            loss = loss + mnfld_loss
            # manifold loss for patches

            all_fi = torch.zeros(n_batchsize, 1).cuda()
            for i in range(n_branch - 1):
                all_fi[i * n_patch_batch : (i + 1) * n_patch_batch, 0] = mnfld_pred_all[i * n_patch_batch : (i + 1) * n_patch_batch, i + 1]
            #last patch
            all_fi[(n_branch - 1) * n_patch_batch:, 0] = mnfld_pred_all[(n_branch - 1) * n_patch_batch:, n_branch]

            # mnfld_loss_patch = 0.0
            # for i in range(n_branch):
                # mnfld_loss_patch = mnfld_loss_patch + (mnfld_pred_all[:,i + 1].abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            mnfld_loss_patch = all_fi[:,0].abs().mean()
            loss = loss + mnfld_loss_patch

            assignment_loss = torch.zeros(1).cuda()
            if assignment_weight > 0.0: 
                # for i in range(n_branch):
                    # assignment_loss = assignment_loss + ((mnfld_pred - mnfld_pred_all[:,i + 1]).abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
                assignment_loss = ((mnfld_pred - all_fi).abs()).mean()
                
            loss = loss + assignment_weight * assignment_loss

            # eikonal loss for all
            grad_loss = torch.zeros(1).cuda()
            # for i in range(n_branch):
            #     single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,i + 1])
            #     grad_loss = grad_loss + ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            # loss = loss + self.grad_lambda * grad_loss
            # eikonal loss for h
            single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,0])
            grad_loss_h = ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss_h

            # normals loss
            if self.with_normals:
                #all normals
                normals = cur_data[:, -self.d_in:]
                #defined for h
                # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1)).mean()
                # normals_loss = 0.0
                # for i in range(n_branch):
                #     branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                #     normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                loss = loss + self.normals_lambda * normals_loss

                mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()

                loss = loss + self.normals_lambda * normals_loss_h

            else:
                normals_loss = torch.zeros(1)
                normals_loss_h = torch.zeros(1)

            #mask loss
            mask_feature = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            #vanilla version
            # for i in range(k - 1):
            #     mask_feature = mask_feature * mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]

            # mask_loss = torch.abs(torch.sum(mask_feature, dim = 1) - 1).mean()
            # mask_loss = (torch.sum((mask_feature - single_branch_mask_gt).abs(), dim = 1) * (1 - cur_feature_mask) ).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

            #cross entropy form
            #flag_softmax set to False
            ce_loss = nn.CrossEntropyLoss()
            mask_loss = ce_loss(mask_feature, single_branch_mask_id)

            #dice version:
            # mask_loss = (mask_feature - single_branch_mask_gt).norm(2, dim=-1).mean()
            
            #focal loss
            # mask_loss = FocalLoss(gamma=5)(mask_feature, single_branch_mask_id)
            

            loss = loss + omega_2 * mask_loss

            #mask only version:
            # loss = mask_loss


            # mask_loss expectation
            mask_expectation = torch.zeros(1).cuda()
            if omega_3 > 0.0:
                mask_nonmfd = nonmnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
                sm_loss = nn.Softmax(dim=1)
                mask_nonmfd_sm = sm_loss(mask_nonmfd)
                mask_expectation = torch.abs(torch.sum(torch.pow(mask_nonmfd_sm, k), dim = 1) - 1).mean()
                #not added yet
                loss = loss + omega_3 * mask_expectation

            # #mask grad loss
            mask_grad_loss = torch.zeros(1).cuda()
            if omega_4 > 0.0:
                mask_nonmfd = nonmnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
                # sm_loss = nn.Softmax(dim=1)
                # mask_nonmfd_sm = sm_loss(mask_nonmfd)
                # # mask_expectation = torch.abs(torch.sum(torch.pow(mask_nonmfd_sm, k), dim = 1) - 1).mean()
                # # #not added yet
                # loss = loss + omega_3 * mask_expectation

                mask_grad_loss = torch.zeros(1).cuda()
                mask_grad = gradient(nonmnfld_pnts, mask_nonmfd)
                mask_grad_loss = mask_grad.norm(1,dim=-1).mean()
                loss = loss + omega_4 * mask_grad_loss


            if epoch % 2000 == 0:
                k = k + 2
                if k > k_max:
                    k = k_max
            #     omega_3 = omega_3 * 10
            #     if omega_3 > omega3_max:
            #         omega_3 = omega3_max

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            #tensorboard
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',self.grad_lambda * grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',self.grad_lambda * grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss all', self.normals_lambda * normals_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss h', self.normals_lambda * normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/mask loss', omega_2 * mask_loss.item(), epoch)
                writer.add_scalar('Loss/mask expectation', omega_3 * mask_expectation.item(), epoch)
                writer.add_scalar('Loss/mask grad', omega_4 * mask_grad_loss.item(), epoch)
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
                print ('assignment_loss {:.6f}\t'.format(assignment_loss.item()))               
                print ('k: ', k, " mask loss: ", mask_loss.item())
                print ('mask expectation: ', mask_expectation.item())
                print ('mask grad: ', mask_grad_loss.item())

    def run_multi_branch_mask_nofea_svm(self):
        #to be changed
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
        assignment_weight = 0.0
        omega_1 = 1.0
        omega_2 = 1.0
        c_svm = args.csvm
        # omega_3 = 0.01
        a = 0.01
        a_max = 10.0
        beta_1 = 10
        beta_2 = 1
        eps = 0.01
        # beta_3_init = 0.02
        beta_3_init = 0.0
        beta_3_max = 0.2
        k = 2 
        k_max = 10

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

            ##save patch sdf
            # self.network.flag_softmax = True
            # for i in range(n_branch + 1):
            #     self.network.flag_output = i + 1
            #     self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_mask" + str(i))

            # self.network.flag_onehot = True
            # for i in range(n_branch + 1):
            #     self.network.flag_output = i + 1
            #     self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i) + "_onehot", with_cuts = True)
            # self.network.flag_output = 0
            # # self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_onehotmask" + str(i))
            # self.network.flag_onehot = False
            return
        if self.evaldist:
            print ("evaluating mesh and normal distance")
            #only for h
            avg_dist = 0.0
            avg_normal_dist = 0.0
            self.network.eval()
            for i, cur_data in enumerate(torch.split(self.data, self.points_batch)):
                mnfld_pnts = cur_data[:, :self.d_in]
                mnfld_pred_all = self.network(mnfld_pnts)
                mnfld_pred = mnfld_pred_all[:,0]
                #print shape
                mnfld_loss = (mnfld_pred.abs()).mean()
                # normals loss
                if self.with_normals:
                    #all normals
                    normals = cur_data[:, -self.d_in:]
                    #defined for h
                    # # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                    # branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                    # normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1)).mean()
                    # # normals_loss = 0.0
                    # # for i in range(n_branch):
                    # #     branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                    # #     normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                    # loss = loss + self.normals_lambda * normals_loss

                    mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                    normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()
                else:
                    # normals_loss = torch.zeros(1)
                    normals_loss_h = torch.zeros(1)
                avg_dist = avg_dist + mnfld_loss * mnfld_pnts.shape[0]
                avg_normal_dist = avg_normal_dist + normals_loss_h * mnfld_pnts.shape[0]
                print ("batch loss: {} mnfld:{:.6f}\t normal:{:.6f}\n".format(i, mnfld_loss.item(), normals_loss_h.item()))
            avg_dist = avg_dist / self.data.shape[0]
            avg_normal_dist = avg_normal_dist / self.data.shape[0]
            print ("Overall loss: mnfld:{:.6f}\t normal:{:.6f}\n".format(avg_dist, avg_normal_dist))
            return
        if args.vismask:
            print("visualize mask for point cloud") 
            branch_color = []
            if n_branch == 1:
                branch_color.append(cm.plasma(0.0))
            else:
                for i in range(n_branch):
                    # branch_color.append(cm.hot(i / n_branch))
                    branch_color.append(cm.plasma(i / (n_branch - 1)))
            branch_color = np.concatenate(branch_color, axis = 0)
            branch_color = branch_color.reshape(-1,4)
            branch_color = branch_color[:,:3] * 255
            vertscolor = np.zeros([self.data.shape[0], 3])
            vertscolor_onehot = np.zeros([self.data.shape[0], 3])
            count = 0
            verts_error = np.zeros([self.data.shape[0], 1])
            verts_error_onehot = np.zeros([self.data.shape[0], 1])
            verts_bgt = np.zeros([self.data.shape[0], n_branch])
            verts_bgt[np.arange(self.data.shape[0]), feature_mask_cpu.astype(int) - 1] = 1.0

            self.network.flag_softmax = True
            for i, cur_data in enumerate(torch.split(self.data, self.points_batch)):
                mnfld_pnts = cur_data[:, :self.d_in]
                mnfld_pred_all = self.network(mnfld_pnts)
                cur_data_length = cur_data.shape[0]
                mask_feature_np = mnfld_pred_all.detach()[:, n_branch + 1: 2 * n_branch + 1].cpu().numpy()
                maxid = np.argmax(mask_feature_np, 1)
                mask_feature_onehot = np.zeros_like(mask_feature_np)
                mask_feature_onehot[np.arange(mask_feature_np.shape[0]), maxid] = 1.0

                # verts_error[count:count + cur_data_length, :] =  np.expand_dims(np.sum(np.abs(mask_feature_np - mask_feature_onehot), 1),1)
                verts_error[count:count + cur_data_length, :] =  np.expand_dims(LA.norm((mask_feature_np - verts_bgt[count:count + cur_data_length, :]), axis = 1),1)
                verts_error_onehot[count:count + cur_data_length, :] =  np.expand_dims(LA.norm((mask_feature_onehot - verts_bgt[count:count + cur_data_length, :]), axis = 1),1)
                vertscolor[count:count + cur_data_length, :] = np.matmul(mask_feature_np, branch_color)
                vertscolor_onehot[count:count + cur_data_length, :] = np.matmul(mask_feature_onehot, branch_color)
                count = count + self.points_batch
            vertscolor[vertscolor > 255] = 255
            vertscolor = vertscolor.astype(int)
            vertscolor_onehot[vertscolor_onehot > 255] = 255
            vertscolor_onehot = vertscolor_onehot.astype(int)

            print ("sum: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error), np.max(verts_error), np.min(verts_error)))
            print ("hot: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error_onehot), np.max(verts_error_onehot), np.min(verts_error_onehot)))
            diff = np.max(verts_error) - np.min(verts_error)
            verts_error = (verts_error - np.min(verts_error)) / diff
            verts_error_color = np.matmul(verts_error, np.array([[255,255,255]])).astype(int)

            diff_onehot = np.max(verts_error_onehot) - np.min(verts_error_onehot)
            verts_error_onehot = (verts_error_onehot - np.min(verts_error_onehot)) / diff_onehot
            verts_error_onehot_color = np.matmul(verts_error_onehot, np.array([[255,255,255]])).astype(int)

            if args.test:
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor, self.foldername +"_mask.ply")
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor_onehot, self.foldername + "_mask_onehot.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_color, self.foldername + "_mask_error.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_onehot_color, self.foldername + "_mask_error_onehot.ply")
            else:
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor, self.foldername +"_mask_train.ply")   
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor_onehot, self.foldername + "_mask_onehot_train.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_color, self.foldername + "_mask_error_train.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_onehot_color, self.foldername + "_mask_error_train_onehot.ply")
            return

        print("training")

        if args.summary == True:
            writer = SummaryWriter(os.path.join("nofea", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1
        a = pow(10, pow_index) * a
        beta_3_init = pow(10, pow_index) * beta_3_init
        if a > a_max:
            a = a_max
        if beta_3_init > beta_3_max:
            beta_3_init = beta_3_max

        print('summary status: ', args.summary)
        # branch mask is predefined
        branch_mask = torch.zeros(n_branch, n_batchsize).cuda()
        single_branch_mask_gt = torch.zeros(n_batchsize, n_branch).cuda()
        #for cross entropy loss, id start from 0
        single_branch_mask_id = torch.zeros([n_batchsize], dtype = torch.long).cuda()
        single_branch_mask_svm = -1 * torch.ones(n_batchsize, n_branch).cuda()

        for i in range(n_branch - 1):
            branch_mask[i, i * n_patch_batch : (i + 1) * n_patch_batch] = 1.0
            single_branch_mask_gt[i * n_patch_batch : (i + 1) * n_patch_batch, i] = 1.0
            single_branch_mask_svm[i * n_patch_batch : (i + 1) * n_patch_batch, i] = 1.0
            single_branch_mask_id[i * n_patch_batch : (i + 1) * n_patch_batch] = i

        #last patch
        branch_mask[n_branch - 1, (n_branch - 1) * n_patch_batch:] = 1.0
        single_branch_mask_gt[(n_branch - 1) * n_patch_batch:, (n_branch - 1)] = 1.0
        single_branch_mask_svm[(n_branch - 1) * n_patch_batch:, (n_branch - 1)] = 1.0
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
                if a < a_max:
                    a = a * 10
                if beta_3_init < beta_3_max:
                    beta_3_init = beta_3_init * 10

            if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
            # if epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                for i in range(n_branch + 1):
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

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]
            #print shape
            # print("mnfld_pnts size: ")
            # print(mnfld_pnts.size())
            # print("mnfld_pred size: ")
            # print(mnfld_pred.size())

            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss
            mnfld_loss = (mnfld_pred.abs()).mean()
            # print ("mnfld loss device: ", mnfld_loss.device)
            loss = loss + mnfld_loss
            # manifold loss for patches

            all_fi = torch.zeros(n_batchsize, 1).cuda()
            for i in range(n_branch - 1):
                all_fi[i * n_patch_batch : (i + 1) * n_patch_batch, 0] = mnfld_pred_all[i * n_patch_batch : (i + 1) * n_patch_batch, i + 1]
            #last patch
            all_fi[(n_branch - 1) * n_patch_batch:, 0] = mnfld_pred_all[(n_branch - 1) * n_patch_batch:, n_branch]

            # mnfld_loss_patch = 0.0
            # for i in range(n_branch):
                # mnfld_loss_patch = mnfld_loss_patch + (mnfld_pred_all[:,i + 1].abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            mnfld_loss_patch = all_fi[:,0].abs().mean()
            loss = loss + mnfld_loss_patch

            assignment_loss = torch.zeros(1).cuda()
            if assignment_weight > 0.0: 
                # for i in range(n_branch):
                    # assignment_loss = assignment_loss + ((mnfld_pred - mnfld_pred_all[:,i + 1]).abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
                assignment_loss = ((mnfld_pred - all_fi).abs()).mean()
                
            loss = loss + assignment_weight * assignment_loss

            # eikonal loss for all
            grad_loss = torch.zeros(1).cuda()
            # for i in range(n_branch):
            #     single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,i + 1])
            #     grad_loss = grad_loss + ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            # loss = loss + self.grad_lambda * grad_loss
            # eikonal loss for h
            single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,0])
            grad_loss_h = ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss_h

            # normals loss
            if self.with_normals:
                #all normals
                normals = cur_data[:, -self.d_in:]
                #defined for h
                # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1)).mean()
                # normals_loss = 0.0
                # for i in range(n_branch):
                #     branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                #     normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                loss = loss + self.normals_lambda * normals_loss

                mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()

                loss = loss + self.normals_lambda * normals_loss_h

            else:
                normals_loss = torch.zeros(1)
                normals_loss_h = torch.zeros(1)

            #mask loss
            # mask_feature = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            #vanilla version
            # for i in range(k - 1):
            #     mask_feature = mask_feature * mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]

            # mask_loss = torch.abs(torch.sum(mask_feature, dim = 1) - 1).mean()
            # mask_loss = (torch.sum((mask_feature - single_branch_mask_gt).abs(), dim = 1) * (1 - cur_feature_mask) ).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

            #mask only version:
            # loss = mask_loss
            # mask_predict = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            svm_dist = mnfld_pred_all[:, 2 * n_branch + 1: 3 * n_branch + 1]
            mask_loss = torch.zeros(1).cuda()
            for i in range(n_branch):
                weight = self.network.svms[i].weight.squeeze()
                mask_loss += weight.t() @weight / 2.0
            mask_loss = mask_loss / n_branch
            mask_loss += c_svm * torch.mean(torch.pow(torch.clamp(1 - svm_dist * single_branch_mask_svm, min=0), 2))

            loss = loss + omega_2 * mask_loss

            # # ce loss
            mask_celoss = torch.zeros(1).cuda()
            # ce_loss = nn.CrossEntropyLoss()
            # mask_celoss = ce_loss(5.0 * svm_dist, single_branch_mask_id)
            # loss = loss + omega_1 * mask_celoss
            
            # mask_loss expectation
            mask_expectation = torch.zeros(1).cuda()
            # mask_nonmfd = nonmnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            # if omega_3 > 0.0:
            #     sm_loss = nn.Softmax(dim=1)
            #     mask_nonmfd_sm = sm_loss(mask_nonmfd)
            #     mask_expectation = torch.abs(torch.sum(torch.pow(mask_nonmfd_sm, k), dim = 1) - 1).mean()
            #     # #not added yet
            #     loss = loss + omega_3 * mask_expectation

            mask_grad_loss = torch.zeros(1).cuda()
            # if omega_4 > 0.0:
            #     mask_grad = gradient(nonmnfld_pnts, mask_nonmfd)
            #     mask_grad_loss = mask_grad.norm(1,dim=-1).mean()
            #     loss = loss + omega_4 * mask_grad_loss


            # if epoch % 2000 == 0:
            #     k = k + 2
            #     if k > k_max:
            #         k = k_max
            #     omega_3 = omega_3 * 10
            #     if omega_3 > omega3_max:
            #         omega_3 = omega3_max

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            #tensorboard
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',self.grad_lambda * grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',self.grad_lambda * grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss all', self.normals_lambda * normals_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss h', self.normals_lambda * normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/mask loss', omega_2 * mask_loss.item(), epoch)
                writer.add_scalar('Loss/mask expectation', mask_expectation.item(), epoch)
                writer.add_scalar('Loss/mask grad', mask_celoss.item(), epoch)
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
                print ('assignment_loss {:.6f}\t'.format(assignment_loss.item()))               
                print ('k: ', k, " mask loss: ", mask_loss.item())
                print ('mask expectation: ', mask_expectation.item())
                print ('mask ce: ', mask_celoss.item())

    def run_multi_branch_mask_nofea_sdfonly(self):
        #to be changed
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
        assignment_weight = 0.0
        omega_1 = 0.0
        omega_2 = 1.0
        # omega_3 = 0.01
        omega_3 = 1
        omega3_max = 10.0
        a = 0.01
        a_max = 10.0
        beta_1 = 10
        beta_2 = 1
        eps = 0.01
        # beta_3_init = 0.02
        beta_3_init = 0.0
        beta_3_max = 0.2
        k = 2 
        k_max = 10

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
            for i in range(n_branch + 1):
                self.network.flag_output = i + 1
                self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_mask" + str(i))

            # self.network.flag_onehot = True
            # for i in range(n_branch + 1):
            #     self.network.flag_output = i + 1
            #     self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i) + "_onehot", with_cuts = True)
            # self.network.flag_output = 0
            # # self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_onehotmask" + str(i))
            # self.network.flag_onehot = False
            return
        if self.evaldist:
            print ("evaluating mesh and normal distance")
            #only for h
            avg_dist = 0.0
            avg_normal_dist = 0.0
            self.network.eval()
            for i, cur_data in enumerate(torch.split(self.data, self.points_batch)):
                mnfld_pnts = cur_data[:, :self.d_in]
                mnfld_pred_all = self.network(mnfld_pnts)
                mnfld_pred = mnfld_pred_all[:,0]
                #print shape
                mnfld_loss = (mnfld_pred.abs()).mean()
                # normals loss
                if self.with_normals:
                    #all normals
                    normals = cur_data[:, -self.d_in:]
                    #defined for h
                    # # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                    # branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                    # normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1)).mean()
                    # # normals_loss = 0.0
                    # # for i in range(n_branch):
                    # #     branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                    # #     normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                    # loss = loss + self.normals_lambda * normals_loss

                    mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                    normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()
                else:
                    # normals_loss = torch.zeros(1)
                    normals_loss_h = torch.zeros(1)
                avg_dist = avg_dist + mnfld_loss * mnfld_pnts.shape[0]
                avg_normal_dist = avg_normal_dist + normals_loss_h * mnfld_pnts.shape[0]
                print ("batch loss: {} mnfld:{:.6f}\t normal:{:.6f}\n".format(i, mnfld_loss.item(), normals_loss_h.item()))
            avg_dist = avg_dist / self.data.shape[0]
            avg_normal_dist = avg_normal_dist / self.data.shape[0]
            print ("Overall loss: mnfld:{:.6f}\t normal:{:.6f}\n".format(avg_dist, avg_normal_dist))
            return
        if args.vismask:
            print("visualize mask for point cloud") 
            branch_color = []
            if n_branch == 1:
                branch_color.append(cm.plasma(0.0))
            else:
                for i in range(n_branch):
                    # branch_color.append(cm.hot(i / n_branch))
                    branch_color.append(cm.plasma(i / (n_branch - 1)))
            branch_color = np.concatenate(branch_color, axis = 0)
            branch_color = branch_color.reshape(-1,4)
            branch_color = branch_color[:,:3] * 255
            vertscolor = np.zeros([self.data.shape[0], 3])
            vertscolor_onehot = np.zeros([self.data.shape[0], 3])
            count = 0
            verts_error = np.zeros([self.data.shape[0], 1])
            verts_error_onehot = np.zeros([self.data.shape[0], 1])
            verts_bgt = np.zeros([self.data.shape[0], n_branch])
            verts_bgt[np.arange(self.data.shape[0]), feature_mask_cpu.astype(int) - 1] = 1.0

            self.network.flag_softmax = True
            for i, cur_data in enumerate(torch.split(self.data, self.points_batch)):
                mnfld_pnts = cur_data[:, :self.d_in]
                mnfld_pred_all = self.network(mnfld_pnts)
                cur_data_length = cur_data.shape[0]
                mask_feature_np = mnfld_pred_all.detach()[:, n_branch + 1: 2 * n_branch + 1].cpu().numpy()
                maxid = np.argmax(mask_feature_np, 1)
                mask_feature_onehot = np.zeros_like(mask_feature_np)
                mask_feature_onehot[np.arange(mask_feature_np.shape[0]), maxid] = 1.0

                # verts_error[count:count + cur_data_length, :] =  np.expand_dims(np.sum(np.abs(mask_feature_np - mask_feature_onehot), 1),1)
                verts_error[count:count + cur_data_length, :] =  np.expand_dims(LA.norm((mask_feature_np - verts_bgt[count:count + cur_data_length, :]), axis = 1),1)
                verts_error_onehot[count:count + cur_data_length, :] =  np.expand_dims(LA.norm((mask_feature_onehot - verts_bgt[count:count + cur_data_length, :]), axis = 1),1)
                vertscolor[count:count + cur_data_length, :] = np.matmul(mask_feature_np, branch_color)
                vertscolor_onehot[count:count + cur_data_length, :] = np.matmul(mask_feature_onehot, branch_color)
                count = count + self.points_batch
            vertscolor[vertscolor > 255] = 255
            vertscolor = vertscolor.astype(int)
            vertscolor_onehot[vertscolor_onehot > 255] = 255
            vertscolor_onehot = vertscolor_onehot.astype(int)

            print ("sum: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error), np.max(verts_error), np.min(verts_error)))
            print ("hot: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error_onehot), np.max(verts_error_onehot), np.min(verts_error_onehot)))
            diff = np.max(verts_error) - np.min(verts_error)
            verts_error = (verts_error - np.min(verts_error)) / diff
            verts_error_color = np.matmul(verts_error, np.array([[255,255,255]])).astype(int)

            diff_onehot = np.max(verts_error_onehot) - np.min(verts_error_onehot)
            verts_error_onehot = (verts_error_onehot - np.min(verts_error_onehot)) / diff_onehot
            verts_error_onehot_color = np.matmul(verts_error_onehot, np.array([[255,255,255]])).astype(int)

            if args.test:
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor, self.foldername +"_mask.ply")
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor_onehot, self.foldername + "_mask_onehot.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_color, self.foldername + "_mask_error.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_onehot_color, self.foldername + "_mask_error_onehot.ply")
            else:
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor, self.foldername +"_mask_train.ply")   
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor_onehot, self.foldername + "_mask_onehot_train.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_color, self.foldername + "_mask_error_train.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_onehot_color, self.foldername + "_mask_error_train_onehot.ply")
            return

        print("training")

        if args.summary == True:
            writer = SummaryWriter(os.path.join("nofea", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1
        a = pow(10, pow_index) * a
        beta_3_init = pow(10, pow_index) * beta_3_init
        if a > a_max:
            a = a_max
        if beta_3_init > beta_3_max:
            beta_3_init = beta_3_max

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
                if a < a_max:
                    a = a * 10
                if beta_3_init < beta_3_max:
                    beta_3_init = beta_3_init * 10

            if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
            # if epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                for i in range(n_branch + 1):
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
            if args.stage == 2:
                self.adjust_learning_rate(epoch - self.startepoch)
            else:
                self.adjust_learning_rate(epoch)

            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]
            #print shape
            # print("mnfld_pnts size: ")
            # print(mnfld_pnts.size())
            # print("mnfld_pred size: ")
            # print(mnfld_pred.size())

            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss
            mnfld_loss = (mnfld_pred.abs()).mean()
            # print ("mnfld loss device: ", mnfld_loss.device)
            loss = loss + mnfld_loss
            # manifold loss for patches

            all_fi = torch.zeros(n_batchsize, 1).cuda()
            for i in range(n_branch - 1):
                all_fi[i * n_patch_batch : (i + 1) * n_patch_batch, 0] = mnfld_pred_all[i * n_patch_batch : (i + 1) * n_patch_batch, i + 1]
            #last patch
            all_fi[(n_branch - 1) * n_patch_batch:, 0] = mnfld_pred_all[(n_branch - 1) * n_patch_batch:, n_branch]

            # mnfld_loss_patch = 0.0
            # for i in range(n_branch):
                # mnfld_loss_patch = mnfld_loss_patch + (mnfld_pred_all[:,i + 1].abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            mnfld_loss_patch = all_fi[:,0].abs().mean()
            loss = loss + mnfld_loss_patch

            assignment_loss = torch.zeros(1).cuda()
            if assignment_weight > 0.0: 
                # for i in range(n_branch):
                    # assignment_loss = assignment_loss + ((mnfld_pred - mnfld_pred_all[:,i + 1]).abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
                assignment_loss = ((mnfld_pred - all_fi).abs()).mean()
                
            loss = loss + assignment_weight * assignment_loss

            # eikonal loss for all
            grad_loss = torch.zeros(1).cuda()
            # for i in range(n_branch):
            #     single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,i + 1])
            #     grad_loss = grad_loss + ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            # loss = loss + self.grad_lambda * grad_loss
            # eikonal loss for h
            single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,0])
            grad_loss_h = ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss_h

            # normals loss
            if self.with_normals:
                #all normals
                normals = cur_data[:, -self.d_in:]
                #defined for h
                # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1)).mean()
                # normals_loss = 0.0
                # for i in range(n_branch):
                #     branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                #     normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                loss = loss + self.normals_lambda * normals_loss

                mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()

                loss = loss + self.normals_lambda * normals_loss_h

            else:
                normals_loss = torch.zeros(1)
                normals_loss_h = torch.zeros(1)

            #mask loss
            mask_loss = torch.zeros(1).cuda()
            # mask_feature = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            #vanilla version
            # for i in range(k - 1):
            #     mask_feature = mask_feature * mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]

            # mask_loss = torch.abs(torch.sum(mask_feature, dim = 1) - 1).mean()
            # mask_loss = (torch.sum((mask_feature - single_branch_mask_gt).abs(), dim = 1) * (1 - cur_feature_mask) ).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

            #cross entropy form
            #flag_softmax set to False
            # ce_loss = nn.CrossEntropyLoss()
            # mask_loss = ce_loss(mask_feature, single_branch_mask_id)

            #dice version:
            # mask_loss = (mask_feature - single_branch_mask_gt).norm(2, dim=-1).mean()
            
            #focal loss
            # mask_loss = FocalLoss(gamma=5)(mask_feature, single_branch_mask_id)
            

            # loss = loss + omega_2 * mask_loss

            #mask only version:
            # loss = mask_loss


            # mask_loss expectation
            mask_expectation = torch.zeros(1).cuda()
            # mask_nonmfd = nonmnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            # sm_loss = nn.Softmax(dim=1)
            # mask_nonmfd_sm = sm_loss(mask_nonmfd)
            # mask_expectation = torch.abs(torch.sum(torch.pow(mask_nonmfd_sm, k), dim = 1) - 1).mean()
            # #not added yet
            # loss = loss + omega_3 * mask_expectation


            if epoch % 2000 == 0:
                k = k + 2
                if k > k_max:
                    k = k_max
            #     omega_3 = omega_3 * 10
            #     if omega_3 > omega3_max:
            #         omega_3 = omega3_max

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            #tensorboard
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',self.grad_lambda * grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',self.grad_lambda * grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss all', self.normals_lambda * normals_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss h', self.normals_lambda * normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/mask loss', omega_2 * mask_loss.item(), epoch)
                writer.add_scalar('Loss/mask expectation', omega_3 * mask_expectation.item(), epoch)
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
                print ('assignment_loss {:.6f}\t'.format(assignment_loss.item()))               
                print ('k: ', k, " mask loss: ", mask_loss.item())
                print ('mask expectation: ', mask_expectation.item())

    def run_multi_branch_mask_nofea_maskonly(self):
        #to be changed
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
        assignment_weight = 0.0
        omega_1 = 0.0
        omega_2 = 1.0
        # omega_3 = 0.01
        omega_3 = 1
        omega_4 = 0.0
        omega3_max = 10.0
        eta = 1.0
        a = 0.01
        a_max = 10.0
        beta_1 = 10
        beta_2 = 1
        eps = 0.01
        # beta_3_init = 0.02
        beta_3_init = 0.0
        beta_3_max = 0.2
        k = 2 
        k_max = 10

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
            for i in range(n_branch + 1):
                self.network.flag_output = i + 1
                self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_mask" + str(i))

            # self.network.flag_onehot = True
            # for i in range(n_branch + 1):
            #     self.network.flag_output = i + 1
            #     self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i) + "_onehot", with_cuts = True)
            # self.network.flag_output = 0
            # # self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_onehotmask" + str(i))
            # self.network.flag_onehot = False
            return
        if self.evaldist:
            print ("evaluating mesh and normal distance")
            #only for h
            avg_dist = 0.0
            avg_normal_dist = 0.0
            self.network.eval()
            for i, cur_data in enumerate(torch.split(self.data, self.points_batch)):
                mnfld_pnts = cur_data[:, :self.d_in]
                mnfld_pred_all = self.network(mnfld_pnts)
                mnfld_pred = mnfld_pred_all[:,0]
                #print shape
                mnfld_loss = (mnfld_pred.abs()).mean()
                # normals loss
                if self.with_normals:
                    #all normals
                    normals = cur_data[:, -self.d_in:]
                    #defined for h
                    # # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                    # branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                    # normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1)).mean()
                    # # normals_loss = 0.0
                    # # for i in range(n_branch):
                    # #     branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                    # #     normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                    # loss = loss + self.normals_lambda * normals_loss

                    mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                    normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()
                else:
                    # normals_loss = torch.zeros(1)
                    normals_loss_h = torch.zeros(1)
                avg_dist = avg_dist + mnfld_loss * mnfld_pnts.shape[0]
                avg_normal_dist = avg_normal_dist + normals_loss_h * mnfld_pnts.shape[0]
                print ("batch loss: {} mnfld:{:.6f}\t normal:{:.6f}\n".format(i, mnfld_loss.item(), normals_loss_h.item()))
            avg_dist = avg_dist / self.data.shape[0]
            avg_normal_dist = avg_normal_dist / self.data.shape[0]
            print ("Overall loss: mnfld:{:.6f}\t normal:{:.6f}\n".format(avg_dist, avg_normal_dist))
            return
        if args.vismask:
            print("visualize mask for point cloud") 
            branch_color = []
            if n_branch == 1:
                branch_color.append(cm.plasma(0.0))
            else:
                for i in range(n_branch):
                    # branch_color.append(cm.hot(i / n_branch))
                    branch_color.append(cm.plasma(i / (n_branch - 1)))
            branch_color = np.concatenate(branch_color, axis = 0)
            branch_color = branch_color.reshape(-1,4)
            branch_color = branch_color[:,:3] * 255
            vertscolor = np.zeros([self.data.shape[0], 3])
            vertscolor_onehot = np.zeros([self.data.shape[0], 3])
            count = 0
            verts_error = np.zeros([self.data.shape[0], 1])
            verts_error_onehot = np.zeros([self.data.shape[0], 1])
            verts_bgt = np.zeros([self.data.shape[0], n_branch])
            verts_bgt[np.arange(self.data.shape[0]), feature_mask_cpu.astype(int) - 1] = 1.0

            self.network.flag_softmax = True
            for i, cur_data in enumerate(torch.split(self.data, self.points_batch)):
                mnfld_pnts = cur_data[:, :self.d_in]
                mnfld_pred_all = self.network(mnfld_pnts)
                cur_data_length = cur_data.shape[0]
                mask_feature_np = mnfld_pred_all.detach()[:, n_branch + 1: 2 * n_branch + 1].cpu().numpy()
                maxid = np.argmax(mask_feature_np, 1)
                mask_feature_onehot = np.zeros_like(mask_feature_np)
                mask_feature_onehot[np.arange(mask_feature_np.shape[0]), maxid] = 1.0

                # verts_error[count:count + cur_data_length, :] =  np.expand_dims(np.sum(np.abs(mask_feature_np - mask_feature_onehot), 1),1)
                verts_error[count:count + cur_data_length, :] =  np.expand_dims(LA.norm((mask_feature_np - verts_bgt[count:count + cur_data_length, :]), axis = 1),1)
                verts_error_onehot[count:count + cur_data_length, :] =  np.expand_dims(LA.norm((mask_feature_onehot - verts_bgt[count:count + cur_data_length, :]), axis = 1),1)
                vertscolor[count:count + cur_data_length, :] = np.matmul(mask_feature_np, branch_color)
                vertscolor_onehot[count:count + cur_data_length, :] = np.matmul(mask_feature_onehot, branch_color)
                count = count + self.points_batch
            vertscolor[vertscolor > 255] = 255
            vertscolor = vertscolor.astype(int)
            vertscolor_onehot[vertscolor_onehot > 255] = 255
            vertscolor_onehot = vertscolor_onehot.astype(int)

            print ("sum: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error), np.max(verts_error), np.min(verts_error)))
            print ("hot: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error_onehot), np.max(verts_error_onehot), np.min(verts_error_onehot)))
            diff = np.max(verts_error) - np.min(verts_error)
            verts_error = (verts_error - np.min(verts_error)) / diff
            verts_error_color = np.matmul(verts_error, np.array([[255,255,255]])).astype(int)

            diff_onehot = np.max(verts_error_onehot) - np.min(verts_error_onehot)
            verts_error_onehot = (verts_error_onehot - np.min(verts_error_onehot)) / diff_onehot
            verts_error_onehot_color = np.matmul(verts_error_onehot, np.array([[255,255,255]])).astype(int)

            if args.test:
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor, self.foldername +"_mask.ply")
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor_onehot, self.foldername + "_mask_onehot.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_color, self.foldername + "_mask_error.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_onehot_color, self.foldername + "_mask_error_onehot.ply")
            else:
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor, self.foldername +"_mask_train.ply")   
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor_onehot, self.foldername + "_mask_onehot_train.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_color, self.foldername + "_mask_error_train.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_onehot_color, self.foldername + "_mask_error_train_onehot.ply")
            return

        print("training")

        if args.summary == True:
            writer = SummaryWriter(os.path.join("nofea", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1
        a = pow(10, pow_index) * a
        beta_3_init = pow(10, pow_index) * beta_3_init
        if a > a_max:
            a = a_max
        if beta_3_init > beta_3_max:
            beta_3_init = beta_3_max

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
                if a < a_max:
                    a = a * 10
                if beta_3_init < beta_3_max:
                    beta_3_init = beta_3_init * 10

            # if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
            # # if epoch % self.conf.get_int('train.plot_frequency') == 0:
            #     # print('saving checkpoint: ', epoch)
            #     print('plot validation epoch: ', epoch)
            #     for i in range(n_branch + 1):
            #         self.network.flag_output = i + 1
            #         self.plot_shapes(epoch, file_suffix = "_" + str(i), with_cuts = False)
            #     self.network.flag_output = 0

            # change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)
            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]

            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            #mask loss
            mask_feature = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            #vanilla version
            # for i in range(k - 1):
            #     mask_feature = mask_feature * mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]

            # mask_loss = torch.abs(torch.sum(mask_feature, dim = 1) - 1).mean()
            # mask_loss = (torch.sum((mask_feature - single_branch_mask_gt).abs(), dim = 1) * (1 - cur_feature_mask) ).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

            #cross entropy form
            #flag_softmax set to False
            if epoch == 10000:
                eta = 0.7

            ce_loss = nn.CrossEntropyLoss()
            mask_loss = ce_loss(mask_feature, single_branch_mask_id)

            #dice version:
            # mask_loss = (mask_feature - single_branch_mask_gt).norm(2, dim=-1).mean()
            
            #focal loss
            # mask_loss = FocalLoss(gamma=5)(mask_feature, single_branch_mask_id)
            # loss = loss + omega_2 * mask_loss

            #eta version:
            # mask_mfd = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            # sm_loss = nn.Softmax(dim=1)
            # mask_mfd_sm = sm_loss(mask_mfd)

            # mask_bool = single_branch_mask_gt.type(torch.BoolTensor)
            # mask_weight = (mask_mfd_sm[mask_bool] < eta).type(torch.float32)
            # mask_mfd_logsm = F.log_softmax(mask_mfd, dim=1)
            # # mask_loss = (mask_weight * (torch.sum(single_branch_mask_gt * mask_mfd_logsm, dim=1))).mean()
            # single_branch_mask_id = single_branch_mask_id.view(-1,1)
            # mask_mfd_logsm = mask_mfd_logsm.gather(1,single_branch_mask_id)
            # mask_mfd_logsm = mask_mfd_logsm.view(-1)
            # mask_loss = -1 * (mask_weight * mask_mfd_logsm).mean()


            loss = mask_loss
            # mask_loss expectation
            mask_expectation = torch.zeros(1).cuda()
            mask_nonmfd = nonmnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            if omega_3 > 0.0:
                sm_loss = nn.Softmax(dim=1)
                mask_nonmfd_sm = sm_loss(mask_nonmfd)
                mask_expectation = torch.abs(torch.sum(torch.pow(mask_nonmfd_sm, k), dim = 1) - 1).mean()
                # #not added yet
                loss = loss + omega_3 * mask_expectation

            mask_grad_loss = torch.zeros(1).cuda()
            if omega_4 > 0.0:
                mask_grad = gradient(nonmnfld_pnts, mask_nonmfd)
                mask_grad_loss = mask_grad.norm(1,dim=-1).mean()
                loss = loss + omega_4 * mask_grad_loss

            if epoch % 2000 == 0:
                k = k + 2
                if k > k_max:
                    k = k_max
            #     omega_3 = omega_3 * 10
            #     if omega_3 > omega3_max:
            #         omega_3 = omega3_max

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            mnfld_loss = torch.zeros(1).cuda()
            mnfld_loss_patch = torch.zeros(1).cuda()
            grad_loss = torch.zeros(1).cuda()
            grad_loss_h = torch.zeros(1).cuda()
            normals_loss = torch.zeros(1).cuda()
            normals_loss_h = torch.zeros(1).cuda()

            #tensorboard
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',self.grad_lambda * grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',self.grad_lambda * grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss all', self.normals_lambda * normals_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss h', self.normals_lambda * normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/mask loss', omega_2 * mask_loss.item(), epoch)
                writer.add_scalar('Loss/mask expectation', omega_3 * mask_expectation.item(), epoch)
                writer.add_scalar('Loss/mask grad', omega_4 * mask_grad_loss.item(), epoch)
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
                print ('k: ', k, " mask loss: ", mask_loss.item())
                print ('mask expectation: ', mask_expectation.item())
                print ('mask grad: ', mask_grad_loss.item())

    def run_multi_branch_mask_nofea_maskonly_svm(self):
        #to be changed
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
        assignment_weight = 0.0
        omega_1 = 0.0
        omega_2 = 1.0
        c_svm=args.csvm
        print("C for svm: ", c_svm)
        # omega_3 = 0.01
        omega_3 = 1
        omega_4 = 0.0
        omega3_max = 10.0
        eta = 1.0
        a = 0.01
        a_max = 10.0
        beta_1 = 10
        beta_2 = 1
        eps = 0.01
        # beta_3_init = 0.02
        beta_3_init = 0.0
        beta_3_max = 0.2
        k = 2 
        k_max = 10

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
            for i in range(n_branch + 1):
                self.network.flag_output = i + 1
                self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_mask" + str(i))

            # self.network.flag_onehot = True
            # for i in range(n_branch + 1):
            #     self.network.flag_output = i + 1
            #     self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i) + "_onehot", with_cuts = True)
            # self.network.flag_output = 0
            # # self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_onehotmask" + str(i))
            # self.network.flag_onehot = False
            return
        if self.evaldist:
            print ("evaluating mesh and normal distance")
            #only for h
            avg_dist = 0.0
            avg_normal_dist = 0.0
            self.network.eval()
            for i, cur_data in enumerate(torch.split(self.data, self.points_batch)):
                mnfld_pnts = cur_data[:, :self.d_in]
                mnfld_pred_all = self.network(mnfld_pnts)
                mnfld_pred = mnfld_pred_all[:,0]
                #print shape
                mnfld_loss = (mnfld_pred.abs()).mean()
                # normals loss
                if self.with_normals:
                    #all normals
                    normals = cur_data[:, -self.d_in:]
                    #defined for h
                    # # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                    # branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                    # normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1)).mean()
                    # # normals_loss = 0.0
                    # # for i in range(n_branch):
                    # #     branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                    # #     normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                    # loss = loss + self.normals_lambda * normals_loss

                    mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                    normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()
                else:
                    # normals_loss = torch.zeros(1)
                    normals_loss_h = torch.zeros(1)
                avg_dist = avg_dist + mnfld_loss * mnfld_pnts.shape[0]
                avg_normal_dist = avg_normal_dist + normals_loss_h * mnfld_pnts.shape[0]
                print ("batch loss: {} mnfld:{:.6f}\t normal:{:.6f}\n".format(i, mnfld_loss.item(), normals_loss_h.item()))
            avg_dist = avg_dist / self.data.shape[0]
            avg_normal_dist = avg_normal_dist / self.data.shape[0]
            print ("Overall loss: mnfld:{:.6f}\t normal:{:.6f}\n".format(avg_dist, avg_normal_dist))
            return
        if args.vismask:
            print("visualize mask for point cloud") 
            branch_color = []
            if n_branch == 1:
                branch_color.append(cm.plasma(0.0))
            else:
                for i in range(n_branch):
                    # branch_color.append(cm.hot(i / n_branch))
                    branch_color.append(cm.plasma(i / (n_branch - 1)))
            branch_color = np.concatenate(branch_color, axis = 0)
            branch_color = branch_color.reshape(-1,4)
            branch_color = branch_color[:,:3] * 255
            vertscolor = np.zeros([self.data.shape[0], 3])
            vertscolor_onehot = np.zeros([self.data.shape[0], 3])
            count = 0
            verts_error = np.zeros([self.data.shape[0], 1])
            verts_error_onehot = np.zeros([self.data.shape[0], 1])
            verts_bgt = np.zeros([self.data.shape[0], n_branch])
            verts_bgt[np.arange(self.data.shape[0]), feature_mask_cpu.astype(int) - 1] = 1.0

            self.network.flag_softmax = True
            for i, cur_data in enumerate(torch.split(self.data, self.points_batch)):
                mnfld_pnts = cur_data[:, :self.d_in]
                mnfld_pred_all = self.network(mnfld_pnts)
                cur_data_length = cur_data.shape[0]
                mask_feature_np = mnfld_pred_all.detach()[:, n_branch + 1: 2 * n_branch + 1].cpu().numpy()
                maxid = np.argmax(mask_feature_np, 1)
                mask_feature_onehot = np.zeros_like(mask_feature_np)
                mask_feature_onehot[np.arange(mask_feature_np.shape[0]), maxid] = 1.0

                # verts_error[count:count + cur_data_length, :] =  np.expand_dims(np.sum(np.abs(mask_feature_np - mask_feature_onehot), 1),1)
                verts_error[count:count + cur_data_length, :] =  np.expand_dims(LA.norm((mask_feature_np - verts_bgt[count:count + cur_data_length, :]), axis = 1),1)
                verts_error_onehot[count:count + cur_data_length, :] =  np.expand_dims(LA.norm((mask_feature_onehot - verts_bgt[count:count + cur_data_length, :]), axis = 1),1)
                vertscolor[count:count + cur_data_length, :] = np.matmul(mask_feature_np, branch_color)
                vertscolor_onehot[count:count + cur_data_length, :] = np.matmul(mask_feature_onehot, branch_color)
                count = count + self.points_batch
            vertscolor[vertscolor > 255] = 255
            vertscolor = vertscolor.astype(int)
            vertscolor_onehot[vertscolor_onehot > 255] = 255
            vertscolor_onehot = vertscolor_onehot.astype(int)

            print ("sum: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error), np.max(verts_error), np.min(verts_error)))
            print ("hot: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error_onehot), np.max(verts_error_onehot), np.min(verts_error_onehot)))
            diff = np.max(verts_error) - np.min(verts_error)
            verts_error = (verts_error - np.min(verts_error)) / diff
            verts_error_color = np.matmul(verts_error, np.array([[255,255,255]])).astype(int)

            diff_onehot = np.max(verts_error_onehot) - np.min(verts_error_onehot)
            verts_error_onehot = (verts_error_onehot - np.min(verts_error_onehot)) / diff_onehot
            verts_error_onehot_color = np.matmul(verts_error_onehot, np.array([[255,255,255]])).astype(int)

            if args.test:
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor, self.foldername +"_mask.ply")
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor_onehot, self.foldername + "_mask_onehot.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_color, self.foldername + "_mask_error.ply")
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_onehot_color, self.foldername + "_mask_error_onehot.ply")
            else:
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor, self.foldername +"_mask_train.ply")   
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), vertscolor_onehot, self.foldername + "_mask_onehot_train.ply")
                save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_color, self.foldername + "_mask_error_train.ply")
                # save_vertnormal_color_ply(self.data.detach().cpu().numpy(), verts_error_onehot_color, self.foldername + "_mask_error_train_onehot.ply")
            return

        print("training")

        if args.summary == True:
            writer = SummaryWriter(os.path.join("nofea", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1
        a = pow(10, pow_index) * a
        beta_3_init = pow(10, pow_index) * beta_3_init
        if a > a_max:
            a = a_max
        if beta_3_init > beta_3_max:
            beta_3_init = beta_3_max

        print('summary status: ', args.summary)
        # branch mask is predefined
        branch_mask = torch.zeros(n_branch, n_batchsize).cuda()
        single_branch_mask_gt = torch.zeros(n_batchsize, n_branch).cuda()
        #for cross entropy loss, id start from 0
        single_branch_mask_id = torch.zeros([n_batchsize], dtype = torch.long).cuda()
        single_branch_mask_svm = -1 * torch.ones(n_batchsize, n_branch).cuda()

        for i in range(n_branch - 1):
            branch_mask[i, i * n_patch_batch : (i + 1) * n_patch_batch] = 1.0
            single_branch_mask_gt[i * n_patch_batch : (i + 1) * n_patch_batch, i] = 1.0
            single_branch_mask_svm[i * n_patch_batch : (i + 1) * n_patch_batch, i] = 1.0
            single_branch_mask_id[i * n_patch_batch : (i + 1) * n_patch_batch] = i

        #last patch
        branch_mask[n_branch - 1, (n_branch - 1) * n_patch_batch:] = 1.0
        single_branch_mask_gt[(n_branch - 1) * n_patch_batch:, (n_branch - 1)] = 1.0
        single_branch_mask_svm[(n_branch - 1) * n_patch_batch:, (n_branch - 1)] = 1.0
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
                if a < a_max:
                    a = a * 10
                if beta_3_init < beta_3_max:
                    beta_3_init = beta_3_init * 10

            # if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
            # # if epoch % self.conf.get_int('train.plot_frequency') == 0:
            #     # print('saving checkpoint: ', epoch)
            #     print('plot validation epoch: ', epoch)
            #     for i in range(n_branch + 1):
            #         self.network.flag_output = i + 1
            #         self.plot_shapes(epoch, file_suffix = "_" + str(i), with_cuts = False)
            #     self.network.flag_output = 0

            # change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)
            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]

            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            #mask loss
            
            #vanilla version
            # for i in range(k - 1):
            #     mask_feature = mask_feature * mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]

            # mask_loss = torch.abs(torch.sum(mask_feature, dim = 1) - 1).mean()
            # mask_loss = (torch.sum((mask_feature - single_branch_mask_gt).abs(), dim = 1) * (1 - cur_feature_mask) ).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

            #cross entropy form
            #flag_softmax set to False
            if epoch == 10000:
                eta = 0.7

            # ce_loss = nn.CrossEntropyLoss()
            # mask_loss = ce_loss(mask_feature, single_branch_mask_id)

            #dice version:
            # mask_loss = (mask_feature - single_branch_mask_gt).norm(2, dim=-1).mean()
            
            #focal loss
            # mask_loss = FocalLoss(gamma=5)(mask_feature, single_branch_mask_id)
            # loss = loss + omega_2 * mask_loss

            #eta version:
            # mask_mfd = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            # sm_loss = nn.Softmax(dim=1)
            # mask_mfd_sm = sm_loss(mask_mfd)

            # mask_bool = single_branch_mask_gt.type(torch.BoolTensor)
            # mask_weight = (mask_mfd_sm[mask_bool] < eta).type(torch.float32)
            # mask_mfd_logsm = F.log_softmax(mask_mfd, dim=1)
            # # mask_loss = (mask_weight * (torch.sum(single_branch_mask_gt * mask_mfd_logsm, dim=1))).mean()
            # single_branch_mask_id = single_branch_mask_id.view(-1,1)
            # mask_mfd_logsm = mask_mfd_logsm.gather(1,single_branch_mask_id)
            # mask_mfd_logsm = mask_mfd_logsm.view(-1)
            # mask_loss = -1 * (mask_weight * mask_mfd_logsm).mean()

            mask_predict = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            svm_dist = mnfld_pred_all[:, 2 * n_branch + 1: 3 * n_branch + 1]
            mask_loss = torch.zeros(1).cuda()
            for i in range(n_branch):
                weight = self.network.svms[i].weight.squeeze()
                mask_loss += weight.t() @weight / 2.0
            mask_loss = mask_loss / n_branch
            mask_loss += c_svm * torch.mean(torch.pow(torch.clamp(1 - svm_dist * single_branch_mask_svm, min=0), 2))


            loss = mask_loss
            # mask_loss expectation
            mask_expectation = torch.zeros(1).cuda()
            # mask_nonmfd = nonmnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            # if omega_3 > 0.0:
            #     sm_loss = nn.Softmax(dim=1)
            #     mask_nonmfd_sm = sm_loss(mask_nonmfd)
            #     mask_expectation = torch.abs(torch.sum(torch.pow(mask_nonmfd_sm, k), dim = 1) - 1).mean()
            #     # #not added yet
            #     loss = loss + omega_3 * mask_expectation

            mask_grad_loss = torch.zeros(1).cuda()
            # if omega_4 > 0.0:
            #     mask_grad = gradient(nonmnfld_pnts, mask_nonmfd)
            #     mask_grad_loss = mask_grad.norm(1,dim=-1).mean()
            #     loss = loss + omega_4 * mask_grad_loss

            if epoch % 2000 == 0:
                k = k + 2
                if k > k_max:
                    k = k_max
            #     omega_3 = omega_3 * 10
            #     if omega_3 > omega3_max:
            #         omega_3 = omega3_max

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            mnfld_loss = torch.zeros(1).cuda()
            mnfld_loss_patch = torch.zeros(1).cuda()
            grad_loss = torch.zeros(1).cuda()
            grad_loss_h = torch.zeros(1).cuda()
            normals_loss = torch.zeros(1).cuda()
            normals_loss_h = torch.zeros(1).cuda()

            #tensorboard
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',self.grad_lambda * grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',self.grad_lambda * grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss all', self.normals_lambda * normals_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss h', self.normals_lambda * normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/mask loss', omega_2 * mask_loss.item(), epoch)
                writer.add_scalar('Loss/mask expectation', omega_3 * mask_expectation.item(), epoch)
                writer.add_scalar('Loss/mask grad', omega_4 * mask_grad_loss.item(), epoch)
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
                print('Train Epoch: [{}/{}]'.format(
                    epoch, self.nepochs))
                print ('k: ', k, " mask loss: ", mask_loss.item())
                print ('mask expectation: ', mask_expectation.item())
                print ('mask grad: ', mask_grad_loss.item())

    def run_multi_branch_maxsdf(self):
        #to be changed
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
        patch_sup = False
        # patch_sup = True


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
            for i in range(n_branch + 1):
                self.network.flag_output = i + 1
                self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            self.plot_masks_maxsdf(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_mask" + str(i))

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
            writer = SummaryWriter(os.path.join("runs_maxsdf", self.foldername))
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

            # if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
            if epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                for i in range(n_branch + 1):
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

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]
            #print shape
            # print("mnfld_pnts size: ")
            # print(mnfld_pnts.size())
            # print("mnfld_pred size: ")
            # print(mnfld_pred.size())

            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss
            mnfld_loss = (mnfld_pred.abs()).mean()
            # print ("mnfld loss device: ", mnfld_loss.device)
            loss = loss + mnfld_loss
            # manifold loss for patches

            all_fi = torch.zeros(n_batchsize, 1).cuda()
            for i in range(n_branch - 1):
                all_fi[i * n_patch_batch : (i + 1) * n_patch_batch, 0] = mnfld_pred_all[i * n_patch_batch : (i + 1) * n_patch_batch, i + 1]
            #last patch
            all_fi[(n_branch - 1) * n_patch_batch:, 0] = mnfld_pred_all[(n_branch - 1) * n_patch_batch:, n_branch]

            # mnfld_loss_patch = 0.0
            # for i in range(n_branch):
                # mnfld_loss_patch = mnfld_loss_patch + (mnfld_pred_all[:,i + 1].abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            mnfld_loss_patch = torch.zeros(1).cuda()
            if patch_sup:
                mnfld_loss_patch = all_fi[:,0].abs().mean()
            loss = loss + mnfld_loss_patch

            # eikonal loss for all
            grad_loss = torch.zeros(1).cuda()
            for i in range(n_branch):
                single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,i + 1])
                grad_loss = grad_loss + ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss
            # eikonal loss for h
            single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,0])
            grad_loss_h = ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss_h

            # normals loss
            if self.with_normals:
                #all normals
                normals = cur_data[:, -self.d_in:]
                #defined for h
                # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                normals_loss = torch.zeros(1).cuda()
                if patch_sup:
                    branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                    normals_loss = ((torch.cross(branch_grad, normals, dim=1).abs()).norm(2, dim=1)).mean() #to be changed
                # normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                loss = loss + self.normals_lambda * normals_loss

                mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()

                loss = loss + self.normals_lambda * normals_loss_h

            else:
                normals_loss = torch.zeros(1)
                normals_loss_h = torch.zeros(1)

            #mask loss
            mask_feature = mnfld_pred_all[:, n_branch + 1:]
            assert(mask_feature.shape[1] == 3 * n_branch)
            mask_loss = (n_branch - torch.sum(torch.pow(mask_feature, a), dim = 1)).mean()
            #original version:
            loss = loss + omega_1 * mask_loss

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
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',self.grad_lambda * grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',self.grad_lambda * grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss all', self.normals_lambda * normals_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss h', self.normals_lambda * normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/mask loss', omega_1 * mask_loss.item(), epoch)
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
                print ("mask loss: ", mask_loss.item())

    def run_multi_branch_maxsdf_nomask(self):
        #to be changed
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
        weight_assignment = 1
        a_assignment = 100
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

        if args.getmask:
            self.get_mask(os.path.join('/mnt/sdf1/haog/code/IGR', self.foldername + "_getmask.txt"))
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

            #assignment loss
            assignment_loss = torch.zeros(1).cuda()
            if  not (args.ab == 'cor' or args.ab == 'cc') and args.assign and epoch > 10000 and not args.baseline:
                wrong_assignment_flag = torch.abs(mnfld_pred - all_fi[:,0]) > args.th_closeness
                if args.linearassign:
                    assignment_loss = (a_assignment * torch.abs(mnfld_pred - all_fi[:,0])[wrong_assignment_flag]).mean()
                else:
                    assignment_loss = torch.exp(a_assignment * torch.abs(mnfld_pred - all_fi[:,0])[wrong_assignment_flag]).mean() - 1.0
                    
                loss = loss + weight_assignment * assignment_loss


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
                writer.add_scalar('Loss/Assignment loss', assignment_loss.item(), epoch)
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
                print ('assignment loss: ', assignment_loss.item())
                print ('offsurface_loss loss: ', offsurface_loss.item())
                
                if args.feature_sample:
                    print('feature mnfld loss: {} patch loss: {} cons loss: {}'.format(feature_mnfld_loss.item(), feature_loss_patch.item(), feature_loss_cons.item()))

        self.tracing()
        # if not args.baseline:
        #     self.visualize_loss()

    def run_multi_branch_maxsdf_nomask_approx(self):
        #to be changed
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
            for i in range(n_branch + 1):
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
            writer = SummaryWriter(os.path.join("runs_maxsdf_approx", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1

        # print('summary status: ', args.summary)
        # for i in range(n_branch - 1):
        #     branch_mask[i, i * n_patch_batch : (i + 1) * n_patch_batch] = 1.0
        #     single_branch_mask_gt[i * n_patch_batch : (i + 1) * n_patch_batch, i] = 1.0
        #     single_branch_mask_id[i * n_patch_batch : (i + 1) * n_patch_batch] = i

        # #last patch
        # branch_mask[n_branch - 1, (n_branch - 1) * n_patch_batch:] = 1.0
        # single_branch_mask_gt[(n_branch - 1) * n_patch_batch:, (n_branch - 1)] = 1.0
        # single_branch_mask_id[(n_branch - 1) * n_patch_batch:] = (n_branch - 1)

        for epoch in range(self.startepoch, self.nepochs + 1):
            #indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            #first feature part then non-feature part
            # indices = torch.tensor(feature_id[0][np.random.choice(n_feature, n_feature_batch, True)]).cuda()
            indices = torch.empty(0,dtype=torch.int64).cuda()
            #current feature mask
            # cur_feature_mask = torch.zeros(n_branch * n_patch_batch + n_feature_batch).cuda()
            # cur_feature_mask[0: n_feature_batch] = 1.0
            cur_feature_mask = torch.zeros(n_batchsize).cuda()
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

            if epoch != self.startepoch and epoch % self.conf.get_int('train.plot_frequency') == 0:
            # if epoch == self.nepochs + 1:
            # if epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                for i in range(n_branch + 1):
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
            mnfld_pred_devide_grad = mnfld_pred / mnfld_grad.norm(2, dim=1)

            approx_h_loss = (mnfld_pred_devide_grad.abs()).mean()
                # mnfld_loss = (mnfld_pred.abs()).mean()

            #eta version
            # mask_mfd = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            # sm_loss = nn.Softmax(dim=1)
            # mask_mfd_sm = sm_loss(mask_mfd)

            # mask_bool = single_branch_mask_gt.type(torch.BoolTensor)
            # mask_weight = (mask_mfd_sm[mask_bool] < eta).type(torch.float32)
            # mask_mfd_logsm = F.log_softmax(mask_mfd, dim=1)
            # # mask_loss = (mask_weight * (torch.sum(single_branch_mask_gt * mask_mfd_logsm, dim=1))).mean()
            
            # print ("mnfld loss device: ", mnfld_loss.device)
            loss = loss + weight_mnfld_h *  approx_h_loss
            # manifold loss for patches

            approx_offset_consistency_loss = torch.zeros(1).cuda()
            offset_normal_loss = torch.zeros(1).cuda()
            approx_offset_patch_loss = torch.zeros(1).cuda()

            if offset_omega > 0.0:

                #for h
                offset_sigma_array = torch.rand(cur_data.shape[0], 1).cuda() * 2 * offset_sigma - offset_sigma
                offset_pnts_pos = mnfld_pnts + cur_data[:,self.d_in:] * offset_sigma_array
                offset_pred_pos = self.network(offset_pnts_pos)

                
                #offset normal loss
                normals = cur_data[:, -self.d_in:]
                offset_grad = gradient(offset_pnts_pos, offset_pred_pos[:, 0])

                offset_pred_divide_grad = offset_pred_pos[:,0]/offset_grad.norm(2, dim=1)

                # offset_loss = offset_loss + (((offset_grad - normals).abs()).norm(2, dim=1)).mean()

                # loss = loss + offset_omega * offset_loss

                # offset_grad_neg = gradient(offset_pnts_neg, offset_pred_neg)
                # offset_loss = offset_loss + (((offset_grad_neg - normals).abs()).norm(2, dim=1)).mean()

                #for all patch
                
                offset_all_fi = torch.zeros(n_batchsize, 1).cuda()
                for i in range(n_branch - 1):
                    offset_all_fi[i * n_patch_batch : (i + 1) * n_patch_batch, 0] = offset_pred_pos[i * n_patch_batch : (i + 1) * n_patch_batch, i + 1]
                #last patch
                offset_all_fi[(n_branch - 1) * n_patch_batch:, 0] = offset_pred_pos[(n_branch - 1) * n_patch_batch:, n_branch]

                #manifold loss
                # offset_patch_loss = offset_patch_loss +  (offset_all_fi - offset_sigma_array).abs().mean()
                
                #offset normal loss
                # normals = cur_data[:, -self.d_in:]
                offset_grad_all = gradient(offset_pnts_pos, offset_all_fi[:,0])
                offset_all_pred_divide_grad = offset_all_fi[:,0]/offset_grad_all.norm(2, dim=1)

                approx_offset_consistency_loss = (offset_pred_divide_grad - offset_all_pred_divide_grad).abs().mean()

                offset_normal_loss = (offset_grad_all - normals).norm(2, dim=1).mean()

                approx_offset_patch_loss = (offset_all_pred_divide_grad - offset_sigma_array).abs().mean()

                loss = loss + offset_omega * (approx_offset_consistency_loss + approx_offset_patch_loss + offset_normal_loss)

                # offset_patch_loss = offset_patch_loss + (((offset_grad_all - normals).abs()).norm(2, dim=1)).mean()
                # loss = loss + offset_omega * offset_patch_loss



            all_fi = torch.zeros(n_batchsize, 1).cuda()
            for i in range(n_branch - 1):
                all_fi[i * n_patch_batch : (i + 1) * n_patch_batch, 0] = mnfld_pred_all[i * n_patch_batch : (i + 1) * n_patch_batch, i + 1]
            #last patch
            all_fi[(n_branch - 1) * n_patch_batch:, 0] = mnfld_pred_all[(n_branch - 1) * n_patch_batch:, n_branch]

            # mnfld_loss_patch = 0.0
            # for i in range(n_branch):
                # mnfld_loss_patch = mnfld_loss_patch + (mnfld_pred_all[:,i + 1].abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            # mnfld_loss_patch = torch.zeros(1).cuda()
            # if patch_sup:
            #     mnfld_loss_patch = all_fi[:,0].abs().mean()
            #     # mnfld_loss_patch = (all_fi[:,0]*all_fi[:,0]).mean()
            # loss = loss + mnfld_loss_patch
            branch_grad = gradient(mnfld_pnts, all_fi[:,0])
            all_fi_devide_grad = all_fi.squeeze() / branch_grad.norm(2, dim=1)
            approx_patch_loss = all_fi_devide_grad.abs().mean()
            loss = loss + approx_patch_loss    

            # eikonal loss for all
            grad_loss = torch.zeros(1).cuda()
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
            if self.with_normals:
                #all normals
                normals = cur_data[:, -self.d_in:]
                #defined for h
                # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                normals_loss = torch.zeros(1).cuda()
                if patch_sup:
                    # normals_loss = ((torch.cross(branch_grad, normals, dim=1).abs()).norm(2, dim=1)).mean()
                    normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1)).mean()
                loss = loss + self.normals_lambda * normals_loss

                normals_loss_h = torch.zeros(1).cuda()

                loss = loss + self.normals_lambda * normals_loss_h

                normal_consistency_loss = torch.zeros(1).cuda()

                mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                # normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1)).mean()
                normal_consistency_loss = (mnfld_grad - branch_grad).abs().norm(2, dim=1).mean()

                # loss = loss + self.normals_lambda * normal_consistency_loss

            else:
                normals_loss = torch.zeros(1)
                normals_loss_h = torch.zeros(1)

            assert(mnfld_pred_devide_grad.shape == all_fi_devide_grad.shape)
            approx_consistency_loss = (mnfld_pred_devide_grad - all_fi_devide_grad).abs().mean()
            loss = loss + approx_consistency_loss
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
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Approx h loss', approx_h_loss.item(), epoch)
                writer.add_scalar('Loss/Approx patch loss', approx_patch_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',self.grad_lambda * grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',self.grad_lambda * grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Patch normal loss ', self.normals_lambda * normals_loss.item(), epoch)
                writer.add_scalar('Loss/h normal loss', self.normals_lambda * normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/Approx consistency loss', approx_consistency_loss, epoch)
                writer.add_scalar('Loss/Normal consistency loss', self.normals_lambda * normal_consistency_loss.item(), epoch)
                writer.add_scalar('Loss/Offset normal loss', offset_omega * offset_normal_loss.item(), epoch)
                writer.add_scalar('Loss/Approx offset cons loss', offset_omega * approx_offset_consistency_loss.item(), epoch)
                writer.add_scalar('Loss/Approx offset patch loss', offset_omega * approx_offset_patch_loss.item(), epoch)
                
                # writer.add_scalar('Loss/Offset loss h', offset_omega * offset_loss.item(), epoch)
                # writer.add_scalar('Loss/Offset loss patch', offset_omega * offset_patch_loss.item(), epoch)
                # # writer.add_scalar('Loss/mask loss', omega_1 * mask_loss.item(), epoch)
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
                    loss.item(), approx_h_loss.item(), approx_patch_loss.item(), grad_loss.item(), grad_loss_h.item(), normals_loss.item(), normals_loss_h.item()))
                # print ("offset loss: ", offset_loss.item())
                # print ("offset patch loss: ", offset_patch_loss.item())
                print ('Approx consistency loss: ', approx_consistency_loss.item())
                print ('Normal consistency loss: ', normal_consistency_loss.item())
                print ('Offset patch/normal/cs loss: {:.6f}/{:.6f}/{:.6f}'.format(approx_offset_patch_loss.item(), offset_normal_loss.item(), approx_offset_consistency_loss.item()))
        self.tracing()
        self.visualize_loss()
        #save visualization of loss
        

    def tracing(self):
        #network definition
        # conf_filename = args.conf
        # conf = ConfigFactory.parse_file('./reconstruction/' + conf_filename)
        # feature_mask = utils.load_feature_mask()
        # feature_mask_file = conf.get_string('train.feature_mask_path')
        # feature_mask = torch.tensor(np.loadtxt(conf.get_string('train.feature_mask_path'))).float()

        input_file = self.conf.get_string('train.input_path')
        # csg_tree = ConfigFactory.parse_file(input_file[:-4]+'_csg.conf').get_list('csg.list')
        device = torch.device('cuda')
        if args.cpu:
            device = torch.device('cpu')

        # if args.baseline:
        #     feature_mask = torch.ones(feature_mask.shape).float()

        nb = int(torch.max(self.feature_mask).item())
        onehot = False
        if args.onehot:
            onehot = True

        if not args.ori:
            network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=3, split_pos = -1, flag_output = 1,
                                                                                            n_branch = nb,
                                                                                            csg_tree = self.csg_tree,
                                                                                            flag_convex = self.csg_flag_convex,
                                                                                            flag_onehot = onehot,
                                                                                            **self.conf.get_config(
                                                                                            'network.inputs'))
            # network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=3, split_pos = -1, flag_output = 1,
            #                                                                                 n_branch = nb + 2,
            #                                                                                 csg_tree = self.csg_tree,
            #                                                                                 flag_convex = self.csg_flag_convex,
            #                                                                                 flag_onehot = onehot,
            #                                                                                 **self.conf.get_config(
            #                                                                                 'network.inputs'))
        else:
            network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=3,
                                                                                    **self.conf.get_config(
                                                                                        'network.inputs'))
        network.to(device)

        # old_checkpnts_dir = os.path.join(self.expdir, self.timestamp, 'checkpoints')

        # ori version

        # foldername = conf.get_string('train.foldername').strip()

        # ckpt_prefix = '/mnt/sdf1/haog/code/IGR/exps/single_shape/'
        ckpt_prefix = '/mnt/data/haog/code/exps/single_shape/'
        # ckpt_prefix = 'exps/'
        save_prefix = '{}/'.format(args.pt)
        if args.aml:
            ckpt_prefix = '/blob/code/exps/single_shape/'
            save_prefix = '/blob/code/IGR/{}/'.format(args.pt)

        if not os.path.exists(save_prefix):
            os.mkdir(save_prefix)

        if args.cpu:
            saved_model_state = torch.load(ckpt_prefix + self.foldername + '/checkpoints/ModelParameters/latest.pth', map_location=device)
            network.load_state_dict(saved_model_state["model_state_dict"])
        else:
            saved_model_state = torch.load(ckpt_prefix + self.foldername + '/checkpoints/ModelParameters/latest.pth')
            network.load_state_dict(saved_model_state["model_state_dict"])
        print('loading finish')
        #trace
        example = torch.rand(224,3).to(device)
        traced_script_module = torch.jit.trace(network, example)
        # traced_script_module = torch.jit.trace(self.network, example)
        # traced_script_module = torch.jit.script(network)
        if onehot:
            traced_script_module.save(save_prefix + self.foldername + "_model_h_oh.pt")
        else:
            traced_script_module.save(save_prefix + self.foldername + "_model_h.pt")

    def visualize_loss(self):
        self.network.eval()
        #all losses list below
        h_loss_pt = torch.zeros(self.data.shape[0]).cuda()
        patch_loss_pt = torch.zeros(self.data.shape[0]).cuda()
        cs_loss_pt = torch.zeros(self.data.shape[0]).cuda()
        h_normal_loss_pt = torch.zeros(self.data.shape[0]).cuda()
        patch_normal_loss_pt = torch.zeros(self.data.shape[0]).cuda()
        normal_cs_loss_pt = torch.zeros(self.data.shape[0]).cuda()
        approx_h_loss_pt = torch.zeros(self.data.shape[0]).cuda()
        approx_patch_loss_pt = torch.zeros(self.data.shape[0]).cuda()
        approx_cs_loss_pt = torch.zeros(self.data.shape[0]).cuda()
        grad_loss_pt = torch.zeros(self.data.shape[0]).cuda()
        assignment_loss_pt = torch.zeros(self.data.shape[0]).cuda()

        # ln2loss = {'h_loss': h_loss_pt, \
        #     'patch_loss': patch_loss_pt,\
        #     'cs_loss': cs_loss_pt,\
        #     'h_normal_loss': h_normal_loss_pt,\
        #     'patch_normal_loss': patch_normal_loss_pt,\
        #     'normal_cs_loss': normal_cs_loss_pt,\
        #     'approx_h_loss': approx_h_loss_pt,\
        #     'approx_patch_loss': approx_patch_loss_pt,\
        #     'approx_cs_loss': approx_cs_loss_pt,\
        #     'grad_loss': grad_loss_pt,\
        #     'penalty_loss': assignment_loss_pt,
        #     }
        ln2loss = {'h_loss': h_loss_pt, \
            'patch_loss': patch_loss_pt,\
            'cs_loss': cs_loss_pt,\
            'h_normal_loss': h_normal_loss_pt,\
            'patch_normal_loss': patch_normal_loss_pt,\
            'normal_cs_loss': normal_cs_loss_pt,\
            'grad_loss': grad_loss_pt,\
            'penalty_loss': assignment_loss_pt,
            }

        allid = torch.arange(self.data.shape[0], dtype=torch.int64)
        n_branch = int(torch.max(self.feature_mask).item())
        n_batchsize = self.points_batch

        wrong_assign_count = torch.zeros(1).cuda()

        for i,indices in enumerate(torch.split(allid, n_batchsize)):
        # for epoch in range(self.startepoch, self.nepochs + 1):
            #indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            #first feature part then non-feature part
            # indices = torch.tensor(feature_id[0][np.random.choice(n_feature, n_feature_batch, True)]).cuda()
            # indices = torch.empty(0,dtype=torch.int64).cuda()
            #current feature mask
            # cur_feature_mask = torch.zeros(n_branch * n_patch_batch + n_feature_batch).cuda()
            # cur_feature_mask[0: n_feature_batch] = 1.0
            # cur_feature_mask = torch.zeros(n_batchsize).cuda()
            # for i in range(n_branch - 1):
            #     indices_nonfeature = torch.tensor(patch_id[i][np.random.choice(patch_id_n[i], n_patch_batch, True)]).cuda()
            #     indices = torch.cat((indices, indices_nonfeature), 0)
            # #last patch
            # indices_nonfeature = torch.tensor(patch_id[n_branch - 1][np.random.choice(patch_id_n[n_branch - 1], n_patch_last, True)]).cuda()
            # indices = torch.cat((indices, indices_nonfeature), 0)            

            cur_data = self.data[indices]
            cur_mask = self.feature_mask[indices]
            mnfld_pnts = cur_data[:, :self.d_in] #n_indices x 3
            mnfld_sigma = self.local_sigma[indices] #noise points

            # change back to train mode
            # self.network.train()
            # self.adjust_learning_rate(epoch)
            # nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            # nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            # nonmnfld_pred = nonmnfld_pred_all[:,0]
            #print shape
            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            # nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss
            mnfld_pred_devide_grad = mnfld_pred / mnfld_grad.norm(2, dim=1)
            approx_h_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]] = (mnfld_pred_devide_grad.abs())
            h_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]] = mnfld_pred.abs()

            all_fi = torch.zeros(indices.shape[0]).cuda()
            for j in range(n_branch):
                tmp_mask = (cur_mask == j + 1)
                all_fi[tmp_mask] = mnfld_pred_all[tmp_mask, j + 1]
                # all_fi[i * n_patch_batch : (i + 1) * n_patch_batch, 0] = mnfld_pred_all[i * n_patch_batch : (i + 1) * n_patch_batch, i + 1]

            print('all fi shape: ', all_fi.shape)
            # print('all fi shape: ', all_fi.shape)
            print ('{} {}'.format(i * n_batchsize, i * n_batchsize + indices.shape[0]))
            print('output shape', patch_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]].shape)
            print('patch loss shape: ', patch_loss_pt.shape)
            patch_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]] = all_fi.abs()
            assert(all_fi.shape == mnfld_pred.shape)
            cs_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]] = (all_fi - mnfld_pred).abs()
            normals = cur_data[:, -self.d_in:]
            h_normal_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]] = (mnfld_grad - normals).norm(2, dim=1)
            branch_grad = gradient(mnfld_pnts, all_fi)
            patch_normal_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]] = (branch_grad - normals).norm(2, dim=1)
            normal_cs_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]] = (mnfld_grad - branch_grad).norm(2, dim=1)
            grad_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]] = (mnfld_grad.norm(2, dim=-1)-1)**2
            all_fi_devide_grad = all_fi / branch_grad.norm(2, dim=1)
            approx_patch_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]] = (all_fi_devide_grad).abs()
            approx_cs_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]] = (mnfld_pred_devide_grad - all_fi_devide_grad).abs()

            #penalty loss
            # wrong_assignment_flag = (mnfld_pred != all_fi)
            wrong_assignment_flag = torch.abs(mnfld_pred - all_fi) > args.th_closeness

            mnfld_pred_abs = torch.abs(mnfld_pred)[wrong_assignment_flag]
            all_fi_abs = torch.abs(all_fi)[wrong_assignment_flag]

            # assignment_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]][wrong_assignment_flag] = 1.0/(mnfld_pred_abs - all_fi_abs + torch.sqrt((mnfld_pred_abs - all_fi_abs) * (mnfld_pred_abs - all_fi_abs) + 1e-6))
            assignment_loss_pt[i * n_batchsize: i * n_batchsize + indices.shape[0]][wrong_assignment_flag] = torch.exp(100 * torch.abs(mnfld_pred - all_fi)[wrong_assignment_flag])

            wrong_assign_count = wrong_assign_count + torch.sum(wrong_assignment_flag)
            
            
        # approx h loss
        print("self name: ", self.foldername)
        print("wrong assignment count: ", wrong_assign_count)
        for k in ln2loss:
            print("{} loss avg: {:6f} max: {:6f}".format(k,torch.mean(ln2loss[k]).item(), torch.max(ln2loss[k]).item()))
            
            save_prefix = '/mnt/sdf1/haog/code/IGR/exps/'
            if args.aml:
                save_prefix = '/blob/code/exps/'
            visualize_ptnormal_loss(save_prefix + 'visloss/{}_{}.ply'.format(self.foldername, k), self.data.detach().cpu().numpy(), ln2loss[k].detach().cpu().numpy())




    def get_mask(self, filename):
        self.network.eval()
        #all losses list below
        # ln2loss = {'h_loss': h_loss_pt, \
        #     'patch_loss': patch_loss_pt,\
        #     'cs_loss': cs_loss_pt,\
        #     'h_normal_loss': h_normal_loss_pt,\
        #     'patch_normal_loss': patch_normal_loss_pt,\
        #     'normal_cs_loss': normal_cs_loss_pt,\
        #     'approx_h_loss': approx_h_loss_pt,\
        #     'approx_patch_loss': approx_patch_loss_pt,\
        #     'approx_cs_loss': approx_cs_loss_pt,\
        #     'grad_loss': grad_loss_pt,\
        #     'penalty_loss': assignment_loss_pt,
        #     }

        n_branch = int(torch.max(self.feature_mask).item())
        n_batchsize = self.points_batch

        #batch version
        # in_folder = '/mnt/sdf1/haog/code/IGR/correspondence_data/1010'
        in_file = '/mnt/sdf1/haog/code/IGR/correspondence_data/1617/nomask_list_normallambda1_mnfldcs_center_assign_10000_begin_linassign_00001617_50k_color_34_nooct_fc.xyz'

        # fs = os.listdir(in_folder)
        # for f in fs:
        if True:
            # f = in_file
            # if f.endswith('.xyz'):
            if True:
                # in_file = os.path.join(in_folder, f)
                # input folder is set here
                # in_file = '/mnt/sdf1/haog/code/IGR/bb_fcdata/broken_bullet_input_50k.xyz'
                # in_file = '/mnt/sdf1/haog/code/IGR/bb_fcdata/bb_nocolor_broken_bullet_input_50k_234_nooct_fc.xyz'
                prefix = in_file.split('.')[0]

                data = torch.tensor(np.loadtxt(in_file)).float().cuda()
                print('shape: ', data.shape)
                allid = torch.arange(data.shape[0], dtype=torch.int64)
                output_mask = torch.zeros(data.shape[0], dtype=torch.int64).cuda()
                for i,indices in enumerate(torch.split(allid, n_batchsize)):
                # for epoch in range(self.startepoch, self.nepochs + 1):
                    #indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
                    #first feature part then non-feature part
                    # indices = torch.tensor(feature_id[0][np.random.choice(n_feature, n_feature_batch, True)]).cuda()
                    # indices = torch.empty(0,dtype=torch.int64).cuda()
                    #current feature mask
                    # cur_feature_mask = torch.zeros(n_branch * n_patch_batch + n_feature_batch).cuda()
                    # cur_feature_mask[0: n_feature_batch] = 1.0
                    # cur_feature_mask = torch.zeros(n_batchsize).cuda()
                    # for i in range(n_branch - 1):
                    #     indices_nonfeature = torch.tensor(patch_id[i][np.random.choice(patch_id_n[i], n_patch_batch, True)]).cuda()
                    #     indices = torch.cat((indices, indices_nonfeature), 0)
                    # #last patch
                    # indices_nonfeature = torch.tensor(patch_id[n_branch - 1][np.random.choice(patch_id_n[n_branch - 1], n_patch_last, True)]).cuda()
                    # indices = torch.cat((indices, indices_nonfeature), 0)            

                    cur_data = data[indices]
                    mnfld_pnts = cur_data[:, :self.d_in] #n_indices x 3
                    # mnfld_sigma = self.local_sigma[indices] #noise points

                    # change back to train mode
                    # self.network.train()
                    # self.adjust_learning_rate(epoch)
                    # nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
                    # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()

                    # forward pass
                    # print ('input shape', mnfld_pnts.shape)
                    mnfld_pred_all = self.network(mnfld_pnts)
                    # nonmnfld_pred_all = self.network(nonmnfld_pnts)
                    # print("mnfld_pnts size: ", mnfld_pnts.shape)
                    mnfld_pred = mnfld_pred_all[:,0].unsqueeze(1)
                    # nonmnfld_pred = nonmnfld_pred_all[:,0]
                    #print shape
                    # compute grad
                    # loss = torch.Tensor([0.0]).float().cuda()
                    pred_diff = (mnfld_pred_all[:,1:] - mnfld_pred).abs()
                    output_mask[indices] = torch.argmin(pred_diff, dim=1) + 1
                # np.savetxt(filename, output_mask.detach().cpu().numpy().astype(int))
                np.savetxt('{}.txt'.format(prefix), output_mask.detach().cpu().numpy().astype(int))

                mask = output_mask.detach().cpu().numpy().astype(int)
                plydata = prefix[:-3] + '.ply'
                mesh = trimesh.load(plydata)
                assert(mesh.faces.shape[0] == mask.shape[0])
                save_mesh_off(prefix + '.off', mesh.vertices, mesh.faces, mask)
                #save colored off

        return

    def run_baseline(self):
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
        n_patch_batch = n_batchsize // (n_branch + 1)
        n_feature_batch = n_batchsize - n_patch_batch * n_branch
        print("patch batch & feature batch: ", n_patch_batch, n_feature_batch)
        print('number of branches: ', n_branch)
        

        #first non feature then feature elements
        assignment_weight = 0.0
        omega_1 = 0.0
        omega_2 = 0.0
        omega_3 = 0.0
        omega_max = 1000.0
        a = 0.01
        a_max = 10.0
        beta_1 = 10
        beta_2 = 1
        eps = 0.01
        # beta_3_init = 0.02
        beta_3_init = 0.0
        beta_3_max = 0.2
        k = 2 #no breaking
        k_max = 12

        # print ("feature mask shape: ", feature_mask_cpu.shape)
        feature_id = np.where(feature_mask_cpu == 0)
        print ("feature id shape: ", feature_id[0].shape)
        n_feature = feature_id[0].shape[0]
        print ("n feature : ", n_feature)
        print ("n feature tyep: ", type(n_feature))
        print ("feature batch type: ", type(n_feature_batch))
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
            for i in range(n_branch + 1):
                self.network.flag_output = i + 1
                self.plot_shapes(epoch=self.startepoch, path=my_path, file_suffix = "_" + str(i), with_cuts = True)
            self.network.flag_output = 0
            self.plot_masks(epoch=self.startepoch, path=my_path, n_branch = n_branch, file_suffix = "_mask" + str(i))

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
            writer = SummaryWriter(os.path.join("runs", self.foldername))
        # set init a
        pow_index = self.startepoch // self.conf.get_int('train.checkpoint_frequency')
        if self.startepoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            pow_index = pow_index - 1
        a = pow(10, pow_index) * a
        beta_3_init = pow(10, pow_index) * beta_3_init
        if a > a_max:
            a = a_max
        if beta_3_init > beta_3_max:
            beta_3_init = beta_3_max

        print('summary status: ', args.summary)
        # branch mask is predefined
        branch_mask = torch.zeros(n_branch, n_branch * n_patch_batch + n_feature_batch).cuda()
        single_branch_mask_gt = torch.zeros(n_branch * n_patch_batch + n_feature_batch, n_branch).cuda()
        #for cross entropy loss, id start from 0
        single_branch_mask_id = torch.zeros([n_branch * n_patch_batch], dtype = torch.long).cuda()
        for i in range(n_branch):
            branch_mask[i, n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch] = 1.0
            single_branch_mask_gt[n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch, i] = 1.0
            single_branch_mask_id[i * n_patch_batch : (i + 1) * n_patch_batch] = i

        for epoch in range(self.startepoch, self.nepochs + 1):
            #indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            #first feature part then non-feature part
            indices = torch.tensor(feature_id[0][np.random.choice(n_feature, n_feature_batch, True)]).cuda()
            #current feature mask
            cur_feature_mask = torch.zeros(n_branch * n_patch_batch + n_feature_batch).cuda()
            cur_feature_mask[0: n_feature_batch] = 1.0
            for i in range(n_branch):
                indices_nonfeature = torch.tensor(patch_id[i][np.random.choice(patch_id_n[i], n_patch_batch, True)]).cuda()
                indices = torch.cat((indices, indices_nonfeature), 0)

            cur_data = self.data[indices]
            mnfld_pnts = cur_data[:, :self.d_in] #n_indices x 3
            mnfld_sigma = self.local_sigma[indices] #noise points

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                self.save_checkpoints(epoch)
                if a < a_max:
                    a = a * 10
                if beta_3_init < beta_3_max:
                    beta_3_init = beta_3_init * 10

            if epoch != 0 and epoch % self.conf.get_int('train.plot_frequency') == 0:
            # if epoch % self.conf.get_int('train.plot_frequency') == 0:
                # print('saving checkpoint: ', epoch)
                print('plot validation epoch: ', epoch)
                for i in range(n_branch + 1):
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

            # use non feature points
            indices_nonfeature = indices[n_feature_batch:]
            non_feature_pnts = self.data[indices_nonfeature][:, :self.d_in]
            non_feature_sigma = self.local_sigma[indices_nonfeature]
            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            # nonmnfld_pnts = self.sampler.get_points(non_feature_pnts.unsqueeze(0), non_feature_sigma.unsqueeze(0)).squeeze()

            # forward pass
            mnfld_pred_all = self.network(mnfld_pnts)
            nonmnfld_pred_all = self.network(nonmnfld_pnts)
            # print("mnfld_pnts size: ", mnfld_pnts.shape)
            mnfld_pred = mnfld_pred_all[:,0]
            nonmnfld_pred = nonmnfld_pred_all[:,0]
            #print shape
            # print("mnfld_pnts size: ")
            # print(mnfld_pnts.size())
            # print("mnfld_pred size: ")
            # print(mnfld_pred.size())

            # compute grad
            # loss = torch.Tensor([0.0]).float().cuda()
            loss = 0.0
            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)
            # manifold loss
            mnfld_loss = (mnfld_pred.abs()).mean()
            # print ("mnfld loss device: ", mnfld_loss.device)
            loss = loss + mnfld_loss
            # manifold loss for patches

            all_fi = torch.zeros(n_batchsize, 1).cuda() #copy with grads
            for i in range(n_branch):
                all_fi[n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch, 0] = mnfld_pred_all[n_feature_batch + i * n_patch_batch : n_feature_batch + (i + 1) * n_patch_batch, i + 1]

            # mnfld_loss_patch = 0.0
            # for i in range(n_branch):
                # mnfld_loss_patch = mnfld_loss_patch + (mnfld_pred_all[:,i + 1].abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
            mnfld_loss_patch = (all_fi[:,0].abs() * (1 - cur_feature_mask)).mean() * n_batchsize / (n_batchsize - n_feature_batch) 
            loss = loss + mnfld_loss_patch

            assignment_loss = torch.zeros(1).cuda()
            if assignment_weight > 0.0: 
                # for i in range(n_branch):
                    # assignment_loss = assignment_loss + ((mnfld_pred - mnfld_pred_all[:,i + 1]).abs() * branch_mask[i]).mean() * n_batchsize / n_patch_batch
                assignment_loss = ((mnfld_pred - all_fi).abs() * (1 - cur_feature_mask)).mean() * n_batchsize / (n_batchsize - n_feature_batch)
                
            loss = loss + assignment_weight * assignment_loss

            # eikonal loss for all
            grad_loss = torch.zeros(1)
            # for i in range(n_branch):
            #     single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,i + 1])
            #     grad_loss = grad_loss + ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            # loss = loss + self.grad_lambda * grad_loss
            # eikonal loss for h
            single_nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred_all[:,0])
            grad_loss_h = ((single_nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            loss = loss + self.grad_lambda * grad_loss_h

            # normals loss
            if self.with_normals:
                #all normals
                normals = cur_data[:, -self.d_in:]
                #defined for h
                # normals_loss = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                branch_grad = gradient(mnfld_pnts, all_fi[:,0])
                normals_loss = (((branch_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
                # normals_loss = 0.0
                # for i in range(n_branch):
                #     branch_grad = gradient(mnfld_pnts, mnfld_pred_all[:, i + 1])
                #     normals_loss = normals_loss + (((branch_grad - normals).abs()).norm(2, dim=1) * branch_mask[i]).mean() * (n_batchsize / n_patch_batch)
                loss = loss + self.normals_lambda * normals_loss

                mnfld_grad = gradient(mnfld_pnts, mnfld_pred_all[:, 0])
                normals_loss_h = (((mnfld_grad - normals).abs()).norm(2, dim=1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

                loss = loss + self.normals_lambda * normals_loss_h

            else:
                normals_loss = torch.zeros(1)
                normals_loss_h = torch.zeros(1)

            # feature loss
            feature_loss = (mnfld_pred.abs() * cur_feature_mask).mean() * n_batchsize / n_feature_batch

            #linear version
            loss = loss + omega_1 * feature_loss

            #mask loss
            mask_feature = mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            #vanilla version
            # for i in range(k - 1):
            #     mask_feature = mask_feature * mnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]

            # mask_loss = torch.abs(torch.sum(mask_feature, dim = 1) - 1).mean()
            # mask_loss = (torch.sum((mask_feature - single_branch_mask_gt).abs(), dim = 1) * (1 - cur_feature_mask) ).mean() * (n_batchsize / (n_batchsize - n_feature_batch))

            #cross entropy form
            # mask_loss = -(torch.sum(single_branch_mask_gt * torch.log(mask_feature), dim = 1) * (1 - cur_feature_mask)).mean() * (n_batchsize / (n_batchsize - n_feature_batch))
            ce_loss = nn.CrossEntropyLoss()
            mask_loss = ce_loss(mask_feature[n_feature_batch:,:], single_branch_mask_id)

            loss = loss + omega_2 * mask_loss

            # mask_loss expectation
            # mask_nonmfd = nonmnfld_pred_all[:, n_branch + 1: 2 * n_branch + 1]
            # mask_expectation = torch.abs(torch.sum(mask_nonmfd * mask_nonmfd, dim = 1) - 1).mean()
            # loss = loss + omega_3 * mask_expectation


            if epoch % 1000 == 0:
                k = 2 * k
                if k > k_max:
                    k = k_max

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            #tensorboard
            if args.summary == True:
                writer.add_scalar('Loss/Total loss', loss.item(), epoch)
                writer.add_scalar('Loss/Manifold loss', mnfld_loss.item(), epoch)
                writer.add_scalar('Loss/Manifold patch loss', mnfld_loss_patch.item(), epoch)
                writer.add_scalar('Loss/Grad loss all',self.grad_lambda * grad_loss.item(), epoch)
                writer.add_scalar('Loss/Grad loss h',self.grad_lambda * grad_loss_h.item(), epoch)
                # writer.add_scalar('Loss/Grad loss G',g_grad_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss all', self.normals_lambda * normals_loss.item(), epoch)
                writer.add_scalar('Loss/Normal loss h', self.normals_lambda * normals_loss_h.item(), epoch)
                writer.add_scalar('Loss/feature loss', omega_1 * feature_loss.item(), epoch)
                writer.add_scalar('Loss/mask loss', omega_2 * mask_loss.item(), epoch)
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
                print ('feature loss: {:.6f}\t assignment_loss {:.6f}\t'.format(feature_loss.item(), assignment_loss.item()))               
                print ('k: ', k, " mask loss: ", mask_loss.item())

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

                #ori
                # plot_cuts(points=pnts,
                #           decoder=self.network,
                #           path=path,
                #           epoch=epoch,
                #           suffix = file_suffix,
                #           near_zero=False)
                # plot_cuts(points=pnts,
                #           decoder=self.network,
                #           path=path,
                #           epoch=epoch,
                #           suffix = '_zero' + file_suffix,
                #           near_zero=True)

    def plot_masks(self, epoch, n_branch, path=None, file_suffix="all"):
        with torch.no_grad():
            self.network.eval()
            if not path:
                path = self.plots_dir
            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            pnts = self.data[indices, :3]
            #draw nonmnfld pts
            mnfld_sigma = self.local_sigma[indices]
            nonmnfld_pnts = self.sampler.get_points(pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            pnts = nonmnfld_pnts
            plot_masks(  points=pnts,
                         decoder=self.network,
                         n_branch = n_branch,
                         path=path,
                         epoch=epoch,
                         suffix = file_suffix)

    def plot_masks_maxsdf(self, epoch, n_branch, path=None, file_suffix="all"):
        with torch.no_grad():
            self.network.eval()
            if not path:
                path = self.plots_dir
            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))
            pnts = self.data[indices, :3]
            #draw nonmnfld pts
            mnfld_sigma = self.local_sigma[indices]
            nonmnfld_pnts = self.sampler.get_points(pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()
            pnts = nonmnfld_pnts
            plot_masks_maxsdf(  points=pnts,
                         decoder=self.network,
                         n_branch = n_branch,
                         path=path,
                         epoch=epoch,
                         suffix = file_suffix)


    def __init__(self, **kwargs):

        self.home_dir = os.path.abspath(os.pardir)

        flag_list = False
        if 'flag_list' in kwargs:
            flag_list = True
        print ('flag list: ', flag_list)

        # config setting

        if type(kwargs['conf']) == str:
            self.conf_filename = './conversion/' + kwargs['conf']
            self.conf = ConfigFactory.parse_file(self.conf_filename)
        else:
            self.conf = kwargs['conf']

        self.expname = kwargs['expname']

        # GPU settings

        self.GPU_INDEX = kwargs['gpu_index']

        # if not self.GPU_INDEX == 'ignore':
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)
        # if not self.GPU_INDEX == 'ignore':
            

        self.num_of_gpus = torch.cuda.device_count()

        self.eval = kwargs['eval']
        self.evaldist = kwargs['evaldist']

        # settings for loading an existing experiment
        #20200927: if checkpoint exists in the folder, then continue, otherwise not

        # if (kwargs['is_continue'] or self.eval) and kwargs['timestamp'] == 'latest':
        #     if os.path.exists(os.path.join(self.home_dir, 'exps', self.expname)):
        #         timestamps = os.listdir(os.path.join(self.home_dir, 'exps', self.expname))
        #         if (len(timestamps)) == 0:
        #             is_continue = False
        #             timestamp = None
        #         else:
        #             timestamp = sorted(timestamps)[-1]
        #             is_continue = True
        #     else:
        #         is_continue = False
        #         timestamp = None
        # else:
        #     timestamp = kwargs['timestamp']
        #     is_continue = kwargs['is_continue'] or self.eval



        self.exps_folder_name = 'exps'

        utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name)))

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
            #not considering testing part
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


        self.expdir = utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name, self.expname))
        utils.mkdir_ifnotexists(self.expdir)

        # if is_continue:
        #     self.timestamp = timestamp
        # else:
        #     self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.timestamp = self.foldername

        self.cur_exp_dir = os.path.join(self.expdir, self.timestamp)
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

        # if args.n_branch == -1:
        # self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=self.d_in, split_pos = kwargs['splitpos'], 
        #                                                                             n_branch = int(torch.max(self.feature_mask).item()),
        #                                                                             **self.conf.get_config(
        #                                                                                 'network.inputs'))

        if not args.ori:
            
            #ori version
            self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=self.d_in, split_pos = kwargs['splitpos'], 
                                                                                    n_branch = int(torch.max(self.feature_mask).item()),
                                                                                    csg_tree = self.csg_tree,
                                                                                    flag_convex = self.csg_flag_convex,
                                                                                    **self.conf.get_config(
                                                                                        'network.inputs'))


            #test version            
            # self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=self.d_in, split_pos = kwargs['splitpos'], 
            #                                                                         n_branch = int(torch.max(self.feature_mask).item()) + 2,
            #                                                                         csg_tree = self.csg_tree,
            #                                                                         flag_convex = self.csg_flag_convex,
            #                                                                         **self.conf.get_config(
            #                                                                             'network.inputs'))
        else:
            self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=self.d_in,
                                                                                    **self.conf.get_config(
                                                                                        'network.inputs'))
        # else:
        #     self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=self.d_in, split_pos = kwargs['splitpos'], 
        #                                                                             n_branch = self.n_branch,
        #                                                                             **self.conf.get_config(
        #                                                                                 'network.inputs'))       
        # # self.network.split_pos = kwargs['splitpos']
        # self.network.n_branch = int(torch.max(self.feature_mask).item())


        print (self.network)
        # summary(self.network, (3,1))

        # print (list(self.network.sdf_0.parameters()))
        # for name, param in self.network.named_parameters():
        #     print (name, param.data[0].size())
        #     print ('type: ', type(param))

        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))
        self.weight_decay = self.conf.get_float('train.weight_decay')

        self.startepoch = 0

        if args.stage == 2:
            # self.optimizer = torch.optim.Adam(
            #     [
            #         {
            #             "params": list(self.network.sdf_0.parameters()) + list(self.network.sdf_1.parameters()) + list(self.network.sdf_2.parameters()) + list(self.network.sdf_3.parameters()),
            #             "lr": self.lr_schedules[0].get_learning_rate(0),
            #             "weight_decay": self.weight_decay
            #         },
            #     ])

            #with mask
            params_sdf = []
            params_mask = []
            params_svm = []
            # for name, param in self.network.named_parameters():
            #     if name.startswith('sdf'):
            #         params_sdf = params_sdf + list(param)
            #     else:
            #         params_mask = params_mask + list(param)

            #might need to be modified according to svm
            for k, v in self.network.state_dict().items():
                if k.startswith('sdf'):
                    params_sdf = params_sdf + list(v)
                elif k.startswith('mask'):
                    params_mask = params_mask + list(v)
                elif k.startswith('svm'):
                    params_svm = params_svm + list(v)

            #only optimize sdf
            # self.optimizer = torch.optim.Adam(
            #     [
            #         {
            #             # "params": list(self.network.sdf_0.parameters()) + list(self.network.sdf_1.parameters()) + list(self.network.sdf_2.parameters()) + list(self.network.sdf_3.parameters()),
            #             "params": params_sdf,
            #             "lr": self.lr_schedules[0].get_learning_rate(0),
            #             "weight_decay": self.weight_decay
            #         },
            #     ])


            #with mask

            self.optimizer = torch.optim.Adam(
                [
                    {
                        # "params": list(self.network.sdf_0.parameters()) + list(self.network.sdf_1.parameters()) + list(self.network.sdf_2.parameters()) + list(self.network.sdf_3.parameters()),
                        "params": params_sdf,
                        "lr": self.lr_schedules[0].get_learning_rate(0),
                        "weight_decay": self.weight_decay
                    },
                    {
                        # "params": list(self.network.mask_0.parameters()) + list(self.network.mask_1.parameters()) + list(self.network.mask_2.parameters()) + list(self.network.mask_3.parameters()) + list(self.network.mask_4.parameters()) + list(self.network.mask_5.parameters()) + list(self.network.mask_6.parameters()) + list(self.network.mask_7.parameters()),
                        "params": params_mask,
                        "lr": self.lr_schedules[1].get_learning_rate(0),
                        "weight_decay": self.weight_decay
                    },
                    {
                        # "params": list(self.network.mask_0.parameters()) + list(self.network.mask_1.parameters()) + list(self.network.mask_2.parameters()) + list(self.network.mask_3.parameters()) + list(self.network.mask_4.parameters()) + list(self.network.mask_5.parameters()) + list(self.network.mask_6.parameters()) + list(self.network.mask_7.parameters()),
                        "params": params_svm,
                        "lr": self.lr_schedules[1].get_learning_rate(0),
                        "weight_decay": self.weight_decay
                    }
                ])

        else:
            self.optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.network.parameters(),
                        "lr": self.lr_schedules[0].get_learning_rate(0),
                        "weight_decay": self.weight_decay
                    },
                ])

        #for MLP_SDF
        # self.optimizer = torch.optim.Adam(
        #     [
        #         {
        #             "params": list(self.network.sdf_0.parameters()) + list(self.network.sdf_1.parameters()) + list(self.network.sdf_2.parameters()) + list(self.network.sdf_3.parameters()),
        #             "lr": self.lr_schedules[0].get_learning_rate(0),
        #             "weight_decay": 0.0001
        #         },
        #         {
        #             "params": list(self.network.mask_0.parameters()) + list(self.network.mask_1.parameters()) + list(self.network.mask_2.parameters()) + list(self.network.mask_3.parameters()),
        #             "lr": self.lr_schedules[0].get_learning_rate(0),
        #             "weight_decay": 0
        #         }
        #     ])

        # if continue load checkpoints

        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            print('loading checkpoint from: ', old_checkpnts_dir)
            if args.stage == 2:
                saved_model_state = torch.load(
                    os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
                self.network.load_state_dict(saved_model_state["model_state_dict"])
                self.startepoch = saved_model_state['epoch']
            else:
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

# origin version
# if __name__ == '__main__':


#     if args.gpu == "auto":
#         deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
#                                     excludeUUID=[])
#         gpu = deviceIDs[0]
#     else:
#         gpu = args.gpu


#     nepoch = args.nepoch
#     if args.stage == 2:
#         nepoch = 2 * args.nepoch


#     trainrunner = ReconstructionRunner(
#             conf=args.conf,
#             points_batch=args.points_batch,
#             nepochs=nepoch,
#             expname=args.expname,
#             gpu_index=gpu,
#             is_continue=args.is_continue,
#             timestamp=args.timestamp,
#             checkpoint=args.checkpoint,
#             eval=args.eval,
#             splitpos = args.splitpos,
#             evaldist = args.evaldist
#     )

#     # trainrunner.run()
#     # trainrunner.run_multi_branch()
#     # trainrunner.run_multi_branch_mlp()
#     # trainrunner.run_multi_branch_mask()
#     # if args.baseline:
#     # trainrunner.run_multi_branch_mask_nofea()
#     if args.stage == 1:
#         # trainrunner.run_multi_branch_mask_nofea_maskonly()
#         trainrunner.run_multi_branch_mask_nofea_maskonly_svm()
#     elif args.stage == 2:
#         # trainrunner.run_multi_branch_mask_nofea_sdfonly()
#         trainrunner.run_multi_branch_mask_nofea()

#     else:
#         trainrunner.run_multi_branch_maxsdf_nomask()

#new version
if __name__ == '__main__':


    if args.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = args.gpu


    nepoch = args.nepoch
    if args.stage == 2:
        nepoch = 2 * args.nepoch

    conf = ConfigFactory.parse_file('./conversion/' + args.conf)
    folderprefix = conf.get_string('train.folderprefix')
    fileprefix_list = conf.get_list('train.fileprefix_list')
    print ('file number: ', len(fileprefix_list))

    trainrunners = []
    for i in range(len(fileprefix_list)):
        fp = fileprefix_list[i]
        print ('cur model: ', fp)

        #check if exists
        # conf_path = os.path.join('/mnt/sdf1/haog/code/IGR/CADdata_noise/', fp + '_csg.conf')
        # if not os.path.exists(conf_path):
        #     continue

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
            if not args.ori:
                trainrunners[i].run_multi_branch_maxsdf_nomask()
                # trainrunners[i].tracing()
            else:
                trainrunners[i].run_ori()

            end = time.time()
            dur = end - begin

            if args.baseline:
                fp = fp+"_bl"

            # not saving time tmp
            # f = open('{}_timing.txt'.format(fp),'w')
            # f.write(str(dur))
            # f.close()
            # trainrunners[i].run_multi_branch_maxsdf_nomask_approx()
