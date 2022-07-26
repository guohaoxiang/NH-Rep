import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad

#borrowed from siren paper
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad

class NHRepNet(nn.Module):
    def __init__(
        self,
        d_in,
        dims_sdf,
        csg_tree,
        skip_in=(),
        flag_convex = True,
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        flag_output = 0,
        n_branch = 2
    ):
        super().__init__()

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.n_branch = n_branch
        self.csg_tree = csg_tree
        self.flag_convex = flag_convex
        self.skip_in = skip_in

        dims_sdf = [d_in] + dims_sdf + [self.n_branch]
        self.sdf_layers = len(dims_sdf)
        for layer in range(0, len(dims_sdf) - 1):
            if layer + 1 in skip_in:
                out_dim = dims_sdf[layer + 1] - d_in
            else:
                out_dim = dims_sdf[layer + 1]
            lin = nn.Linear(dims_sdf[layer], out_dim)
            if geometric_init:
                if layer == self.sdf_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims_sdf[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            setattr(self, "sdf_"+str(layer), lin)
        if geometric_init:
            if beta > 0:
                self.activation = nn.Softplus(beta=beta)
            # vanilla relu
            else:
                self.activation = nn.ReLU()
        else:
            #siren
            self.activation = Sine()
        self.final_activation = nn.ReLU()

    # composite f_i to h
    def nested_cvx_output(self, value_matrix, list_operation, cvx_flag=True):
        list_value = []
        for v in list_operation:
            if type(v) != list:
                list_value.append(v)
        op_mat = torch.zeros(value_matrix.shape[1],len(list_value)).cuda()
        for i in range(len(list_value)):
            op_mat[list_value[i]][i] = 1.0

        mat_mul = torch.matmul(value_matrix, op_mat)
        if len(list_operation) == len(list_value):
            # leaf node
            if cvx_flag:
                return torch.max(mat_mul, 1)[0].unsqueeze(1)
            else:
                return torch.min(mat_mul, 1)[0].unsqueeze(1)
        else:
            list_output = [mat_mul]
            for v in list_operation:
                if type(v) == list:
                    list_output.append(self.nested_cvx_output(value_matrix, v, not cvx_flag))
            if cvx_flag:
                return torch.max(torch.cat(list_output, 1), 1)[0].unsqueeze(1)
            else:
                return torch.min(torch.cat(list_output, 1), 1)[0].unsqueeze(1)

    def min_soft_blend(self, mat, rho):
        res = mat[:,0]
        for i in range(1, mat.shape[1]):
            srho = res * res + mat[:,i] * mat[:,i] - rho * rho
            res = res + mat[:,i] - torch.sqrt(res * res + mat[:,i] * mat[:,i] + 1.0/(8 * rho * rho) * srho * (srho - srho.abs()))
        return res.unsqueeze(1)

    def max_soft_blend(self, mat, rho):
        res = mat[:,0]
        for i in range(1, mat.shape[1]):
            srho = res * res + mat[:,i] * mat[:,i] - rho * rho
            res = res + mat[:,i] + torch.sqrt(res * res + mat[:,i] * mat[:,i] + 1.0/(8 * rho * rho) * srho * (srho - srho.abs()))
        return res.unsqueeze(1)

    #r-function blending
    def nested_cvx_output_soft_blend(self, value_matrix, list_operation, cvx_flag=True):
        rho = 0.05
        list_value = []
        for v in list_operation:
            if type(v) != list:
                list_value.append(v)
        op_mat = torch.zeros(value_matrix.shape[1],len(list_value)).cuda()
        for i in range(len(list_value)):
            op_mat[list_value[i]][i] = 1.0

        mat_mul = torch.matmul(value_matrix, op_mat)
        if len(list_operation) == len(list_value):
            # leaf node
            if cvx_flag:
                return self.max_soft_blend(mat_mul, rho)
            else:
                return self.min_soft_blend(mat_mul, rho)
        else:
            list_output = [mat_mul]
            for v in list_operation:
                if type(v) == list:
                    list_output.append(self.nested_cvx_output_soft_blend(value_matrix, v, not cvx_flag))
            if cvx_flag:
                return self.max_soft_blend(torch.cat(list_output, 1), rho)
            else:
                return self.min_soft_blend(torch.cat(list_output, 1), rho)


    def forward(self, input):
        x = input
        for layer in range(0, self.sdf_layers - 1):
            lin = getattr(self, "sdf_" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)
            if layer < self.sdf_layers - 2:
                x = self.activation(x)
        output_value = x #all f_i

        h = self.nested_cvx_output(output_value, self.csg_tree, self.flag_convex)
        # r-function blending
        # h = self.nested_cvx_output_soft_blend(output_value, self.csg_tree, self.flag_convex)
        
        if self.flag_output == 0:
            return torch.cat((h, output_value), 1) # return all
        elif self.flag_output == 1:
            return h #return h
        else:
            return output_value[:, self.flag_output - 2] #return f_i