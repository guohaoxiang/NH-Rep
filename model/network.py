import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad

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

class ImplicitNetMultiBranch(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.split_pos = split_pos
        self.n_branch = n_branch

        for b in range(0, n_branch):
            for layer in range(0, self.num_layers - 1):

                if layer + 1 in skip_in:
                    out_dim = dims[layer + 1] - d_in
                else:
                    out_dim = dims[layer + 1]
                lin = nn.Linear(dims[layer], out_dim)
                # if true preform preform geometric initialization
                if geometric_init:
                    if layer == self.num_layers - 2:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                        torch.nn.init.constant_(lin.bias, -radius_init)
                    else:
                        torch.nn.init.constant_(lin.bias, 0.0)
                        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                else:
                    #siren
                    in_dim = dims[layer]
                    if layer == 0:
                        #first layer
                        torch.nn.init.uniform_(lin.weight, -1 / in_dim, 1 / in_dim)
                    else:
                        torch.nn.init.uniform_(lin.weight, -np.sqrt(6 / in_dim) / 30, np.sqrt(6 / in_dim) / 30)

                # print ("b :", b)
                # print ("layer: ", layer)
                setattr(self, "lin_" + str(b) + "_" + str(layer), lin)
                # print ("setattr: ", "lin_" + str(b) + "_" + str(layer))

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

    def forward(self, input):
        split_pos = self.split_pos #not consider temperarily
        output_value = torch.zeros(input.shape[0], self.n_branch).cuda()
        for b in range(0, self.n_branch):
            x = input
            for layer in range(0, self.num_layers - 1):
                # print ("setattr: ", "lin_" + str(b) + "_" + str(layer))
                lin = getattr(self, "lin_" + str(b) + "_" + str(layer))
                if layer in self.skip_in:
                    x = torch.cat([x, input], -1) / np.sqrt(2)
                x = lin(x)
                if layer < self.num_layers - 2:
                    x = self.activation(x)
                #for split pos
                # if b == 0 and layer == split_pos:
                #     intermediate = x[b]
            # print ("x shape: ", x.shape)
            # print ("output value: ", output_value.shape)
            output_value[:, b] = x[:, 0]
        if self.flag_output == 0:
            return torch.cat((torch.max(output_value, 1).values.unsqueeze(1), output_value),1)
        elif self.flag_output == 1:
            return torch.max(output_value, 1).values.unsqueeze(1)
        else:
            return output_value[:, self.flag_output - 2]

class ImplicitNetMultiBranchInitTransform(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.split_pos = split_pos
        self.n_branch = n_branch

        for b in range(0, n_branch):
            for layer in range(0, self.num_layers - 1):

                if layer + 1 in skip_in:
                    out_dim = dims[layer + 1] - d_in
                else:
                    out_dim = dims[layer + 1]
                lin = nn.Linear(dims[layer], out_dim)
                # if true preform preform geometric initialization
                if geometric_init:
                    if layer == self.num_layers - 2:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                        torch.nn.init.constant_(lin.bias, -radius_init)
                    else:
                        torch.nn.init.constant_(lin.bias, 0.0)
                        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                else:
                    #siren
                    in_dim = dims[layer]
                    if layer == 0:
                        #first layer
                        torch.nn.init.uniform_(lin.weight, -1 / in_dim, 1 / in_dim)
                    else:
                        torch.nn.init.uniform_(lin.weight, -np.sqrt(6 / in_dim) / 30, np.sqrt(6 / in_dim) / 30)

                # print ("b :", b)
                # print ("layer: ", layer)
                setattr(self, "lin_" + str(b) + "_" + str(layer), lin)
                # print ("setattr: ", "lin_" + str(b) + "_" + str(layer))

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

    def forward(self, input):
        split_pos = self.split_pos #not consider temperarily
        # output_value = torch.zeros(input.shape[0], self.n_branch).cuda()
        output_value = []
        #test code for exp 6 & 7
        # transforms = torch.tensor([[-0.25,0,0],[0.25,0,0]]).cuda()
        # transforms = torch.tensor([[0,0,0],[-0.5,-0.5,-0.5]]).cuda()

        # transforms = transforms.unsqueeze(1).repeat(1,input.shape[0],1)
        # transforms = np.array([[0,0,0],[-0.5,-0.5,-0.5]])
        coeff = np.array([1.0,-1.0])

        for b in range(0, self.n_branch):
            # x = input + transforms[b]
            x = input
            # x[:,0] = x[:,0] + transforms[b,0]
            # x[:,1] = x[:,1] + transforms[b,1]
            # x[:,2] = x[:,2] + transforms[b,2]

            for layer in range(0, self.num_layers - 1):
                # print ("setattr: ", "lin_" + str(b) + "_" + str(layer))
                lin = getattr(self, "lin_" + str(b) + "_" + str(layer))
                if layer in self.skip_in:
                    x = torch.cat([x, input], -1) / np.sqrt(2)
                x = lin(x)
                if layer < self.num_layers - 2:
                    x = self.activation(x)
                #for split pos
                # if b == 0 and layer == split_pos:
                #     intermediate = x[b]
            # print ("x shape: ", x.shape)
            # print ("output value: ", output_value.shape)
            
            # output_value[:, b] = x[:, 0]

            output_value.append(x)
            # output_value = x
        #test code for maxsdf 6&7:
        output_value = torch.cat(output_value, 1)
        # output_value[:,1] = -output_value[:,1]
        # output_value[:][1] = -output_value[:][1]

        # inverse_id = torch.tensor([5,7,8], dtype=torch.int64)
        # regular_id = torch.tensor([0,1,2,3,4,6], dtype=torch.int64)
        # omega_0 = torch.max(output_value[:,regular_id], 1)[0]
        # I_1 = torch.max(output_value[:,inverse_id], 1)[0]

        #replace above code with matmul
        inverse_id_mat = torch.zeros(9, 3).cuda()
        regular_id_mat = torch.zeros(9, 6).cuda()
        inverse_id_mat[5][0] = 1.0
        inverse_id_mat[7][1] = 1.0
        inverse_id_mat[8][2] = 1.0
        regular_id_mat[0][0] = 1.0
        regular_id_mat[1][1] = 1.0
        regular_id_mat[2][2] = 1.0
        regular_id_mat[3][3] = 1.0
        regular_id_mat[4][4] = 1.0
        regular_id_mat[6][5] = 1.0
        omega_0 = torch.max(torch.matmul(output_value, regular_id_mat), 1)[0]
        I_1 = torch.max(torch.matmul(output_value, inverse_id_mat), 1)[0]

        h = torch.max(omega_0, -I_1)

        if self.flag_output == 0:
            return torch.cat((h.unsqueeze(1), output_value),1)
        elif self.flag_output == 1:
            return h.unsqueeze(1)
        else:
            return output_value[:, self.flag_output - 2]

class ImplicitNetMultiBranchMLP(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2,
        mlp_relu = True
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.split_pos = split_pos
        self.n_branch = n_branch

        for b in range(0, n_branch):
            for layer in range(0, self.num_layers - 1):

                if layer + 1 in skip_in:
                    out_dim = dims[layer + 1] - d_in
                else:
                    out_dim = dims[layer + 1]
                lin = nn.Linear(dims[layer], out_dim)
                # if true preform preform geometric initialization
                if geometric_init:
                    if layer == self.num_layers - 2:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                        torch.nn.init.constant_(lin.bias, -radius_init)
                    else:
                        torch.nn.init.constant_(lin.bias, 0.0)
                        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                else:
                    #siren
                    in_dim = dims[layer]
                    if layer == 0:
                        #first layer
                        torch.nn.init.uniform_(lin.weight, -1 / in_dim, 1 / in_dim)
                    else:
                        torch.nn.init.uniform_(lin.weight, -np.sqrt(6 / in_dim) / 30, np.sqrt(6 / in_dim) / 30)

                # print ("b :", b)
                # print ("layer: ", layer)
                setattr(self, "lin_" + str(b) + "_" + str(layer), lin)
                # print ("setattr: ", "lin_" + str(b) + "_" + str(layer))

        #latter part of the network
        dims_mlp = [self.n_branch + 3, 256, 256, 256, self.n_branch]
        self.mlp_layers = len(dims_mlp)
        for layer in range(0, len(dims_mlp) - 1):
            out_dim = dims_mlp[layer + 1]
            lin = nn.Linear(dims_mlp[layer], out_dim)
            setattr(self, "mlp_"+str(layer), lin)

        if geometric_init:
            if beta > 0:
                self.activation = nn.Softplus(beta=beta)

            # vanilla relu
            else:
                self.activation = nn.ReLU()
        else:
            #siren
            self.activation = Sine()

        if mlp_relu:
            self.mlp_activation = nn.ReLU()
        else:
            self.mlp_activation = nn.Softplus(beta=beta)


        self.final_activation = nn.ReLU()

    def forward(self, input):
        split_pos = self.split_pos #not consider temperarily
        output_value = torch.zeros(input.shape[0], self.n_branch).cuda()
        for b in range(0, self.n_branch):
            x = input
            for layer in range(0, self.num_layers - 1):
                # print ("setattr: ", "lin_" + str(b) + "_" + str(layer))
                lin = getattr(self, "lin_" + str(b) + "_" + str(layer))
                if layer in self.skip_in:
                    x = torch.cat([x, input], -1) / np.sqrt(2)
                x = lin(x)
                if layer < self.num_layers - 2:
                    x = self.activation(x)
                #for split pos
                # if b == 0 and layer == split_pos:
                #     intermediate = x[b]
            # print ("x shape: ", x.shape)
            # print ("output value: ", output_value.shape)
            output_value[:, b] = x[:, 0]

        # k_value = torch.zeros(input.shape[0], self.n_branch).cuda()
        #concate value with input
        x = torch.cat((output_value, input),1)
        for layer in range(0, self.mlp_layers - 1):
            lin = getattr(self, "mlp_" + str(layer))
            x = lin(x)
            if layer < self.mlp_layers - 2:
                x = self.mlp_activation(x)
        output_value = torch.cat((output_value, x),1)

        if self.flag_output == 0:
            return torch.cat((torch.max(x, 1).values.unsqueeze(1), output_value),1)
        elif self.flag_output == 1:
            return torch.max(x, 1).values.unsqueeze(1)
        else:
            return output_value[:, self.flag_output - 2]

class ImplicitNetMultiBranchMask(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2,
        flag_onehot = False #not one hot
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.split_pos = split_pos
        self.n_branch = n_branch
        self.flag_onehot = flag_onehot

        for b in range(0, n_branch):
            for layer in range(0, self.num_layers - 1):

                if layer + 1 in skip_in:
                    out_dim = dims[layer + 1] - d_in
                else:
                    out_dim = dims[layer + 1]
                lin = nn.Linear(dims[layer], out_dim)
                # if true preform preform geometric initialization
                if geometric_init:
                    if layer == self.num_layers - 2:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                        torch.nn.init.constant_(lin.bias, -radius_init)
                    else:
                        torch.nn.init.constant_(lin.bias, 0.0)
                        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                else:
                    #siren
                    in_dim = dims[layer]
                    if layer == 0:
                        #first layer
                        torch.nn.init.uniform_(lin.weight, -1 / in_dim, 1 / in_dim)
                    else:
                        torch.nn.init.uniform_(lin.weight, -np.sqrt(6 / in_dim) / 30, np.sqrt(6 / in_dim) / 30)

                # print ("b :", b)
                # print ("layer: ", layer)
                setattr(self, "lin_" + str(b) + "_" + str(layer), lin)
                # print ("setattr: ", "lin_" + str(b) + "_" + str(layer))

        #latter part of the network
        dims_mask = [3, 256, 256, 256, self.n_branch]
        self.mask_layers = len(dims_mask)
        for layer in range(0, len(dims_mask) - 1):
            out_dim = dims_mask[layer + 1]
            lin = nn.Linear(dims_mask[layer], out_dim)
            setattr(self, "mask_"+str(layer), lin)
        if geometric_init:
            if beta > 0:
                self.activation = nn.Softplus(beta=beta)
            # vanilla relu
            else:
                self.activation = nn.ReLU()
        else:
            #siren
            self.activation = Sine()
        self.mask_activation = nn.Softplus(beta=beta)
        self.mask_softmax = nn.Softmax(dim=1)
        self.final_activation = nn.ReLU()

    def forward(self, input):
        split_pos = self.split_pos #not consider temperarily
        output_value = torch.zeros(input.shape[0], self.n_branch).cuda()
        for b in range(0, self.n_branch):
            x = input
            for layer in range(0, self.num_layers - 1):
                # print ("setattr: ", "lin_" + str(b) + "_" + str(layer))
                lin = getattr(self, "lin_" + str(b) + "_" + str(layer))
                if layer in self.skip_in:
                    x = torch.cat([x, input], -1) / np.sqrt(2)
                x = lin(x)
                if layer < self.num_layers - 2:
                    x = self.activation(x)
                #for split pos
                # if b == 0 and layer == split_pos:
                #     intermediate = x[b]
            # print ("x shape: ", x.shape)
            # print ("output value: ", output_value.shape)
            output_value[:, b] = x[:, 0]

        # k_value = torch.zeros(input.shape[0], self.n_branch).cuda()
        x = input
        for layer in range(0, self.mask_layers - 1):
            lin = getattr(self, "mask_" + str(layer))
            x = lin(x)
            if layer < self.mask_layers - 2:
                x = self.mask_activation(x)
        x = self.mask_softmax(x)
        if not self.flag_onehot:
            h = torch.sum(x * output_value, dim = 1)
            # h = torch.max(output_value, dim = 1)[0]
        else:
            # one hot
            # multiple max value might emerge
            mask_bool = (x == x.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
            h = output_value[mask_bool]
            if not h.shape[0] == input.shape[0]:
                x = x + torch.randn(x.shape).cuda() * 1e-6
                mask_bool = (x == x.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
                h = output_value[mask_bool]
            if not h.shape[0] == input.shape[0]:
                print ("mask bool sum: ", mask_bool.sum())
                print ("h input shape: ", h.shape[0], " ", input.shape[0])
            assert(h.shape[0] == input.shape[0])
            # max_index = x.argmax(dim = 1)
            # h = torch.zeros(input.shape[0]).cuda()
            # for i in range(input.shape[0]):
            #     h[i] = output_value[i, max_index[i]]
        # onehot case needs to be implemented
        output_value = torch.cat((output_value, x),1)

        if self.flag_output == 0:
            return torch.cat((h.unsqueeze(1), output_value), 1)
        elif self.flag_output == 1:
            return h.unsqueeze(1)
        else:
            return output_value[:, self.flag_output - 2]

class ImplicitNetParallel(nn.Module):
    def __init__(
        self,
        d_in,
        dims_sdf,
        dims_mask,
        skip_in=(),
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2,
        flag_onehot = False, #not one hot
        flag_softmax = False
    ):
        super().__init__()

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.n_branch = n_branch
        self.flag_onehot = flag_onehot
        self.flag_softmax  = False

        #network, no skip connection
        # dims_sdf = [3, 256, 256, 256, self.n_branch] #short version
        dims_sdf = [d_in] + dims_sdf + [self.n_branch]
        # dims_sdf = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        # dims_mask = [3, 256, 256, 256, self.n_branch]
        # dims_mask = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        dims_mask = [d_in] + dims_mask + [self.n_branch]

        self.sdf_layers = len(dims_sdf)
        self.mask_layers = len(dims_mask)
        for layer in range(0, len(dims_sdf) - 1):
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

        for layer in range(0, len(dims_mask) - 1):
            out_dim = dims_mask[layer + 1]
            lin = nn.Linear(dims_mask[layer], out_dim)
            setattr(self, "mask_"+str(layer), lin)
        if geometric_init:
            if beta > 0:
                self.activation = nn.Softplus(beta=beta)
            # vanilla relu
            else:
                self.activation = nn.ReLU()
        else:
            #siren
            self.activation = Sine()
        self.mask_activation = nn.Softplus(beta=beta)
        self.mask_softmax = nn.Softmax(dim=1)
        self.final_activation = nn.ReLU()

    def forward(self, input):
        x = input
        for layer in range(0, self.sdf_layers - 1):
            lin = getattr(self, "sdf_" + str(layer))
            x = lin(x)
            if layer < self.sdf_layers - 2:
                x = self.activation(x)
        output_value = x
        x = input
        for layer in range(0, self.mask_layers - 1):
            lin = getattr(self, "mask_" + str(layer))
            x = lin(x)
            if layer < self.mask_layers - 2:
                x = self.mask_activation(x)
        output_mask_pre = x
        output_mask = self.mask_softmax(x)
        if not self.flag_onehot:
            h = torch.sum(output_mask * output_value, dim = 1)
            # h = torch.max(output_value, dim = 1)[0]
        else:
            # one hot
            # multiple max value might emerge
            mask_bool = (output_mask == output_mask.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
            h = output_value[mask_bool]
            if not h.shape[0] == input.shape[0]:
                output_mask = output_mask + torch.randn(x.shape).cuda() * 1e-6
                mask_bool = (output_mask == output_mask.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
                h = output_value[mask_bool]
            if not h.shape[0] == input.shape[0]:
                print ("mask bool sum: ", mask_bool.sum())
                print ("h input shape: ", h.shape[0], " ", input.shape[0])
            assert(h.shape[0] == input.shape[0])

        if self.flag_softmax:
            output_value = torch.cat((output_value, output_mask),1)
        else:
            output_value = torch.cat((output_value, output_mask_pre),1)

        if self.flag_output == 0:
            return torch.cat((h.unsqueeze(1), output_value), 1)
        elif self.flag_output == 1:
            return h.unsqueeze(1)
        else:
            return output_value[:, self.flag_output - 2]

class ImplicitNetParallelSVM(nn.Module):
    def __init__(
        self,
        d_in,
        dims_sdf,
        dims_mask,
        skip_in=(),
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2,
        flag_onehot = False, #not one hot
        n_svm_feature = 9,
        argmax_beta = 5,
    ):
        super().__init__()

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.flag_onehot = flag_onehot
        self.n_branch = n_branch
        self.argmax_beta = argmax_beta
        #network, no skip connection
        # dims_sdf = [3, 256, 256, 256, self.n_branch] #short version
        dims_sdf = [d_in] + dims_sdf + [self.n_branch]
        # dims_sdf = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        # dims_mask = [3, 256, 256, 256, self.n_branch]
        # dims_mask = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        dims_mask = [d_in] + dims_mask + [n_svm_feature]

        self.sdf_layers = len(dims_sdf)
        self.mask_layers = len(dims_mask)
        for layer in range(0, len(dims_sdf) - 1):
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

        for layer in range(0, len(dims_mask) - 1):
            out_dim = dims_mask[layer + 1]
            lin = nn.Linear(dims_mask[layer], out_dim)
            setattr(self, "mask_"+str(layer), lin)
        self.svms = nn.ModuleList()
        for i in range(self.n_branch):
            in_dim = n_svm_feature
            out_dim = 1
            lin = nn.Linear(in_dim, out_dim)
            self.svms.append(lin)

        if geometric_init:
            if beta > 0:
                self.activation = nn.Softplus(beta=beta)
            # vanilla relu
            else:
                self.activation = nn.ReLU()
        else:
            #siren
            self.activation = Sine()
        self.mask_activation = nn.Softplus(beta=beta)
        self.mask_softmax = nn.Softmax(dim=1)
        self.final_activation = nn.ReLU()

    def forward(self, input):
        x = input
        for layer in range(0, self.sdf_layers - 1):
            lin = getattr(self, "sdf_" + str(layer))
            x = lin(x)
            if layer < self.sdf_layers - 2:
                x = self.activation(x)
        output_value = x
        x = input
        for layer in range(0, self.mask_layers - 1):
            lin = getattr(self, "mask_" + str(layer))
            x = lin(x)
            if layer < self.mask_layers - 2:
                x = self.mask_activation(x)
        output_mask_pre = x
        output_mask = torch.zeros(input.shape[0], self.n_branch).cuda()

        #get output_mask
        # dist_p2s = torch.zeros(input.shape[0], self.n_branch).cuda()
        dist_p2s = []
        for i in range(self.n_branch):
            # lin = getattr(self, "svm_" + str(i))
            lin = self.svms[i]
            # dist_p2s[:,i] = lin(output_mask_pre)[:,0]
            dist_p2s.append(lin(output_mask_pre))
        dist_p2s = torch.cat(dist_p2s, 1)
        h = torch.zeros(input.shape[0]).cuda()

        if self.flag_onehot:
            dist_bool = (dist_p2s == dist_p2s.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
            h = output_value[dist_bool]
            if not h.shape[0] == input.shape[0]:
                dist_bool_sum = torch.sum(dist_bool, dim = 1)
                errid = dist_bool_sum > 1.9
                dist_bool_part = dist_bool[errid]
                print("multiple array size: ", dist_bool_part.shape[0])
                dist_bool_part_new = torch.zeros(dist_bool_part.shape).type(torch.BoolTensor)
                for i in range(dist_bool_part.shape[0]):
                    for j in range(self.n_branch):
                        if dist_bool_part[i][j] == True:
                            dist_bool_part_new[i][j] = True
                            break
                dist_bool[errid] = dist_bool_part_new
                h = output_value[dist_bool]
            # for cpu export
            # output_mask = torch.zeros(input.shape[0], self.n_branch) 
            output_mask[dist_bool] = 1.0
        else:
            #ori version
            # dist_max = dist_p2s.max(dim = 1, keepdim = True)[0].expand(-1, self.n_branch)
            # dist_min = dist_p2s.min(dim = 1, keepdim = True)[0].expand(-1, self.n_branch)
            # dist_p2s_n = (dist_p2s - dist_min)/(dist_max - dist_min)
            # dist_p2s_n = torch.pow(dist_p2s_n, 8)
            # dist_p2s_sum = torch.sum(dist_p2s_n, 1, keepdim=True).expand(-1,self.n_branch)
            # output_mask = dist_p2s_n / dist_p2s_sum

            #softmax version
            output_mask = self.mask_softmax(self.argmax_beta * dist_p2s)


            h = torch.sum(output_value * output_mask, 1)

        output_value = torch.cat((output_value, output_mask),1)

        # output_mask = self.mask_softmax(x)
        # if not self.flag_onehot:
        #     h = torch.sum(output_mask * output_value, dim = 1)
        #     # h = torch.max(output_value, dim = 1)[0]
        # else:
        #     # one hot
        #     # multiple max value might emerge
        #     mask_bool = (output_mask == output_mask.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
        #     h = output_value[mask_bool]
        #     if not h.shape[0] == input.shape[0]:
        #         output_mask = output_mask + torch.randn(x.shape).cuda() * 1e-6
        #         mask_bool = (output_mask == output_mask.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
        #         h = output_value[mask_bool]
        #     if not h.shape[0] == input.shape[0]:
        #         print ("mask bool sum: ", mask_bool.sum())
        #         print ("h input shape: ", h.shape[0], " ", input.shape[0])
        #     assert(h.shape[0] == input.shape[0])

        if self.flag_output == 0:
            return torch.cat((h.unsqueeze(1), output_value, dist_p2s), 1)
        elif self.flag_output == 1:
            return h.unsqueeze(1)
        else:
            return output_value[:, self.flag_output - 2]


# class ImplicitNetParallelPack(nn.Module):


class ImplicitNetParallelNoMask(nn.Module):
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
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2,
        flag_onehot = False #not one hot
    ):
        super().__init__()

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.n_branch = n_branch
        self.flag_onehot = flag_onehot
        self.flag_softmax  = False
        self.csg_tree = csg_tree
        self.flag_convex = flag_convex
        self.skip_in = skip_in

        #network, no skip connection
        # dims_sdf = [3, 256, 256, 256, self.n_branch] #short version
        dims_sdf = [d_in] + dims_sdf + [self.n_branch]
        # dims_sdf = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        # dims_mask = [3, 256, 256, 256, self.n_branch]
        # dims_mask = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
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

    def min_soft(self, mat, m = 0):
        #mat: mxn
        #return: mx1
        if m == 0:
            res = mat[:,0]
            for i in range(1, mat.shape[1]):
                res = res + mat[:,i] - torch.sqrt(res * res + mat[:,i] * mat[:,i])
        else:
            res = mat[:,0]
            for i in range(1, mat.shape[1]):
                # res = res + mat[:,i] - torch.sqrt(res * res + mat[:,i] * mat[:,i])
                tmp = torch.sqrt(res * res + mat[:,i] * mat[:,i])
                res = (res + mat[:,i] - tmp) * torch.pow(tmp, m)

        return res.unsqueeze(1)

    def max_soft(self, mat, m = 0):
        #mat: mxn
        #return: mx1
        if m == 0:
            res = mat[:,0]
            for i in range(1, mat.shape[1]):
                res = res + mat[:,i] + torch.sqrt(res * res + mat[:,i] * mat[:,i])
        else:
            res = mat[:,0]
            for i in range(1, mat.shape[1]):
                # res = res + mat[:,i] - torch.sqrt(res * res + mat[:,i] * mat[:,i])
                tmp = torch.sqrt(res * res + mat[:,i] * mat[:,i])
                res = (res + mat[:,i] + tmp) * torch.pow(tmp, m)


        return res.unsqueeze(1)


    def nested_cvx_output_soft(self, value_matrix, list_operation, cvx_flag=True):
        degree = 0

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
                # return torch.max(mat_mul, 1)[0].unsqueeze(1)
                return self.max_soft(mat_mul, degree)
            else:
                # return torch.min(mat_mul, 1)[0].unsqueeze(1)
                return self.min_soft(mat_mul, degree)
        else:
            list_output = [mat_mul]
            for v in list_operation:
                if type(v) == list:
                    list_output.append(self.nested_cvx_output_soft(value_matrix, v, not cvx_flag))
                    # list_output.append(self.nested_cvx_output(value_matrix, v, not cvx_flag))

            if cvx_flag:
                # return torch.max(torch.cat(list_output, 1), 1)[0].unsqueeze(1)
                return self.max_soft(torch.cat(list_output, 1), degree)
            else:
                # return torch.min(torch.cat(list_output, 1), 1)[0].unsqueeze(1)
                return self.min_soft(torch.cat(list_output, 1), degree)


    def min_soft_blend(self, mat, rho):
        #mat: mxn
        #return: mx1
        # rho = 0.25
        # rho = 0.05
        # rho = 0.02



        if True:
            res = mat[:,0]
            for i in range(1, mat.shape[1]):
                # res = res + mat[:,i] - torch.sqrt(res * res + mat[:,i] * mat[:,i])
                srho = res * res + mat[:,i] * mat[:,i] - rho * rho
                res = res + mat[:,i] - torch.sqrt(res * res + mat[:,i] * mat[:,i] + 1.0/(8 * rho * rho) * srho * (srho - srho.abs()))

            #modified 0516
            # srho = (mat*mat).sum(-1) - rho * rho
            # res = mat.sum(-1) - torch.sqrt(srho + rho * rho + 1.0/(8 * rho * rho) * srho * (srho - srho.abs()))

        return res.unsqueeze(1)

    def max_soft_blend(self, mat, rho):
        #mat: mxn
        #return: mx1
        # rho = 0.25
        # rho = 0.05
        # rho = 0.02


        if True:
            res = mat[:,0]
            for i in range(1, mat.shape[1]):
                # res = res + mat[:,i] + torch.sqrt(res * res + mat[:,i] * mat[:,i])
                srho = res * res + mat[:,i] * mat[:,i] - rho * rho
                res = res + mat[:,i] + torch.sqrt(res * res + mat[:,i] * mat[:,i] + 1.0/(8 * rho * rho) * srho * (srho - srho.abs()))

            #modified 0516
            # srho = (mat*mat).sum(-1) - rho * rho
            # res = mat.sum(-1) + torch.sqrt(srho + rho * rho + 1.0/(8 * rho * rho) * srho * (srho - srho.abs()))


        return res.unsqueeze(1)

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
                # return torch.max(mat_mul, 1)[0].unsqueeze(1)
                return self.max_soft_blend(mat_mul, rho)
            else:
                # return torch.min(mat_mul, 1)[0].unsqueeze(1)
                return self.min_soft_blend(mat_mul, rho)
        else:
            list_output = [mat_mul]
            for v in list_operation:
                if type(v) == list:
                    list_output.append(self.nested_cvx_output_soft_blend(value_matrix, v, not cvx_flag))
            if cvx_flag:
                # return torch.max(torch.cat(list_output, 1), 1)[0].unsqueeze(1)
                return self.max_soft_blend(torch.cat(list_output, 1), rho)
            else:
                # return torch.min(torch.cat(list_output, 1), 1)[0].unsqueeze(1)
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
        output_value = x


        #ori version
        #defined here
        # self.csg_tree = [4,5]
        # self.flag_convex = False
        # self.csg_tree = [3,4]
        # self.flag_convex = False

        #define
        
        # self.csg_tree = [1,2,3,4,12]
        # self.flag_convex = False

        # self.csg_tree = [[11,12,]]

        

        h = self.nested_cvx_output(output_value, self.csg_tree, self.flag_convex)
        # h = self.nested_cvx_output_soft(output_value, self.csg_tree, self.flag_convex)
        # h = self.nested_cvx_output_soft_blend(output_value, self.csg_tree, self.flag_convex) #blend version


        
        #changed version:
        # output_value[:,0] = input[:,0]
        # output_value[:,1] = input[:,2]
        # output_value[:,2] = torch.sqrt(input[:,0] * input[:,0] + input[:,2] * input[:,2]) - 0.6
        # tmp = torch.min(output_value[:,:2],1)[0].unsqueeze(1)
        # h = tmp
        # h = torch.max(torch.cat([output_value[:,2].unsqueeze(1), tmp], 1),1)[0].unsqueeze(1)
        #changed below
        # h = self.nested_cvx_output(output_value, [1,3,[7,9]], False)


        # h = torch.sign(h)


        # if self.flag_output == 0:
        #     return torch.cat((h.unsqueeze(1), output_value), 1)
        # elif self.flag_output == 1:
        #     return h.unsqueeze(1)
        # else:
        #     return output_value[:, self.flag_output - 2]

        #ori version
        if self.flag_output == 0:
            return torch.cat((h, output_value), 1)
        elif self.flag_output == 1:
            return h
        else:
            return output_value[:, self.flag_output - 2]
        
        #test version
        # maxlasttwo = torch.max(output_value[:, -2:], 1)[0].unsqueeze(1)
        # maxlasttwo = torch.min(output_value[:, -2:], 1)[0].unsqueeze(1)

        # maxlasttwo2 = torch.min(output_value[:, -4:-2], 1)[0].unsqueeze(1)

        # maxlasttwo = torch.max(output_value[:, -2:], 1)[0].unsqueeze(1)

        # maxlasttwo2 = torch.max(output_value[:, -4:-2], 1)[0].unsqueeze(1)

        # output_value_new = torch.cat([output_value[:,:-4], maxlasttwo2 ,maxlasttwo], 1)


        # if self.flag_output == 0:
        #     return torch.cat((h, output_value_new), 1)
        # elif self.flag_output == 1:
        #     return h
        # else:
        #     return output_value[:, self.flag_output - 2]

class ImplicitNetMultiBranchNoMask(nn.Module):
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
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2,
        flag_onehot = False #not one hot
    ):
        super().__init__()

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.n_branch = n_branch
        self.flag_onehot = flag_onehot
        self.flag_softmax  = False
        self.csg_tree = csg_tree
        self.flag_convex = flag_convex

        #network, no skip connection
        # dims_sdf = [3, 256, 256, 256, self.n_branch] #short version
        dims_sdf = [d_in] + dims_sdf + [1]
        # dims_sdf = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        # dims_mask = [3, 256, 256, 256, self.n_branch]
        # dims_mask = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        self.sdf_layers = len(dims_sdf)
        for b in range(0, n_branch):
            for layer in range(0, len(dims_sdf) - 1):
                out_dim = dims_sdf[layer + 1]
                lin = nn.Linear(dims_sdf[layer], out_dim)
                if geometric_init:
                    if layer == self.sdf_layers - 2:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims_sdf[layer]), std=0.00001)
                        torch.nn.init.constant_(lin.bias, -radius_init)
                    else:
                        torch.nn.init.constant_(lin.bias, 0.0)
                        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                setattr(self, "sdf_" + str(b) + '_'+str(layer), lin)

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

    def forward(self, input):
        x = input
        # output_value = torch.zeros(input.shape[0], self.n_branch).cuda()
        output_value = []
        for b in range(self.n_branch):
            x = input
            for layer in range(0, self.sdf_layers - 1):
                lin = getattr(self, "sdf_" + str(b) + '_' + str(layer))
                x = lin(x)
                if layer < self.sdf_layers - 2:
                    x = self.activation(x)
            # output_value[:,b] = x[:,0]
            output_value.append(x)
        output_value = torch.cat(output_value, 1)
        h = self.nested_cvx_output(output_value, self.csg_tree, self.flag_convex)
        if self.flag_output == 0:
            return torch.cat((h, output_value), 1)
        elif self.flag_output == 1:
            return h
        else:
            return output_value[:, self.flag_output - 2]


class ImplicitNetParallelMaxSDF(nn.Module):
    def __init__(
        self,
        d_in,
        dims_sdf,
        dims_mask,
        skip_in=(),
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2,
        flag_onehot = False #not one hot
    ):
        super().__init__()

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.n_branch = n_branch
        self.flag_onehot = True
        self.flag_softmax  = True
        self.flag_halfneg = True

        #network, no skip connection
        # dims_sdf = [3, 256, 256, 256, self.n_branch] #short version
        dims_sdf = [d_in] + dims_sdf + [self.n_branch]
        # dims_sdf = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        # dims_mask = [3, 256, 256, 256, self.n_branch]
        # dims_mask = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        dims_mask = [d_in] + dims_mask + [self.n_branch * 3]

        self.sdf_layers = len(dims_sdf)
        self.mask_layers = len(dims_mask)
        for layer in range(0, len(dims_sdf) - 1):
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

        for layer in range(0, len(dims_mask) - 1):
            out_dim = dims_mask[layer + 1]
            lin = nn.Linear(dims_mask[layer], out_dim)
            setattr(self, "mask_"+str(layer), lin)
        if geometric_init:
            if beta > 0:
                self.activation = nn.Softplus(beta=beta)
            # vanilla relu
            else:
                self.activation = nn.ReLU()
        else:
            #siren
            self.activation = Sine()
        self.mask_activation = nn.Softplus(beta=beta)
        self.mask_softmax = nn.Softmax(dim=1)
        self.final_activation = nn.ReLU()

    def forward(self, input):
        x = input
        for layer in range(0, self.sdf_layers - 1):
            lin = getattr(self, "sdf_" + str(layer))
            x = lin(x)
            if layer < self.sdf_layers - 2:
                x = self.activation(x)
        output_value = x
        if self.flag_halfneg:
            begin_id = (self.n_branch + 1) // 2
            output_value[:, begin_id:] = -output_value[:, begin_id:]
        x = input
        for layer in range(0, self.mask_layers - 1):
            lin = getattr(self, "mask_" + str(layer))
            x = lin(x)
            if layer < self.mask_layers - 2:
                x = self.mask_activation(x)
        # output_mask_pre = x
        # output_mask = self.mask_softmax(x)
        output_mask_dist = torch.zeros(input.shape[0], self.n_branch * 3).cuda()
        output_mask = torch.zeros(input.shape[0], self.n_branch).cuda()
        for b in range(0, self.n_branch):
            output_mask_dist[:, 3*b : 3*(b+1)] = self.mask_softmax(x[:, 3*b : 3*(b+1)])
            output_mask[:,b] = output_mask_dist[:, 3*b + 2] - output_mask_dist[:, 3*b]

        h = torch.max(output_mask * output_value, dim = 1)[0]
        output_value = torch.cat((output_value, output_mask_dist),1)

        if self.flag_output == 0:
            return torch.cat((h.unsqueeze(1), output_value), 1)
        elif self.flag_output == 1:
            return h.unsqueeze(1)
        else:
            return output_value[:, self.flag_output - 2]

class ImplicitNetParallelDropout(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2,
        flag_onehot = False #not one hot
    ):
        super().__init__()

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.n_branch = n_branch
        self.flag_onehot = flag_onehot
        self.flag_softmax  = False

        #network, no skip connection
        dims_sdf = [3, 256, 256, 256, self.n_branch] #short version
        # dims_sdf = [3, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        dims_mask = [3, 256, 256, 256, self.n_branch]
        # dims_mask = [3, 256, 256, 256, 256, 256, self.n_branch] #long version

        self.sdf_layers = len(dims_sdf)
        self.mask_layers = len(dims_mask)
        for layer in range(0, len(dims_sdf) - 1):
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
            if layer < self.sdf_layers - 2:
                setattr(self, "drop_"+str(layer), nn.Dropout(0.01))

        for layer in range(0, len(dims_mask) - 1):
            out_dim = dims_mask[layer + 1]
            lin = nn.Linear(dims_mask[layer], out_dim)
            setattr(self, "mask_"+str(layer), lin)
        if geometric_init:
            if beta > 0:
                self.activation = nn.Softplus(beta=beta)
            # vanilla relu
            else:
                self.activation = nn.ReLU()
        else:
            #siren
            self.activation = Sine()
        self.mask_activation = nn.Softplus(beta=beta)
        self.mask_softmax = nn.Softmax(dim=1)
        self.final_activation = nn.ReLU()

    def forward(self, input):
        x = input
        for layer in range(0, self.sdf_layers - 1):
            lin = getattr(self, "sdf_" + str(layer))
            x = lin(x)
            if layer < self.sdf_layers - 2:
                drop = getattr(self, "drop_" + str(layer))
                x = self.activation(drop(x))
        output_value = x
        x = input
        for layer in range(0, self.mask_layers - 1):
            lin = getattr(self, "mask_" + str(layer))
            x = lin(x)
            if layer < self.mask_layers - 2:
                x = self.mask_activation(x)
        output_mask_pre = x
        output_mask = self.mask_softmax(x)
        if not self.flag_onehot:
            h = torch.sum(output_mask * output_value, dim = 1)
            # h = torch.max(output_value, dim = 1)[0]
        else:
            # one hot
            # multiple max value might emerge
            mask_bool = (output_mask == output_mask.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
            h = output_value[mask_bool]
            if not h.shape[0] == input.shape[0]:
                output_mask = output_mask + torch.randn(x.shape).cuda() * 1e-6
                mask_bool = (output_mask == output_mask.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
                h = output_value[mask_bool]
            if not h.shape[0] == input.shape[0]:
                print ("mask bool sum: ", mask_bool.sum())
                print ("h input shape: ", h.shape[0], " ", input.shape[0])
            assert(h.shape[0] == input.shape[0])
        # output_value = torch.cat((output_value, output_mask),1)
        if self.flag_softmax:
            output_value = torch.cat((output_value, output_mask),1)
        else:
            output_value = torch.cat((output_value, output_mask_pre),1)

        if self.flag_output == 0:
            return torch.cat((h.unsqueeze(1), output_value), 1)
        elif self.flag_output == 1:
            return h.unsqueeze(1)
        else:
            return output_value[:, self.flag_output - 2]

class ImplicitNetMaskSDF(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2,
        flag_onehot = False #not one hot
    ):
        super().__init__()

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.n_branch = n_branch
        self.flag_onehot = flag_onehot

        #network, no skip connection
        dims_sdf = [3 + self.n_branch, 256, 256, 256, self.n_branch] #short version
        # dims_sdf = [3 + self.n_branch, 256, 256, 256, 256, 256, 256, 256, self.n_branch] #long version
        dims_mask = [3, 256, 256, 256, self.n_branch]
        # dims_mask = [3, 256, 256, 256, 256, 256, self.n_branch] #long version

        self.sdf_layers = len(dims_sdf)
        self.mask_layers = len(dims_mask)
        for layer in range(0, len(dims_sdf) - 1):
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

        for layer in range(0, len(dims_mask) - 1):
            out_dim = dims_mask[layer + 1]
            lin = nn.Linear(dims_mask[layer], out_dim)
            setattr(self, "mask_"+str(layer), lin)
        if geometric_init:
            if beta > 0:
                self.activation = nn.Softplus(beta=beta)
            # vanilla relu
            else:
                self.activation = nn.ReLU()
        else:
            #siren
            self.activation = Sine()
        self.mask_activation = nn.Softplus(beta=beta)
        self.mask_softmax = nn.Softmax(dim=1)
        self.final_activation = nn.ReLU()

    def forward(self, input):
        x = input
        for layer in range(0, self.mask_layers - 1):
            lin = getattr(self, "mask_" + str(layer))
            x = lin(x)
            if layer < self.mask_layers - 2:
                x = self.mask_activation(x)
        output_mask = self.mask_softmax(x)
        x = torch.cat((output_mask, input), 1)
        for layer in range(0, self.sdf_layers - 1):
            lin = getattr(self, "sdf_" + str(layer))
            x = lin(x)
            if layer < self.sdf_layers - 2:
                x = self.activation(x)
        output_value = x
        
        if not self.flag_onehot:
            h = torch.sum(output_mask * output_value, dim = 1)
            # h = torch.max(output_value, dim = 1)[0]
        else:
            # one hot
            # multiple max value might emerge
            mask_bool = (output_mask == output_mask.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
            h = output_value[mask_bool]
            if not h.shape[0] == input.shape[0]:
                output_mask = output_mask + torch.randn(x.shape).cuda() * 1e-6
                mask_bool = (output_mask == output_mask.max(dim = 1, keepdim = True)[0]).type(torch.BoolTensor)
                h = output_value[mask_bool]
            if not h.shape[0] == input.shape[0]:
                print ("mask bool sum: ", mask_bool.sum())
                print ("h input shape: ", h.shape[0], " ", input.shape[0])
            assert(h.shape[0] == input.shape[0])
        output_value = torch.cat((output_value, output_mask),1)

        if self.flag_output == 0:
            return torch.cat((h.unsqueeze(1), output_value), 1)
        elif self.flag_output == 1:
            return h.unsqueeze(1)
        else:
            return output_value[:, self.flag_output - 2]

class ImplicitNetOri(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=1,
        beta=100
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, input):

        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x


class ImplicitNet(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True, #set false for siren
        radius_init=1,
        beta=100,
        # beta=0,
        flag_output = 0,
        split_pos = -1,
        n_branch = 2 #not used
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.flag_output = flag_output #0: all 1: h, 2: f, 3: g
        self.split_pos = split_pos
        self.n_branch = n_branch #not used

        #for f
        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                #siren
                in_dim = dims[layer]
                if layer == 0:
                    #first layer
                    torch.nn.init.uniform_(lin.weight, -1 / in_dim, 1 / in_dim)
                else:
                    torch.nn.init.uniform_(lin.weight, -np.sqrt(6 / in_dim) / 30, np.sqrt(6 / in_dim) / 30)




            setattr(self, "lin" + str(layer), lin)

        #second part, for g
        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
               #siren
                in_dim = dims[layer]
                if layer == 0:
                    #first layer
                    torch.nn.init.uniform_(lin.weight, -1 / in_dim, 1 / in_dim)
                else:
                    torch.nn.init.uniform_(lin.weight, -np.sqrt(6 / in_dim) / 30, np.sqrt(6 / in_dim) / 30)


            setattr(self, "lin_g" + str(layer), lin)

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

    def forward(self, input):

# init version of forwarding
#         x = input

#         for layer in range(0, self.num_layers - 1):

#             lin = getattr(self, "lin" + str(layer))

#             if layer in self.skip_in:
#                 x = torch.cat([x, input], -1) / np.sqrt(2)

#             x = lin(x)

#             if layer < self.num_layers - 2:
#                 x = self.activation(x)

# #second part, for g
#         xg = input
#         for layer in range(0, self.num_layers - 1):

#             lin = getattr(self, "lin_g" + str(layer))

#             if layer in self.skip_in:
#                 xg = torch.cat([xg, input], -1) / np.sqrt(2)

#             xg = lin(xg)

#             if layer < self.num_layers - 2:
#                 xg = self.activation(xg)

#network sharing, assert only one skip link exists
        split_pos = self.split_pos
        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

            if layer == split_pos:
                intermediate = x

#second part, for g
        if split_pos == -1:
            xg = input
        else:
            xg = intermediate
        for layer in range(split_pos + 1, self.num_layers - 1):

            lin = getattr(self, "lin_g" + str(layer))

            if layer in self.skip_in:
                xg = torch.cat([xg, input], -1) / np.sqrt(2)

            xg = lin(xg)

            if layer < self.num_layers - 2:
                xg = self.activation(xg)

        # x = x + nn.ReLU((xg))
        coeff_f = 0
        coeff_g = 0
        coeff_expg = 0 #setting it to 1 is meaningless
        coeff_max = 1

        a = 10
        if self.flag_output == 0:
            return torch.cat((coeff_f * x + coeff_g * self.final_activation(xg) + coeff_expg * torch.exp( -a * torch.abs(xg)) + coeff_max * torch.max(x, xg), x, xg), 1) 
        elif self.flag_output == 1:
            return coeff_f * x + coeff_g * self.final_activation(xg) + coeff_expg * torch.exp( -a * torch.abs(xg)) + coeff_max * torch.max(x, xg)
        elif self.flag_output == 2:
            return x
        elif self.flag_output == 3:
            return xg

