import plotly.graph_objs as go
import plotly.offline as offline
import torch
import numpy as np
from skimage import measure
import os
import utils.general as utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from conversion.save_implicit_function_vtk import save_implicit_function_vtk

def get_threed_scatter_trace(points,caption = None,colorscale = None,color = None):

    if (type(points) == list):
        trace = [go.Scatter3d(
            x=p[0][:, 0],
            y=p[0][:, 1],
            z=p[0][:, 2],
            mode='markers',
            name=p[1],
            marker=dict(
                size=3,
                line=dict(
                    width=2,    
                ),
                opacity=0.9,
                colorscale=colorscale,
                showscale=True,
                color=color,
            ), text=caption) for p in points]

    else:

        trace = [go.Scatter3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            mode='markers',
            name='projection',
            marker=dict(
                size=3,
                line=dict(
                    width=2,
                ),
                opacity=0.9,
                colorscale=colorscale,
                showscale=True,
                color=color,
            ), text=caption)]

    return trace


def plot_threed_scatter(points,path,epoch,in_epoch):
    trace = get_threed_scatter_trace(points)
    layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                                           yaxis=dict(range=[-2, 2], autorange=False),
                                                           zaxis=dict(range=[-2, 2], autorange=False),
                                                           aspectratio=dict(x=1, y=1, z=1)))

    fig1 = go.Figure(data=trace, layout=layout)

    filename = '{0}/scatter_iteration_{1}_{2}.html'.format(path, epoch, in_epoch)
    offline.plot(fig1, filename=filename, auto_open=False)


def plot_surface(decoder,path,epoch, shapename,resolution,mc_value,is_uniform_grid,verbose,save_html,save_ply,overwrite, points=None, with_points=False, latent=None, connected=False, suffix = "all"):

    filename = '{0}/igr_{1}_{2}'.format(path, epoch, shapename)

    if (not os.path.exists(filename) or overwrite):

        if with_points:
            pnts_val = decoder(points)
            print ("pnts size: ", pnts_val.shape)
            # modified on 20200922
            # pnts_val_all = decoder(points)
            # pnts_val = pnts_val_all[:,0]
            pnts_val = pnts_val.cpu()
            points = points.cpu()
            caption = ["decoder : {0}".format(val.item()) for val in pnts_val.squeeze()]
            trace_pnts = get_threed_scatter_trace(points[:,-3:],caption=caption)

        surface = get_surface_trace(points,decoder,latent,resolution,mc_value,is_uniform_grid,verbose,save_ply, connected)
        trace_surface = surface["mesh_trace"]

        layout = go.Layout(title= go.layout.Title(text=shapename), width=1200, height=1200, scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                                               yaxis=dict(range=[-2, 2], autorange=False),
                                                               zaxis=dict(range=[-2, 2], autorange=False),
                                                               aspectratio=dict(x=1, y=1, z=1)))
        if (with_points):
            fig1 = go.Figure(data=trace_pnts + trace_surface, layout=layout)
        else:
            fig1 = go.Figure(data=trace_surface, layout=layout)


        if (save_html):
            offline.plot(fig1, filename=filename + suffix + '.html', auto_open=False)
        if (not surface['mesh_export'] is None):
            surface['mesh_export'].export(filename + suffix + '.ply', 'ply')
        return surface['mesh_export']


def get_surface_trace(points,decoder,latent,resolution,mc_value,is_uniform,verbose,save_ply, connected=False):

    trace = []
    meshexport = None

    if (is_uniform):
        grid = get_grid_uniform(resolution)
    else:
        if not points is None:
            grid = get_grid(points[:,-3:],resolution)
        else:
            grid = get_grid(None, resolution)

    z = []

    for i,pnts in enumerate(torch.split(grid['grid_points'],100000,dim=0)):
        if (verbose):
            print ('{0}'.format(i/(grid['grid_points'].shape[0] // 100000) * 100))

        if (not latent is None):
            pnts = torch.cat([latent.expand(pnts.shape[0], -1), pnts], dim=1)
        z.append(decoder(pnts).detach().cpu().numpy())
        # z.append(decoder(pnts)[:,0].detach().cpu().numpy())
    z = np.concatenate(z,axis=0)

    if (not (np.min(z) > mc_value or np.max(z) < mc_value)):

        import trimesh
        z  = z.astype(np.float64)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=mc_value,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0],grid['xyz'][1][0],grid['xyz'][2][0]])
        if (save_ply):
            meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
            if connected:
                connected_comp = meshexport.split(only_watertight=False)
                max_area = 0
                max_comp = None
                for comp in connected_comp:
                    if comp.area > max_area:
                        max_area = comp.area
                        max_comp = comp
                meshexport = max_comp

        def tri_indices(simplices):
            return ([triplet[c] for triplet in simplices] for c in range(3))

        I, J, K = tri_indices(faces)

        trace.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                          i=I, j=J, k=K, name='',
                          color='orange', opacity=0.5))
    #trace and export are the same
    return {"mesh_trace":trace,
            "mesh_export":meshexport}


def plot_cuts_axis(points,decoder,latent,path,epoch,near_zero,axis,file_name_sep='/'):
    onedim_cut = np.linspace(-1.0, 1.0, 200)
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    min_axis = points[:,axis].min(dim=0)[0].item()
    max_axis = points[:,axis].max(dim=0)[0].item()
    mask = np.zeros(3)
    mask[axis] = 1.0
    if (axis == 0):
        position_cut = np.vstack(([np.zeros(xx.shape[0]), xx, yy]))
    elif (axis == 1):
        position_cut = np.vstack(([xx,np.zeros(xx.shape[0]), yy]))
    elif (axis == 2):
        position_cut = np.vstack(([xx, yy, np.zeros(xx.shape[0])]))
    position_cut = [position_cut + i*mask.reshape(-1, 1) for i in np.linspace(min_axis - 0.1, max_axis + 0.1, 50)]
    for index, pos in enumerate(position_cut):
        #fig = tools.make_subplots(rows=1, cols=1)

        field_input = utils.to_cuda(torch.tensor(pos.T, dtype=torch.float))
        z = []
        for i, pnts in enumerate(torch.split(field_input, 10000, dim=0)):
            if (not latent is None):
                pnts = torch.cat([latent.expand(pnts.shape[0], -1), pnts], dim=1)
            z.append(decoder(pnts).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        if (near_zero):
            if (np.min(z) < -1.0e-5):
                start = -0.1
            else:
                start = 0.0
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='axis {0} = {1}'.format(axis,pos[axis, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=False,
                                contours=dict(
                                     start=start,
                                     end=0.1,
                                     size=0.01
                                     )
                                # ),colorbar = {'dtick': 0.05}
                                )
        else:
            # trace1 = go.Contour(x=onedim_cut,
            #                     y=onedim_cut,
            #                     z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
            #                     name='axis {0} = {1}'.format(axis,pos[axis, 0]),  # colorbar=dict(len=0.4, y=0.8),
            #                     autocontour=True,
            #                     ncontours=70
            #                     # contours=dict(
            #                     #      start=-0.001,
            #                     #      end=0.001,
            #                     #      size=0.00001
            #                     #      )
            #                     # ),colorbar = {'dtick': 0.05}
            #                     )

            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='axis {0} = {1}'.format(axis,pos[axis, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=False,
                                # ncontours=70
                                contours=dict(
                                     start=-0.8,
                                     end=0.8,
                                     size=0.15
                                     )
                                )

        layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                               yaxis=dict(range=[-1, 1], autorange=False),
                                                               aspectratio=dict(x=1, y=1)),
                           title=dict(text='axis {0} = {1}'.format(axis,pos[axis, 0])))
        # fig['layout']['xaxis2'].update(range=[-1, 1])
        # fig['layout']['yaxis2'].update(range=[-1, 1], scaleanchor="x2", scaleratio=1)

        filename = '{0}{1}cutsaxis_{2}_{3}_{4}.html'.format(path,file_name_sep,axis, epoch, index)
        fig1 = go.Figure(data=[trace1], layout=layout)
        # offline.plot(fig1, filename=filename, auto_open=False)

        # fig1 = go.Figure(data=[trace1], layout=layout)
        fig1.write_image(filename.replace('.html', '.png'))


def plot_cuts(points,decoder,path,epoch,near_zero,latent=None, suffix = "all"):
    onedim_cut = np.linspace(-1, 1, 200)
    # onedim_cut = np.linspace(-1, 1, 500)

    # onedim_cut = np.linspace(-1, 1, 1000) #modified on 20201101
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    min_y = points[:,-2].min(dim=0)[0].item()
    max_y = points[:,-2].max(dim=0)[0].item()
    position_cut = np.vstack(([xx, np.zeros(xx.shape[0]), yy]))
    position_cut = [position_cut + np.array([0., i, 0.]).reshape(-1, 1) for i in np.linspace(min_y - 0.1, max_y + 0.1, 10)]
    for index, pos in enumerate(position_cut):
        #fig = tools.make_subplots(rows=1, cols=1)
        field_input = torch.tensor(pos.T, dtype=torch.float).cuda()
        z = []
        for i, pnts in enumerate(torch.split(field_input, 1000, dim=-1)):
            input_=pnts
            if (not latent is None):
                input_ = torch.cat([latent.expand(pnts.shape[0],-1) ,pnts],dim=1)
            z.append(decoder(input_).detach().cpu().numpy())
            # z.append(decoder(input_)[:,0].detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        if (near_zero):
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=False,
                                contours=dict(
                                     start=-0.001,
                                     end=0.001,
                                     size=0.00001
                                     )
                                # ),colorbar = {'dtick': 0.05}
                                )
        else:
            # trace1 = go.Contour(x=onedim_cut,
            #                     y=onedim_cut,
            #                     z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
            #                     name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
            #                     autocontour=True,
            #                     # contours=dict(
            #                     #      start=-0.001,
            #                     #      end=0.001,
            #                     #      size=0.00001
            #                     #      )
            #                     # ),colorbar = {'dtick': 0.05}
            #                     )
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=False,
                                contours=dict(
                                     start=-0.8,
                                     end=0.8,
                                     size=0.2
                                     )
                                # ),colorbar = {'dtick': 0.05}
                                )


        layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                               yaxis=dict(range=[-1, 1], autorange=False),
                                                               aspectratio=dict(x=1, y=1)),
                           title=dict(text='y = {0}'.format(pos[1, 0])))
        # fig['layout']['xaxis2'].update(range=[-1, 1])
        # fig['layout']['yaxis2'].update(range=[-1, 1], scaleanchor="x2", scaleratio=1)
        utils.mkdir_ifnotexists(path)
        filename = '{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '.html'
        fig1 = go.Figure(data=[trace1], layout=layout)
        fig1.write_image('{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '.png')
        #save html
        # offline.plot(fig1, filename=filename, auto_open=False)
    
        # save contour, resolution is too low
        # X, Y = np.meshgrid(onedim_cut, onedim_cut)
        # fig = plt.figure()
        # surf1 = plt.contourf(X, Y, z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]))
        # plt.contour(X, Y, z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]))
        # fig.colorbar(surf1)
        # plt.savefig('{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '.png', dpi = 100)

def plot_masks(points,decoder, n_branch, path,epoch,latent=None, suffix = "all"):
    # output type should be zero
    # onedim_cut = np.linspace(-1, 1, 200)
    onedim_cut = np.linspace(-1, 1, 500) #modified on 20201101
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    min_y = points[:,-2].min(dim=0)[0].item()
    max_y = points[:,-2].max(dim=0)[0].item()
    position_cut = np.vstack(([xx, np.zeros(xx.shape[0]), yy]))
    position_cut = [position_cut + np.array([0., i, 0.]).reshape(-1, 1) for i in np.linspace(min_y - 0.1, max_y + 0.1, 10)]
    
    branch_color = []
    if n_branch == 1:
        branch_color.append(cm.plasma(0.0))
    else:
        for i in range(n_branch):
            # branch_color.append(cm.hot(i / n_branch))
            branch_color.append(cm.plasma(i / (n_branch - 1)))

    branch_color = np.concatenate(branch_color, axis = 0)
    branch_color = branch_color.reshape(-1,4)

    for index, pos in enumerate(position_cut):
        #fig = tools.make_subplots(rows=1, cols=1)

        field_input = torch.tensor(pos.T, dtype=torch.float).cuda()
        img = []
        img_onehot = []
        z = []
        for i, pnts in enumerate(torch.split(field_input, 1000, dim=-1)):
            input_=pnts
            if (not latent is None):
                input_ = torch.cat([latent.expand(pnts.shape[0],-1) ,pnts],dim=1)
            mask_feature = decoder(input_).detach()[:, n_branch + 1: 2 * n_branch + 1]
            mask_feature_np = mask_feature.cpu().numpy()
            maxid = np.argmax(mask_feature_np, 1)
            mask_feature_onehot = np.zeros_like(mask_feature_np)
            mask_feature_onehot[np.arange(mask_feature_np.shape[0]), maxid] = 1.0

            z.append(mask_feature.argmax(dim=1).unsqueeze(1).cpu().numpy())
            img.append(np.matmul(mask_feature_np, branch_color))
            img_onehot.append(np.matmul(mask_feature_onehot, branch_color))
            # z.append(decoder(input_)[:,0].detach().cpu().numpy())
        z = np.concatenate(z, axis=0)
        img = np.concatenate(img, axis = 0).reshape(onedim_cut.shape[0],onedim_cut.shape[0],4)[::-1]
        img_onehot = np.concatenate(img_onehot, axis = 0).reshape(onedim_cut.shape[0],onedim_cut.shape[0],4)[::-1]
        # print ("img max: ", img.max())
        # img = img / img.max()
        img[img>1.0] = 1.0

        #go version
        # trace1 = go.Contour(x=onedim_cut,
        #                     y=onedim_cut,
        #                     z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
        #                     name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
        #                     autocontour=False,
        #                     contours=dict(
        #                          start=-0.5,
        #                          end=n_branch,
        #                          size=1
        #                          )
        #                     # ),colorbar = {'dtick': 0.05}
        #                     )

        # layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
        #                                                        yaxis=dict(range=[-1, 1], autorange=False),
        #                                                        aspectratio=dict(x=1, y=1)),
        #                    title=dict(text='y = {0}'.format(pos[1, 0])))
        # # fig['layout']['xaxis2'].update(range=[-1, 1])
        # # fig['layout']['yaxis2'].update(range=[-1, 1], scaleanchor="x2", scaleratio=1)
        # utils.mkdir_ifnotexists(path)
        # filename = '{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '.html'
        # fig1 = go.Figure(data=[trace1], layout=layout)
        # offline.plot(fig1, filename=filename, auto_open=False)
        utils.mkdir_ifnotexists(path)
        plt.imsave('{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '.png' ,img)
        plt.imsave('{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '_onehot.png' ,img_onehot)
        # figurename = '{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '.png'
        # fig1.write_image(figurename)
        # save contour, resolution is too low
        # X, Y = np.meshgrid(onedim_cut, onedim_cut)
        # fig = plt.figure()
        # surf1 = plt.contourf(X, Y, z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]))
        # plt.contour(X, Y, z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]))
        # fig.colorbar(surf1)
        # plt.savefig('{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '.png', dpi = 100)

    #save vtk file 
    res = 128
    mg = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res), np.linspace(-1, 1, res), indexing='ij')
    grid_pts = np.concatenate((mg[2].reshape(-1,1), mg[1].reshape(-1,1), mg[0].reshape(-1,1)), axis=1)
    # np.savetxt('grid_pts.xyz', grid_pts)
    # grid_predict = clf.predict(grid_pts)  
    grid_pts_tensor = torch.tensor(grid_pts, dtype=torch.float).cuda()
    grid_predict = []
    for pts in torch.split(grid_pts_tensor, 1000):
        mask_feature = decoder(pts).detach()[:, n_branch + 1: 2 * n_branch + 1]
        grid_predict.append(mask_feature.argmax(dim=1).unsqueeze(1).cpu().numpy())
    grid_predict = np.concatenate(grid_predict, axis=0)
    save_implicit_function_vtk(grid_predict.reshape(-1), '{0}/mask.vtk'.format(path))

def plot_masks_maxsdf(points,decoder, n_branch, path,epoch,latent=None, suffix = "all"):
    # output type should be zero
    # onedim_cut = np.linspace(-1, 1, 200)
    onedim_cut = np.linspace(-1, 1, 500) #modified on 20201101
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    min_y = points[:,-2].min(dim=0)[0].item()
    max_y = points[:,-2].max(dim=0)[0].item()
    position_cut = np.vstack(([xx, np.zeros(xx.shape[0]), yy]))
    position_cut = [position_cut + np.array([0., i, 0.]).reshape(-1, 1) for i in np.linspace(min_y - 0.1, max_y + 0.1, 10)]
    
    # branch_color = []
    # if n_branch == 1:
    #     branch_color.append(cm.plasma(0.0))
    # else:
    #     for i in range(n_branch):
    #         # branch_color.append(cm.hot(i / n_branch))
    #         branch_color.append(cm.plasma(i / (n_branch - 1)))
    # branch_color = np.concatenate(branch_color, axis = 0)
    # branch_color = branch_color.reshape(-1,4)
    #mask color
    mask_color = []
    for i in range(3):
        mask_color.append(cm.plasma(i / 2))
    mask_color = np.concatenate(mask_color, axis = 0)
    mask_color = mask_color.reshape(-1,4)

    for index, pos in enumerate(position_cut):
        #fig = tools.make_subplots(rows=1, cols=1)
        field_input = torch.tensor(pos.T, dtype=torch.float).cuda()
        imgs = []
        for i in range(n_branch):
            imgs.append([])
        # img_onehot = []
        # z = []
        for i, pnts in enumerate(torch.split(field_input, 1000, dim=-1)):
            input_=pnts
            if (not latent is None):
                input_ = torch.cat([latent.expand(pnts.shape[0],-1) ,pnts],dim=1)
            mask_feature = decoder(input_).detach()[:, n_branch + 1:]
            assert(mask_feature.shape[1] == 3 * n_branch)
            mask_feature_np = mask_feature.cpu().numpy()
            # maxid = np.argmax(mask_feature_np, 1)
            # mask_feature_onehot = np.zeros_like(mask_feature_np)
            # mask_feature_onehot[np.arange(mask_feature_np.shape[0]), maxid] = 1.0

            # z.append(mask_feature.argmax(dim=1).unsqueeze(1).cpu().numpy())
            # img.append(np.matmul(mask_feature_np, branch_color))
            for j in range(n_branch):
                imgs[j].append(np.matmul(mask_feature_np[:, 3 * j: 3 * (j + 1)], mask_color))
            # img_onehot.append(np.matmul(mask_feature_onehot, branch_color))
            # z.append(decoder(input_)[:,0].detach().cpu().numpy())
        # z = np.concatenate(z, axis=0)
        for i in range(n_branch):
            imgs[i] = np.concatenate(imgs[i], axis = 0).reshape(onedim_cut.shape[0],onedim_cut.shape[0],4)[::-1]
            imgs[i][imgs[i]>1.0] = 1.0
            utils.mkdir_ifnotexists(path)
            plt.imsave('{0}/cuts{1}_{2}_b{3}'.format(path, epoch, index, i) + suffix + '.png' ,imgs[i])
        # img = np.concatenate(img, axis = 0).reshape(onedim_cut.shape[0],onedim_cut.shape[0],4)[::-1]
        # img_onehot = np.concatenate(img_onehot, axis = 0).reshape(onedim_cut.shape[0],onedim_cut.shape[0],4)[::-1]
        # print ("img max: ", img.max())
        # img = img / img.max()
        # img[img>1.0] = 1.0
        # utils.mkdir_ifnotexists(path)
        # plt.imsave('{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '.png' ,img)
        # plt.imsave('{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '_onehot.png' ,img_onehot)
        # figurename = '{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '.png'
        # fig1.write_image(figurename)
        # save contour, resolution is too low
        # X, Y = np.meshgrid(onedim_cut, onedim_cut)
        # fig = plt.figure()
        # surf1 = plt.contourf(X, Y, z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]))
        # plt.contour(X, Y, z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]))
        # fig.colorbar(surf1)
        # plt.savefig('{0}/cuts{1}_{2}'.format(path, epoch, index) + suffix + '.png', dpi = 100)

def get_grid(points,resolution):
    eps = 0.1
    input_min = torch.min(points, dim=0)[0].squeeze().cpu().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().cpu().numpy()
    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points":grid_points,
            "shortest_axis_length":length,
            "xyz":[x,y,z],
            "shortest_axis_index":shortest_axis}


def get_grid_uniform(resolution):
    x = np.linspace(-1.2,1.2, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = utils.to_cuda(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float))

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.4,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}