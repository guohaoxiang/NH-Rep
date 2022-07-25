from sklearn import svm
import numpy as np
from numpy import linalg as LA
from plywrite import save_vert_color_ply, save_vertnormal_color_ply
from save_implicit_function_vtk import save_implicit_function_vtk, save_implicit_function_color_vtk
from matplotlib import cm

# input_path='/mnt/sdf1/haog/code/IGR/CADdata/fandisk_40k_cot_rev_nocolor.xyz'
# input_mask_path='/mnt/sdf1/haog/code/IGR/CADdata/fandisk_mask_40k_cot_rev_nocolor.txt'

# input_path='/mnt/sdf1/haog/code/IGR/CADdata/fandisk_40k_cot_nocolor.xyz'
# input_mask_path='/mnt/sdf1/haog/code/IGR/CADdata/fandisk_mask_40k_cot_nocolor.txt'

input_path='fandisk_extend.xyz'
input_mask_path='fandisk_extend_mask.txt'

test_path='/mnt/sdf1/haog/code/IGR/CADdata/fandisk_40k_cot_rev_nocolor_test.xyz'
test_mask_path='/mnt/sdf1/haog/code/IGR/CADdata/fandisk_mask_40k_cot_rev_nocolor_test.txt'

pts = np.loadtxt(input_path)
X = pts[:,:3]
y = np.loadtxt(input_mask_path)
pts_test = np.loadtxt(test_path)
X_test = pts_test[:,:3]
y_test = np.loadtxt(test_mask_path)

clf=svm.SVC(kernel='poly', degree=6, coef0=10)
clf.fit(X,y)
n_branch = int(np.max(y))
input_predict = clf.predict(X)
verts_bgt = np.zeros([X.shape[0], n_branch])
verts_bgt[np.arange(X.shape[0]), y.astype(int) - 1] = 1.0
verts_pred = np.zeros([X.shape[0], n_branch])
verts_pred[np.arange(X.shape[0]), input_predict.astype(int) - 1] = 1.0
verts_error = np.expand_dims(LA.norm((verts_pred - verts_bgt), axis = 1),1)
diff = np.max(verts_error) - np.min(verts_error)
print ("sum: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error), np.max(verts_error), np.min(verts_error)))
verts_error = (verts_error - np.min(verts_error)) / diff
verts_error_color = np.matmul(verts_error, np.array([[255,255,255]])).astype(int)

test_predict = clf.predict(X_test)
verts_pred_test = np.zeros([X_test.shape[0], n_branch])
verts_pred_test[np.arange(X_test.shape[0]), test_predict.astype(int) - 1] = 1.0
verts_bgt_test = np.zeros([X_test.shape[0], n_branch])
verts_bgt_test[np.arange(X_test.shape[0]), y_test.astype(int) - 1] = 1.0
verts_error_test = np.expand_dims(LA.norm((verts_pred_test - verts_bgt_test), axis = 1),1)
diff_test = np.max(verts_error_test) - np.min(verts_error_test)
print ("sum: avg/max error: {:.6f}/{:.6f} min error: {:.6f} ".format(np.average(verts_error_test), np.max(verts_error_test), np.min(verts_error_test)))
verts_error_test = (verts_error_test - np.min(verts_error_test)) / diff_test
verts_error_color_test = np.matmul(verts_error_test, np.array([[255,255,25*-5]])).astype(int)

# save_vertnormal_color_ply(pts, verts_error_color, "fandisk_mask_error_poly_d6r10ext_train.ply")
# save_vertnormal_color_ply(pts_test, verts_error_color_test, "fandisk_mask_error_poly_d6r10ext_test.ply")

save_vertnormal_color_ply(pts, verts_error_color, "fandisk_mask_error_poly_d6r10cotrevextend_train.ply")
save_vertnormal_color_ply(pts_test, verts_error_color_test, "fandisk_mask_error_poly_d6r10cotrevextend_test.ply")


res = 128
mg = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res), np.linspace(-1, 1, res), indexing='ij')
grid_pts = np.concatenate((mg[2].reshape(-1,1), mg[1].reshape(-1,1), mg[0].reshape(-1,1)), axis=1)
# np.savetxt('grid_pts.xyz', grid_pts)
grid_predict = clf.predict(grid_pts)

save_implicit_function_vtk(grid_predict.reshape(-1), "fandisk_implicit_d6r10revextend.vtk")

# colormap = np.zeros([n_branch, 4])
# for i in range(n_branch):
# 	colormap[i] = np.array(cm.plasma(i * 1.0/(n_branch - 1)))

# grid_color = np.zeros([res * res * res, 4])

# for i in range(res * res * res):
# 	grid_color[i] = colormap[int(grid_predict[i]) -1]
# # grid_color[:,0] = (grid_predict.reshape(-1) - np.min(grid_predict))/(np.max(grid_predict) - np.min(grid_predict))
# save_implicit_function_color_vtk(grid_predict, grid_color, "fandisk_implicit_color.vtk")
