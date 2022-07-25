#extend mask
import numpy as np
from scipy.spatial import cKDTree as KDTree

input_path='/mnt/sdf1/haog/code/IGR/CADdata/fandisk_40k_cot_rev_nocolor.xyz'
input_mask_path='/mnt/sdf1/haog/code/IGR/CADdata/fandisk_mask_40k_cot_rev_nocolor.txt'

max_dist=0.05 #maximum moving distance for each point
depth = 5 #maximum depth for searching	


pts = np.loadtxt(input_path)
X = pts[:,:3]
y = np.loadtxt(input_mask_path)
normals = pts[:,3:]
tree = KDTree(X)

pos_move = 2 * max_dist * np.ones(X.shape[0])
neg_move = -2 * max_dist * np.ones(X.shape[0])

for i in range(X.shape[0]):
	j = 0
	while j < depth:
		pos_move[i] = pos_move[i] / 2
		new_pt = X[i] + normals[i] * pos_move[i]
		_, pid = tree.query(new_pt)
		if pid == i:
			break
		j = j + 1
	if j==depth:
		pos_move[i] = 0
	j = 0
	while j < depth:
		neg_move[i] = neg_move[i] / 2
		new_pt = X[i] + normals[i] * neg_move[i]
		_, pid = tree.query(new_pt)
		if pid == i:
			break
		j = j + 1
	if j==depth:
		neg_move[i] = 0

new_pos_pts = np.concatenate((X + np.expand_dims(pos_move, axis=1)*normals, normals), axis=1)
new_neg_pts = np.concatenate((X + np.expand_dims(neg_move, axis=1)*normals, normals), axis=1)

np.savetxt('fandisk_pos.xyz', new_pos_pts)
np.savetxt('fandisk_neg.xyz', new_neg_pts)


extend_depth = 5
extend_pts = new_neg_pts
extend_mask = y

for i in range(extend_depth):
	coeff_neg = (4.0 - i)/5.0
	coeff_pos = (i + 1.0)/5.0
	extend_pts = np.concatenate((extend_pts, coeff_neg * new_neg_pts + coeff_pos * new_pos_pts), axis=0)
	extend_mask = np.concatenate((extend_mask, y), axis=0)

np.savetxt('fandisk_extend.xyz', extend_pts)
np.savetxt('fandisk_extend_mask.xyz', extend_mask)