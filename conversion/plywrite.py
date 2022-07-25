from plyfile import PlyData, PlyElement
import numpy as np

def tet_ply(text, byte_order):
    vertex = np.array([(0, 0, 0),
                          (0, 1, 1),
                          (1, 0, 1),
                          (1, 1, 0)],
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    face = np.array([([0, 1, 2],    255, 255, 255),
                        ([0, 2, 3], 255,   0,   0),
                        ([0, 1, 3],   0, 255,   0),
                        ([1, 2, 3],   0,   0, 255)],
                       dtype=[('vertex_indices', 'i4', (3,)),
                              ('red', 'u1'), ('green', 'u1'),
                              ('blue', 'u1')])
    print(vertex)
    print(face)

    return PlyData(
        [
            PlyElement.describe(
                vertex, 'vertex',
                comments=['tetrahedron vertices']
            ),
            PlyElement.describe(face, 'face')
        ],
        text=text, byte_order=byte_order,
        comments=['single tetrahedron with colored faces']
    )

def save_vert_color_ply(vert_pos, vert_color, filename):
	assert(vert_pos.shape[0] == vert_color.shape[0])
	vertex = np.array([(vert_pos[i][0], vert_pos[i][1], vert_pos[i][2], vert_color[i][0], vert_color[i][1], vert_color[i][2]) for i in range(vert_pos.shape[0])], dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                              ('blue', 'u1')])
	plydata = PlyData(
        [
            PlyElement.describe(
                vertex, 'vertex',
                comments=['tetrahedron vertices']
            )
        ],
        text=True, byte_order='=',
    )
	plydata.write(filename)
	return;

def save_vertnormal_color_ply(vert_posnormal, vert_color, filename):
  assert(vert_posnormal.shape[0] == vert_color.shape[0])
  vertex = np.array([(vert_posnormal[i][0], vert_posnormal[i][1], vert_posnormal[i][2], vert_posnormal[i][3], vert_posnormal[i][4], vert_posnormal[i][5], vert_color[i][0], vert_color[i][1], vert_color[i][2]) for i in range(vert_posnormal.shape[0])], dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'),
                              ('blue', 'u1')])
  plydata = PlyData(
        [
            PlyElement.describe(
                vertex, 'vertex',
                comments=['tetrahedron vertices']
            )
        ],
        text=True, byte_order='=',
    )
  plydata.write(filename)
  return;


# plydata = tet_ply(True, '=')

# print(plydata)
# plydata.write("ply.ply")

# vert_pos = np.array([[0, 0, 0],
#                      [0, 1, 1],
#                      [1, 0, 1],
#                      [1, 1, 0]])

# vert_posnormal = np.concatenate((vert_pos, vert_pos), axis = 1)

# vert_color = np.array([[255, 255, 255],
# 					   [255,   0,   0],
# 					   [  0, 255,   0],
# 					   [  0,   0, 255]])

# # save_vert_color_ply(vert_pos, vert_color, "vert.ply")
# save_vertnormal_color_ply(vert_posnormal, vert_color, "vert.ply")
