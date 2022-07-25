import numpy as np
import os


def save_implicit_function_vtk(pt_value, filename):
	res = round(pow(pt_value.shape[0], 1/3))
	assert(pt_value.shape[0] == res * res * res)
	f = open(filename, 'w')
	f.write("# vtk DataFile Version 3.0\nImplicit function\nASCII\nDATASET RECTILINEAR_GRID\n")
	f.write("DIMENSIONS {} {} {}\n".format(res, res, res))
	f.write("X_COORDINATES {} float\n".format(res))
	for i in range(res-1):
		f.write('{:.6f} '.format(i * 2.0/(res-1)-1))
	f.write('{:.6f}\n'.format(1.0))

	f.write("Y_COORDINATES {} float\n".format(res))
	for i in range(res-1):
		f.write('{:.6f} '.format(i * 2.0/(res-1)-1))
	f.write('{:.6f}\n'.format(1.0))

	f.write("Z_COORDINATES {} float\n".format(res))
	for i in range(res-1):
		f.write('{:.6f} '.format(i * 2.0/(res-1)-1))
	f.write('{:.6f}\n'.format(1.0))

	f.write('POINT_DATA {}\n'.format(pt_value.shape[0]))
	f.write('SCALARS funcvalue float 1\n')
	f.write('LOOKUP_TABLE default\n')
	for i in range(pt_value.shape[0]):
		f.write('{:.6f} '.format(pt_value[i]))
	f.write('\n')

	f.close()

def save_implicit_function_color_vtk(pt_value, pt_color, filename):
	res = round(pow(pt_value.shape[0], 1/3))
	assert(pt_value.shape[0] == res * res * res)
	assert(pt_value.shape[0] == pt_color.shape[0])
	max_value = np.max(pt_value)

	f = open(filename, 'w')
	f.write("# vtk DataFile Version 3.0\nImplicit function\nASCII\nDATASET RECTILINEAR_GRID\n")
	f.write("DIMENSIONS {} {} {}\n".format(res, res, res))
	f.write("X_COORDINATES {} float\n".format(res))
	for i in range(res-1):
		f.write('{:.6f} '.format(i * 2.0/(res-1)-1))
	f.write('{:.6f}\n'.format(1.0))

	f.write("Y_COORDINATES {} float\n".format(res))
	for i in range(res-1):
		f.write('{:.6f} '.format(i * 2.0/(res-1)-1))
	f.write('{:.6f}\n'.format(1.0))

	f.write("Z_COORDINATES {} float\n".format(res))
	for i in range(res-1):
		f.write('{:.6f} '.format(i * 2.0/(res-1)-1))
	f.write('{:.6f}\n'.format(1.0))

	f.write('POINT_DATA {}\n'.format(pt_value.shape[0]))
	f.write('SCALARS funcvalue float 1\n')
	f.write('LOOKUP_TABLE my_table\n')
	for i in range(pt_value.shape[0]):
		f.write('{:.6f}\n'.format(pt_value[i]))
	f.write('LOOKUP_TABLE my_table {}\n'.format(pt_value.shape[0]))
	for i in range(pt_color.shape[0]):
		f.write('{:.2f} {:.2f} {:.2f} {:.2f}\n'.format(pt_color[i][0], pt_color[i][1], pt_color[i][2], pt_color[i][3]))
	f.write('\n')

	f.close()

if __name__ == '__main__':
	pt_value = np.ones(32 * 32 * 32)
	save_implicit_function_vtk(pt_value, "save_im_test.vtk")