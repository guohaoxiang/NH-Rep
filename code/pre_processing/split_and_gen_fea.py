import os
import numpy as np
from load_obj import *
import yaml
import platform
sys_info = platform.system()

in_path = 'abc_data'
out_path = 'raw_input'

def split_single_sample(mesh_file):
  overwrite_files = True
  def normalize_mesh_padding(mesh, padding=0.05):
    coord_max = mesh.vertices.max(axis=0)
    coord_min = mesh.vertices.min(axis=0)
    center = (coord_max + coord_min) / 2.0
    bbox_length = (coord_max - coord_min).max()
    mesh.vertices = (mesh.vertices - center) * (1-padding) / bbox_length
    return mesh
  
  #sample points on each curve and save curve informations
  mesh_fileall = mesh_file
  mesh = load_obj_simple(mesh_file)
  vert2comp = color_connected_mesh(mesh)
  face2comp = [0] * len(mesh.faces)
  print('filename {} comp: {}'.format(mesh_file, max(vert2comp) + 1))
  nface = len(mesh.faces)
  for i in range(nface):
    face2comp[i] = vert2comp[mesh.faces[i][0]]
  ncomp = max(vert2comp) + 1
  comp2faces = [[] for i in range(ncomp)]
  for i in range(nface):
    comp2faces[face2comp[i]].append(i)
  perpatch_faceo2n = [{} for i in range(ncomp)]

  # hangning vert??
  for i in range(ncomp):
    if not len(comp2faces[i]) > 0:
      print('!!!! component with no face exists !!!!')
    assert(len(comp2faces[i]) > 0)
    for j in range(len(comp2faces[i])):
      perpatch_faceo2n[i][comp2faces[i][j]] = j



  yaml_file = mesh_file.replace(".obj", ".yml")
  assert(os.path.exists(yaml_file))
  with open(yaml_file, "r") as f: feature_dict_all = yaml.safe_load(f)
  
  curves_all = []
  patches_all = []
  if 'curves' in feature_dict_all:
    curves_all = feature_dict_all['curves']
  patches_all = feature_dict_all['surfaces']

  #classifiy patches and curves
  comp2curves = [[] for i in range(ncomp)]
  comp2patches = [[] for i in range(ncomp)]
  for i in range(len(curves_all)):
    cur_comp = vert2comp[curves_all[i]['vert_indices'][0]]
    comp2curves[cur_comp].append(curves_all[i])

  for i in range(len(patches_all)):
    cur_comp = face2comp[patches_all[i]['face_indices'][0]]
    comp2patches[cur_comp].append(patches_all[i])

  #update face id of comp2patches, remove vert id
  for i in range(ncomp):
    for j in range(len(comp2patches[i])):
      comp2patches[i][j].pop('vert_indices', None)
      for k in range(len(comp2patches[i][j]['face_indices'])):
        comp2patches[i][j]['face_indices'][k] = perpatch_faceo2n[i][comp2patches[i][j]['face_indices'][k]]

  for compid in range(ncomp):
    print('compid: ', compid)
    mesh_file = mesh_fileall.replace('.obj', '_{}.obj'.format(compid))
    curves = comp2curves[compid]
    patches = comp2patches[compid]
    feature_dict = {'curves': curves, 'surfaces': patches}
    #update vert id
    vert_valid_id = np.unique(mesh.faces[comp2faces[compid]].reshape(-1))
    # print('unique vert size:', vert_valid_id.shape[0])
    vert_o2n = -1 * np.ones(mesh.vertices.shape[0], dtype = np.int64)
    vert_o2n[vert_valid_id] = np.arange(vert_valid_id.shape[0])

    cur_faces = vert_o2n[mesh.faces[comp2faces[compid]].reshape(-1)].reshape(-1,3)
    cur_verts = mesh.vertices[vert_valid_id]
    write_obj_simple(os.path.join(out_path, os.path.basename(mesh_file)), cur_verts, cur_faces)
    output_yaml = os.path.join(out_path, os.path.basename(mesh_file.replace(".obj", ".yml")))

    #update face id of patches
    flag_cylinder_cube = False
    with open(output_yaml, 'w') as yaml_file:
      yaml.dump(feature_dict, yaml_file, default_flow_style=True)

    if len(feature_dict['surfaces']) == 3 or len(feature_dict['surfaces']) == 6:
      count_cylinder = 0
      count_plane = 0
      for i in range(len(feature_dict['surfaces'])):
        if feature_dict['surfaces'][i]['type'] == 'Cylinder':
          count_cylinder += 1
        if feature_dict['surfaces'][i]['type'] == 'Plane':
          count_plane += 1

      if count_cylinder == 1 and count_plane == 2:
        flag_cylinder_cube = True

      if count_cylinder == 0  and count_plane == 6:
        flag_cylinder_cube = True

    if flag_cylinder_cube:
      tmpfile = os.path.join(out_path, os.path.basename(mesh_file.replace(".obj", ".cylindercube")))

      f = open(tmpfile, 'w')
      f.close()

def sample_points_on_single_curve(points, sample_num):
  #calculate curve length
  curve_length = 0
  cum_line_length = []
  for i in range(points.shape[0]-1):
    length = np.sqrt((np.square(points[i+1] - points[i])).sum())
    curve_length += length
    cum_line_length.append(length)
  
  cum_line_length = np.cumsum(np.array(cum_line_length))
  segment_curve_length = curve_length / (sample_num - 1)
  sampled_points = np.zeros([sample_num, 3])
  sampled_points[0] = points[0]
  
  cur_line_id = 0
  for i in range(1, sample_num-1):
    target_length = i*segment_curve_length
    while(target_length > cum_line_length[cur_line_id]): cur_line_id+=1
    start_points = points[cur_line_id]
    end_points = points[cur_line_id+1]
    cur_line_length = np.sqrt((np.square(end_points - start_points)).sum())
    proportion = (cum_line_length[cur_line_id] - target_length) / cur_line_length
    if(proportion < -1e-5 or proportion > 1.00001):
      print("Assertion Violation: proportion out of range: {}".format(proportion))
    if(proportion < 0):
      proportion = 0
    elif(proportion > 1):
      proportion = 1
    sampled_points[i] = proportion*start_points + (1-proportion)*end_points
  sampled_points[sample_num-1] = points[-1]
  return sampled_points


def sample_points_on_each_curve(mesh, curves, sample_num=100):  
  curve_points = []
  for curve_id in range(len(curves)):
    curve_points.append(sample_points_on_single_curve(mesh.vertices[curves[curve_id]['vert_indices']], sample_num))
  if len(curve_points) > 0:
    return np.concatenate(curve_points, axis = 0)
  else:
    return np.array([])

def gen_fea_file(mesh_file):
  overwrite_files = True
  def normalize_mesh_padding(mesh, padding=0.05):
    coord_max = mesh.vertices.max(axis=0)
    coord_min = mesh.vertices.min(axis=0)
    center = (coord_max + coord_min) / 2.0
    bbox_length = (coord_max - coord_min).max()
    mesh.vertices = (mesh.vertices - center) * (1-padding) / bbox_length
    return mesh
  
  #sample points on each curve and save curve informations
  mesh_fileall = mesh_file
  mesh = load_obj_simple(mesh_file)
  vert2comp = color_connected_mesh(mesh)
  face2comp = [0] * len(mesh.faces)
  print('filename {} comp: {}'.format(mesh_file, max(vert2comp) + 1))
  nface = len(mesh.faces)
  for i in range(nface):
    face2comp[i] = vert2comp[mesh.faces[i][0]]
  ncomp = max(vert2comp) + 1
  comp2faces = [[] for i in range(ncomp)]
  for i in range(nface):
    comp2faces[face2comp[i]].append(i)
  perpatch_faceo2n = [{} for i in range(ncomp)]

  # hanging vert
  for i in range(ncomp):
    if not len(comp2faces[i]) > 0:
      print('!!!! component with no face exists !!!!')
    assert(len(comp2faces[i]) > 0)
    for j in range(len(comp2faces[i])):
      perpatch_faceo2n[i][comp2faces[i][j]] = j


  yaml_file = mesh_file.replace(".obj", ".yml")
  assert(os.path.exists(yaml_file))
  with open(yaml_file, "r") as f: feature_dict_all = yaml.safe_load(f)
  
  curves_all = []
  patches_all = []
  if 'curves' in feature_dict_all:
    curves_all = feature_dict_all['curves']
  patches_all = feature_dict_all['surfaces']

  #classifiy patches and curves
  comp2curves = [[] for i in range(ncomp)]
  comp2patches = [[] for i in range(ncomp)]
  for i in range(len(curves_all)):
    cur_comp = vert2comp[curves_all[i]['vert_indices'][0]]
    comp2curves[cur_comp].append(curves_all[i])

  for i in range(len(patches_all)):
    cur_comp = face2comp[patches_all[i]['face_indices'][0]]
    comp2patches[cur_comp].append(patches_all[i])

  #update face id of comp2patches, remove vert id
  for i in range(ncomp):
    for j in range(len(comp2patches[i])):
      comp2patches[i][j].pop('vert_indices', None)
      for k in range(len(comp2patches[i][j]['face_indices'])):
        comp2patches[i][j]['face_indices'][k] = perpatch_faceo2n[i][comp2patches[i][j]['face_indices'][k]]
  
  for compid in range(ncomp):
    mesh_file = mesh_fileall.replace('.obj', '_{}.obj'.format(compid))
    curves = comp2curves[compid]
    patches = comp2patches[compid]
    feature_dict = {'curves': curves, 'surfaces': patches}
    
    curve_sample_points_file = os.path.join(out_path, os.path.basename(mesh_file.replace(".obj", "_curve.sample.xyz")))
    if True:
      curve_sample_points = np.reshape(sample_points_on_each_curve(mesh, curves), [-1,3])
      np.savetxt(curve_sample_points_file, curve_sample_points)
    
    #update vert id
    vert_valid_id = np.unique(mesh.faces[comp2faces[compid]].reshape(-1))
    vert_o2n = -1 * np.ones(mesh.vertices.shape[0], dtype = np.int64)
    vert_o2n[vert_valid_id] = np.arange(vert_valid_id.shape[0])
    cur_faces = vert_o2n[mesh.faces[comp2faces[compid]].reshape(-1)].reshape(-1,3)
    cur_verts = mesh.vertices[vert_valid_id]
    write_obj_simple(os.path.join(out_path, os.path.basename(mesh_file)), cur_verts, cur_faces)
    
    #save feature file
    feature_list = []
    flag_only_sharp = False
    for curve_idx in range(len(curves)):
      if('closed' in curves[curve_idx]):
        assert(curves[curve_idx]['type'] == 'BSpline')
        closed_curve = curves[curve_idx]['closed']
      else:
        assert(curves[curve_idx]['type'] != 'BSpline')
        curve_vert_idx = curves[curve_idx]['vert_indices']
        if(curve_vert_idx[0] == curve_vert_idx[-1]):
          closed_curve = True
        else:
          closed_curve = False
      if flag_only_sharp and not curves[curve_idx]['sharp']:
        continue

      if True:
        for vid in range(len(curves[curve_idx]['vert_indices']) - 1):
          feature_list.append(vert_o2n[curves[curve_idx]['vert_indices'][vid]])
          feature_list.append(vert_o2n[curves[curve_idx]['vert_indices'][vid + 1]])
        if closed_curve:
          feature_list.append(vert_o2n[curves[curve_idx]['vert_indices'][-1]])
          feature_list.append(vert_o2n[curves[curve_idx]['vert_indices'][0]])

    feature_file = os.path.join(out_path, os.path.basename(mesh_file.replace(".obj", ".fea")))

    f = open(feature_file, 'w')
    f.write('{}\n'.format(len(feature_list)//2))
    for i in range(len(feature_list)//2):
      f.write('{} {}\n'.format(feature_list[2 * i], feature_list[2 * i + 1]))
    f.close()


def split_all_models():

  if not os.path.exists(out_path):
      os.mkdir(out_path)

  task_list = []
  #source folder set below
  file_list = os.listdir(in_path)
  for file in file_list:
    if file.endswith(".obj"):
      task_list.append(os.path.join(in_path, file))

  for t in task_list:
      split_single_sample(t)
      gen_fea_file(t)

def main():
  split_all_models()

if __name__ == '__main__':
  print('Program running...')
  main()