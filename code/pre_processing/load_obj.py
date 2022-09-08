import os
import numpy as np
import struct

class mytrimesh:
  def __init__(self, verts, faces):
    self.vertices = verts
    self.faces = faces

def load_obj_simple(filename):
  vertex_list = []
  face_list = []
  #triangle mesh
  with open(filename,"r") as rf:
    for line in rf:
      if(line[0] == 'v' and line[1] == ' '):
        #vertex
        arr = line.split()
        vertex_list.append([float(arr[1]), float(arr[2]), float(arr[3])])
      elif(line[0] == 'f' and line[1] == ' '):
        #face
        arr = line.split()
        face_list.append([int(arr[1].split("/")[0]), int(arr[2].split("/")[0]), int(arr[3].split("/")[0])])
  return mytrimesh(np.array(vertex_list, dtype=np.float32), np.array(face_list, dtype=np.int32)-1)

class DisjointSet:
    def __init__(self, n):
        self.fa = list(range(n))
        self.rank = [1] * n
        self.lands = n
 
    def find(self, x):
        if x == self.fa[x]:
            return x
        else:
            self.fa[x] = self.find(self.fa[x])
            return self.fa[x]
 
    def union(self, i, j):
        x = self.find(i)
        y = self.find(j)
        if x == y: return
        if self.rank[x] <= self.rank[y]:
            self.fa[x] = y
        else:
            self.fa[y] = x
        
        if self.rank[x] == self.rank[y] and x != y:
            self.rank[y] += 1
        self.lands -= 1

def color_connected_mesh(mesh):
  nvert = len(mesh.vertices)
  res = [0] * nvert
  tree = DisjointSet(nvert)
  faces = mesh.faces
  nface = len(faces)
  for i in range(nface):
    tree.union(faces[i][0], faces[i][1])
    tree.union(faces[i][2], faces[i][1])
    tree.union(faces[i][0], faces[i][2])

  if tree.lands != 1:
    used = {}
    nvid = 0
    for i in range(nvert):
      oid = tree.find(i)
      if oid in used:
        res[i] = used[oid]
      else:
        used[oid] = nvid
        res[i] = nvid
        nvid += 1

  return res


def write_obj_simple(filename, vertices, faces):
  with open(filename,"w") as wf:
    for pos in vertices:
      wf.write("v {} {} {}\n".format(pos[0], pos[1], pos[2]))
    for face in faces:
      wf.write("f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1))
      
def normalize_mesh(input_filename, output_filename):
  verts, faces = load_obj_simple(input_filename)
  verts = normalize_model(verts)
  write_obj_simple(output_filename, verts, faces)
  