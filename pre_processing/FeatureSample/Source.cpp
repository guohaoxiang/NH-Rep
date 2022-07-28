#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include "cxxopts.hpp"
#include "Mesh3D.h"
#include "helper.h"
#include "happly.h"
#include "Tree.h"

using namespace MeshLib;

#define MIN_PATCH_AREA 1e-4


std::string GetFileExtension(const std::string& FileName)
{
	if (FileName.find_last_of(".") != std::string::npos)
		return FileName.substr(FileName.find_last_of(".") + 1);
	return "";
}

bool load_feature_file(const char* filename, std::vector<std::pair<int, int>>& ungrouped_feature)
{
	ungrouped_feature.clear();
	std::ifstream ifs(filename);
	if (!ifs.is_open())
	{
		std::cout << "Cannot Open Input Feature File" << std::endl;
		return false;
	}

	//fea file: first part
	int pair_size = 0;
	ifs >> pair_size;
	std::pair<int, int> tmp_pair(-1, -1);
	for (size_t i = 0; i < pair_size; i++)
	{
		ifs >> tmp_pair.first >> tmp_pair.second;
		ungrouped_feature.push_back(tmp_pair);
	}

	return true;
}

void save_feature_file(const char* filename, const std::vector<std::pair<int, int>>& ungrouped_feature)
{
	std::ofstream ofs(filename);
	ofs << ungrouped_feature.size() << std::endl;
	for (size_t i = 0; i < ungrouped_feature.size(); i++)
	{
		ofs << ungrouped_feature[i].first << " " << ungrouped_feature[i].second << std::endl;
	}

	ofs.close();
}

void save_conf_file(const char* filename, const std::string str, bool flag_convex = true)
{
	std::ofstream ofs(filename);
	ofs << "csg{\n    list = ";
	ofs << str << std::endl;
	ofs << "    flag_convex = " << int(flag_convex) << "," << std::endl;
	ofs << "}";
	
	ofs.close();
}

int check_mesh_edge_convex(Mesh3d* m, HE_edge<double>* he)
{
	//return 0: smooth, 1: convex, 2: concave
	int hetri_vertidsum = 0, hepairtri_vertidsum = 0;

	if (m->is_on_boundary(he))
	{
		return 0;
	}
	
	HE_edge<double>* begin_edge = he;
	HE_edge<double>* edge = he;
	do
	{
		hetri_vertidsum += edge->vert->id;
		edge = edge->next;
	} while (edge != begin_edge);

	begin_edge = he->pair;
	edge = he->pair;
	do
	{
		hepairtri_vertidsum += edge->vert->id;
		edge = edge->next;
	} while (edge != begin_edge);

	int hetri_otherid = hetri_vertidsum - he->vert->id - he->pair->vert->id;
	int hepairtri_otherid = hepairtri_vertidsum - he->vert->id - he->pair->vert->id;
	double product = he->face->normal.Dot(m->get_vertices_list()->at(hetri_otherid)->pos - m->get_vertices_list()->at(hepairtri_otherid)->pos);
	double face_cos_value = he->face->normal.Dot(he->pair->face->normal);
	int res = 0;
	if (product > 0.0)
	{
		if (face_cos_value < th_smooth_cos_value)
			res = 1;
		//return true;
	}
	else
	{
		if (face_cos_value < th_smooth_cos_value)
		res = 2;
		//return false;
	}
	return res;
}


int main(int argc, char** argv)
{
	try
	{
		cxxopts::Options options("FeaturedModelPointSample", "Point Sampling program for featured CAD models (author: Haoxiang Guo, Email: guohaoxiangxiang@gmail.com)");
		options
			.positional_help("[optional args]")
			.show_positional_help()
			.allow_unrecognised_options()
			.add_options()
			("i,input", "input mesh (obj/off format)", cxxopts::value<std::string>())
			("f,feature", "input feature file (fea format)", cxxopts::value<std::string>())
			("p,pointcloud", "input pointcloud", cxxopts::value<std::string>())
			("pf", "face of input pointcloud", cxxopts::value<std::string>())
			("o,output", "output mesh/points (obj/off/points/xyz format)", cxxopts::value<std::string>())
			("k,mask", "output mask file (txt format)", cxxopts::value<std::string>())
			("fs", "number of samples on feature edges(default: 10000)", cxxopts::value<int>())
			("ns", "number of samples on non-feature faces(default: 40000)", cxxopts::value<int>())
			("m,mode", "processing mode: 0 for normalization and 1 for feature sample", cxxopts::value<int>())
			("c,color", "whether coloring is used, 0: not used, 1: used, default: 0", cxxopts::value<int>())
			("mp", "maximum number of patches in each colored cluster, only work for csg, default -1(no upper bound)", cxxopts::value<int>())
			("cot", "whether cotangent weight is used for sampling, 0: not used, 1: used, default: 0", cxxopts::value<int>())
			("s,sigma", "sigma for noisy points position, default 0.0", cxxopts::value<double>())
			("sn", "sigma for noisy points normal in degrees, default 0.0", cxxopts::value<double>())
			("csg", "whether generating csg tree for model, default: 0", cxxopts::value<int>())
			("convex", "whether the first layer is convex, default: 0", cxxopts::value<int>())
			("r,repairturn", "whether the turn vertex are repaired, default: 1", cxxopts::value<int>())
			("verbose", "verbose setting, default: 0", cxxopts::value<int>())
			("strict", "treat all edges as either strictly convex or concave")
			("repairtree", "repair tree feature")
			("h,help", "print help");
		
		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help({ "", "Group" }) << std::endl;
			exit(0);
		}
		int n_nonfeature_sample = 50000;
		int n_feature_sample = 0;
		int min_sample_perpatch = 50;
		double sigma = -1.0;
		double sigma_n = -1.0;
		assert(result.count("m"));
		int processing_mode = result["m"].as<int>();
		assert(result.count("i") && result.count("o"));
		auto& inputfile = result["i"].as<std::string>();
		auto& outputfile = result["o"].as<std::string>();
		//output pts by colors
		int last_dot = (int)outputfile.find_last_of(".");
		auto output_prefix = outputfile.substr(0, last_dot);
		int flag_csg = 0;

		bool flag_sample_pts = true; //simply generate mask and csg tree of a given point cloud 
		if (result.count("p"))
		{
			flag_sample_pts = false;
		}

		std::string inputext = GetFileExtension(inputfile);
		Mesh3d mesh;
		if (inputext == "obj")
			mesh.load_obj(inputfile.c_str());
		else if (inputext == "off")
			mesh.load_off(inputfile.c_str());
		std::cout << "verts: " << mesh.get_vertices_list()->size() << " face:  " << mesh.get_faces_list()->size() << std::endl;
		bool flag_verbose = false;
		if (result.count("verbose"))
		{
			flag_verbose = (bool)result["verbose"].as<int>();
		}
		
		bool flag_strict = false;
		if (result.count("strict"))
		{
			flag_strict = true;
		}

		if (processing_mode == 0)
		{
			//normalization part begin
			//[-0.9, 0.9]^3
			std::vector<TinyVector<double, 3>> pts_nl(mesh.get_vertices_list()->size());
			double max_range = mesh.xmax - mesh.xmin;
			max_range = max_range < (mesh.ymax - mesh.ymin) ? (mesh.ymax - mesh.ymin) : max_range;
			max_range = max_range < (mesh.zmax - mesh.zmin) ? (mesh.zmax - mesh.zmin) : max_range;

			double xcenter = (mesh.xmin + mesh.xmax) / 2;
			double ycenter = (mesh.ymin + mesh.ymax) / 2;
			double zcenter = (mesh.zmin + mesh.zmax) / 2;
			std::cout << "center " << xcenter << " " << ycenter << " " << zcenter << std::endl;

			for (size_t i = 0; i < mesh.get_vertices_list()->size(); i++)
			{
				mesh.get_vertices_list()->at(i)->pos[0] = (mesh.get_vertices_list()->at(i)->pos[0] - xcenter) / max_range * 1.8;
				mesh.get_vertices_list()->at(i)->pos[1] = (mesh.get_vertices_list()->at(i)->pos[1] - ycenter) / max_range * 1.8;
				mesh.get_vertices_list()->at(i)->pos[2] = (mesh.get_vertices_list()->at(i)->pos[2] - zcenter) / max_range * 1.8;
			}

			//output mesh
			std::string outputext = GetFileExtension(outputfile);
			if (outputext == "obj")
			{
				mesh.write_obj(outputfile.c_str());

			}
			else if (outputext == "off")
			{
				mesh.write_off(outputfile.c_str());
			}
			return 1;
		}
		else if (processing_mode == 1)
		{
			//first sample feature parts then non-feature parts
			//mask: 
			//feature: 0
			//non feature: 1,2,3...indicating coloring
			assert(result.count("f") && result.count("k"));
			auto& inputfeaturefile = result["f"].as<std::string>();
			auto& outputmaskfile = result["k"].as<std::string>();
			if (result.count("fs"))
				n_feature_sample = result["fs"].as<int>();
			if (result.count("ns"))
				n_nonfeature_sample = result["ns"].as<int>();
			if (result.count("s"))
				sigma = result["s"].as<double>();
			if (result.count("sn"))
				sigma_n = result["sn"].as<double>();
			bool flag_repair_turn_features = true;
			if (result.count("r"))
				flag_repair_turn_features = result["r"].as<int>();

			bool flag_repair_tree_features = false;
			if (result.count("repairtree"))
			{
				flag_repair_tree_features = true;
			}
			
			
			if (result.count("csg"))
				flag_csg = result["csg"].as<int>();
			
			bool flag_skip_hanging_features = false; //not skipping hanging features
			std::vector<int> sample_mask;
			std::vector<std::pair<int, int>> ungrouped_features;
			load_feature_file(inputfeaturefile.c_str(), ungrouped_features);
			std::vector<TinyVector<double, 3>> sample_pts, sample_pt_normals;
			std::vector<size_t> sample_pts_tris; //used for assign labels of sample pts
			if (!flag_sample_pts)
			{
				auto& inputpcfile = result["p"].as<std::string>();
				load_xyz_file(inputpcfile.c_str(), sample_pts, sample_pt_normals);

				sample_pts_tris.resize(sample_pts.size(), 0);
				if (result.count("pf"))
				{
					auto& inputpffile = result["pf"].as<std::string>();

					std::ifstream ifs(inputpffile);
					for (size_t ii = 0; ii < sample_pts.size(); ii++)
					{
						ifs >> sample_pts_tris[ii];
					}

					ifs.close();
				}
				else
				{
					//search id by aabb
					std::vector<std::array<double, 3>> tri_verts;
					std::vector<std::vector<size_t>> tri_faces;
					get_mesh_vert_faces(mesh, tri_verts, tri_faces);
					Eigen::MatrixXd input_pts(tri_verts.size(), 3);
					Eigen::MatrixXi input_faces(tri_faces.size(), 3);
					for (size_t i = 0; i < tri_verts.size(); i++)
					{
						for (size_t j = 0; j < 3; j++)
						{
							input_pts(i, j) = tri_verts[i][j];
						}
					}

					for (size_t i = 0; i < tri_faces.size(); i++)
					{
						for (size_t j = 0; j < 3; j++)
						{
							input_faces(i, j) = tri_faces[i][j];
						}
					}


					std::vector<vec3d> closest;
					std::vector<double> dist;
					compute_shortest_dist_AABB(input_pts, input_faces, sample_pts, closest, dist, sample_pts_tris);

				}
				
			}



			//skip elements with no features
			if (flag_csg && ungrouped_features.empty())
			{
				std::cout << "empty feature file: " << inputfeaturefile << std::endl;
				return 1;
			}

			//feature check: no hanging feature
			std::vector<std::vector<int>> feature_v2he(mesh.get_num_of_vertices()); //id of hes ematating from each vertex

			std::vector<std::pair<int, int>> ungrouped_features_new;
			for (size_t i = 0; i < ungrouped_features.size(); i++)
			{
				int id0 = ungrouped_features[i].first;
				int id1 = ungrouped_features[i].second;

				HE_edge<double>* begin_edge = mesh.get_vertices_list()->at(id0)->edge;
				HE_edge<double>* edge = mesh.get_vertices_list()->at(id0)->edge;
				bool flag_found = false;
				do
				{
					if (id1 == edge->vert->id)
					{
						feature_v2he[id0].push_back(edge->id);
						feature_v2he[id1].push_back(edge->pair->id);
						flag_found = true;
						break;
					}
					edge = edge->pair->next;
				} while (edge != begin_edge);
				//assert(flag_found == true);
				if (flag_found == true)
				{
					ungrouped_features_new.push_back(ungrouped_features[i]);
				}
			}

			if (ungrouped_features.size() != ungrouped_features_new.size())
				ungrouped_features = ungrouped_features_new;

			std::vector<size_t> turn_verts, hanging_verts;

			for (size_t i = 0; i < mesh.get_num_of_vertices(); i++)
			{
				if (feature_v2he[i].size() == 1)
				{
					std::cout << "input file: " << inputfile << std::endl;
					std::cout << "hanging vertex exists: " << i << std::endl;
					std::ofstream ofs(inputfile + "hanging");
					ofs.close();
					hanging_verts.push_back(i);
				}
				else if (feature_v2he[i].size() == 2)
				{
					//check edge convex status
					int flag_convex_edge0 = check_mesh_edge_convex(&mesh, mesh.get_edges_list()->at(feature_v2he[i][0]));
					int flag_convex_edge1 = check_mesh_edge_convex(&mesh, mesh.get_edges_list()->at(feature_v2he[i][1]));
					if (flag_convex_edge0 * flag_convex_edge1 != 0 && flag_convex_edge0 != flag_convex_edge1)
					{
						std::ofstream ofs(inputfile + "turn");
						ofs.close();
						std::cout << "input file: " << inputfile << std::endl;
						std::cout << "turn vertex exists: " << i << std::endl;
						turn_verts.push_back(i);
					}
				}
			}

			//do not handle hanging vertex
			if (flag_skip_hanging_features && !hanging_verts.empty())
			{
				return 0;
			}

			//feature parts first
			std::random_device rd;
			std::mt19937 e2(rd());
			std::uniform_real_distribution<double> unif_dist(0, 1);

			std::vector<double> feature_length(ungrouped_features.size(), 0.0);
			double total_feature_length = 0.0;
			for (size_t i = 0; i < ungrouped_features.size(); i++)
			{
				int id0 = ungrouped_features[i].first;
				int id1 = ungrouped_features[i].second;
				feature_length[i] = (mesh.get_vertices_list()->at(id0)->pos - mesh.get_vertices_list()->at(id1)->pos).Length();
				total_feature_length += feature_length[i];
			}
			
			std::vector<double> line_bound(ungrouped_features.size() + 1, 0.0);
			for (size_t i = 0; i < ungrouped_features.size(); i++)
			{
				line_bound[i + 1] = line_bound[i] + feature_length[i] / total_feature_length;
			}

			//sampling
			if (flag_sample_pts)
			{
				for (size_t i = 0; i < n_feature_sample; i++)
				{
					double u = unif_dist(e2);
					auto iter = std::upper_bound(line_bound.begin(), line_bound.end(), u);
					int fid = (int)std::distance(line_bound.begin(), iter);
					assert(fid != ungrouped_features.size() + 1);
					fid = std::max(0, fid - 1);
					//sample
					int id0 = ungrouped_features[fid].first;
					int id1 = ungrouped_features[fid].second;
					double s = unif_dist(e2);
					sample_pts.push_back(s * mesh.get_vertices_list()->at(id0)->pos + (1.0 - s) * mesh.get_vertices_list()->at(id1)->pos);
					sample_pt_normals.push_back(TinyVector<double, 3>(1.0, 0.0, 0.0));
					sample_mask.push_back(0);
				}
			}
			std::vector<TinyVector<size_t, 3>> tri_verts(mesh.get_faces_list()->size());
			std::vector<TinyVector<double, 3>> tri_normals;
			//get tri list
			for (size_t i = 0; i < mesh.get_faces_list()->size(); i++)
			{
				HE_edge<double>* begin_edge = mesh.get_faces_list()->at(i)->edge;
				HE_edge<double>* edge = mesh.get_faces_list()->at(i)->edge;
				int local_id = 0;
				do
				{
					tri_verts[i][local_id++] = edge->pair->vert->id;
					edge = edge->next;
				} while (edge != begin_edge);
				tri_normals.push_back(mesh.get_faces_list()->at(i)->normal);
			}
			std::vector<TinyVector<double, 3>> vert_pos;
			for (size_t i = 0; i < mesh.get_vertices_list()->size(); i++)
			{
				vert_pos.push_back(mesh.get_vertices_list()->at(i)->pos);
			}

			//cluster faces: assuming the features are all close
			std::vector<bool> he_feature_flag(mesh.get_edges_list()->size(), false);
			for (size_t i = 0; i < ungrouped_features.size(); i++)
			{
				int id0 = ungrouped_features[i].first;
				int id1 = ungrouped_features[i].second;
				//iterate over all verts emanating from id0
				HE_edge<double>* edge = mesh.get_vertices_list()->at(id0)->edge;
				do
				{
					if (edge->vert->id == id1)
					{
						break;
					}
					edge = edge->pair->next;
				} while (edge != mesh.get_vertices_list()->at(id0)->edge);
				assert(edge->vert->id == id1);
				he_feature_flag[edge->id] = true;
				he_feature_flag[edge->pair->id] = true;
			}
			
			std::vector<std::vector<int>> grouped_features; //grouped features: only one he of a pair is stored
			std::vector<int> he2gid;
			get_grouped_edges(mesh, he_feature_flag, feature_v2he, grouped_features, he2gid);
			if (flag_repair_turn_features && !turn_verts.empty())
			{
				//repair turning-point features
				std::vector<bool> flag_feature_points(mesh.get_vertices_list()->size(), false);
				for (size_t i = 0; i < he_feature_flag.size(); i++)
				{
					if (he_feature_flag[i])
					{
						flag_feature_points[mesh.get_edges_list()->at(i)->vert->id] = true;
					}
				}

				//repair turn vertex
				for (size_t i = 0; i < turn_verts.size(); i++)
				{
					size_t cur_vert = turn_verts[i];
					std::vector<bool> flag_feature_points_tmp = flag_feature_points;
					size_t cur_group = he2gid[feature_v2he[cur_vert][0]];
					for (auto heid : grouped_features[cur_group])
					{
						flag_feature_points_tmp[mesh.get_edges_list()->at(heid)->vert->id] = false;
						flag_feature_points_tmp[mesh.get_edges_list()->at(heid)->pair->vert->id] = false;
					}

					//distance from cur_vert to all other verts
					std::vector<size_t> prev_map;
					std::vector<double> dist;
					dijkstra_mesh(&mesh, cur_vert, prev_map, dist);
					
					std::set<std::pair<double, size_t>> dist_id_set;
					for (size_t j = 0; j < flag_feature_points_tmp.size(); j++)
					{
						if (flag_feature_points_tmp[j])
						{
							dist_id_set.insert(std::pair<double, size_t>(dist[j], j));
						}
					}

					for (auto& dist_id : dist_id_set)
					{
						size_t nnvid = dist_id.second;
						bool flag_usable = true;
						std::vector<HE_edge<double>*> tmp_hes;
						//add path from nnvid to cur_vert
						while (nnvid != cur_vert)
						{
							size_t prev_vert = prev_map[nnvid];
							HE_edge<double>* begin_edge = mesh.get_vertices_list()->at(nnvid)->edge;
							HE_edge<double>* edge = begin_edge;
							do
							{
								if (edge->vert->id == prev_vert)
								{
									break;
								}
								edge = edge->pair->next;
							} while (edge != begin_edge);
							assert(edge->vert->id == prev_vert);
							if (he_feature_flag[edge->id])
							{
								flag_usable = false;
								break;
							}
							else
							{
								tmp_hes.push_back(edge);
							}

							nnvid = prev_vert;
						}
						if (flag_usable)
						{
							for (auto he : tmp_hes)
							{
								ungrouped_features.push_back(std::pair<int, int>(he->vert->id, he->pair->vert->id));
								he_feature_flag[he->id] = true;
								he_feature_flag[he->pair->id] = true;
								feature_v2he[he->vert->id].push_back(he->pair->id);
								feature_v2he[he->pair->vert->id].push_back(he->id);
							}
							break;
						}
					}
				}
				//save repaired features
				save_feature_file((output_prefix + "_repairturn.fea").c_str(), ungrouped_features);
			}
			
			std::vector<int> face_color_init(mesh.get_faces_list()->size(), -1); //starting from 1
			std::vector<std::vector<int>> face_clusters;
			std::vector<std::pair<int, int>> feature_twoface_colors;//color of faces on two sides of the feature, *.first < *.second

			int start_id = 0;
			int cluster_id = 1;
			while (start_id != -1)
			{
				std::vector<int> onecluster;
				std::queue<int> q;
				q.push(start_id);
				face_color_init[start_id] = cluster_id;
				while (!q.empty())
				{
					int front = q.front();
					q.pop();
					onecluster.push_back(front);
					HE_edge<double>* edge = mesh.get_faces_list()->at(front)->edge;
					do
					{
						if (he_feature_flag[edge->id] == false)
						{
							//not feature
							int pair_fid = edge->pair->face->id;
							if (face_color_init[pair_fid] == -1)
							{
								q.push(pair_fid);
								face_color_init[pair_fid] = cluster_id;
							}
						}

						edge = edge->next;
					} while (edge != mesh.get_faces_list()->at(front)->edge);

				}

				face_clusters.push_back(onecluster);
				start_id = -1;
				//find next start_id
				for (size_t i = 0; i < face_color_init.size(); i++)
				{
					if (face_color_init[i] == -1)
					{
						start_id = i;
						break;
					}
				}
				cluster_id++;
			}
			std::vector<int> face_color = face_color_init; //starting from 1
			//check face area
			std::vector<double> tri_areas;
			get_all_tri_area(mesh, tri_areas);
			std::vector<double> patch_areas(cluster_id, 0.0);
			for (size_t i = 0; i < face_color.size(); i++)
			{
				patch_areas[face_color[i]] += tri_areas[i];
			}
			
			
			if (*std::min_element(patch_areas.begin() + 1, patch_areas.end()) < MIN_PATCH_AREA)
			{
				std::cout << "too small patch area : " << *std::min_element(patch_areas.begin() + 1, patch_areas.end()) << std::endl;
				std::ofstream ofs(output_prefix + "smallpatch");
				ofs.close();
			}

			int n_color = cluster_id;
			//coloring
			bool flag_coloring = false;
			if (result.count("c"))
				flag_coloring = result["c"].as<int>();

			

			bool flag_first_convex = false;
			if (result.count("convex"))
				flag_first_convex = result["convex"].as<int>();

			if (flag_coloring && !flag_csg)
			{	
				std::vector<std::set<size_t>> connectivity(cluster_id - 1);
				//color - 1
				//here assume faces on both sides of a feature belongs to df patches
				for (size_t i = 0; i < he_feature_flag.size(); i++)
				{
					if (he_feature_flag[i])
					{
						HE_edge<double>* e1 = mesh.get_edges_list()->at(i);
						HE_edge<double>* e2 = e1->pair;
						if (face_color_init[e1->face->id] == face_color_init[e2->face->id]) continue;
						connectivity[face_color_init[e1->face->id] - 1].insert(face_color_init[e2->face->id] - 1);
					}
				}

				//print graph
				std::cout << "graph:" << std::endl;
				for (size_t i = 0; i < connectivity.size(); i++)
				{
					std::cout << i + 1 << ": ";
					for (auto v : connectivity[i])
					{
						std::cout << v + 1 << " ";
					}
					std::cout << std::endl;
				}

				std::vector<std::vector<size_t>> colored_vertices;
				greedy_graph_coloring(cluster_id - 1, connectivity, colored_vertices);
				std::cout << "number of colors: " << colored_vertices.size() << std::endl;
				n_color = colored_vertices.size() + 1;
				//update face_color
				for (size_t i = 0; i < colored_vertices.size(); i++)
				{
					for (size_t j = 0; j < colored_vertices[i].size(); j++)
					{
						size_t local_id = colored_vertices[i][j];
						for (size_t k = 0; k < face_clusters[local_id].size(); k++)
						{
							face_color[face_clusters[local_id][k]] = i + 1;
						}
					}
				}
			}

			if (flag_csg)
			{
				//coloring is based on vertices
				std::vector<std::set<size_t>> connectivity(cluster_id - 1);
				std::vector<std::set<size_t>> connectivity_v(cluster_id - 1); //connectivity based on vertices
				std::map<std::pair<size_t, size_t>, double> fp2product;
				
				//construct csg tree by voting				
				std::map<std::pair<size_t, size_t>, int> flag_fpconvex; //update 1203, 0: smooth, 1: convex, 2:concave
				//color - 1
				
				//face pair convexity is determined by grouped edges
				std::vector<std::array<int, 3>> ge2count(grouped_features.size(), std::array<int, 3>{0, 0, 0});
				std::map<std::pair<size_t, size_t>, std::set<int>> fp2ge; //face pair to grouped edges
				for (size_t i = 0; i < he_feature_flag.size(); i++)
				{
					if (he_feature_flag[i])
					{
						HE_edge<double>* e1 = mesh.get_edges_list()->at(i);
						HE_edge<double>* e2 = e1->pair;
						if (face_color_init[e1->face->id] != face_color_init[e2->face->id])
							connectivity[face_color_init[e1->face->id] - 1].insert(face_color_init[e2->face->id] - 1);
						else
							continue;
						size_t fid1 = face_color_init[e1->face->id], fid2 = face_color_init[e2->face->id]; //starting from zero
						size_t minfid = std::min(fid1, fid2);
						size_t maxfid = std::max(fid1, fid2);
						
						//triangle face
						size_t tfid1 = e1->face->id, tfid2 = e2->face->id;
						size_t ev1 = e1->vert->id, ev2 = e2->vert->id;
						size_t tv1 = tri_verts[tfid1][0] + tri_verts[tfid1][1] + tri_verts[tfid1][2] - ev1 - ev2;
						size_t tv2 = tri_verts[tfid2][0] + tri_verts[tfid2][1] + tri_verts[tfid2][2] - ev1 - ev2;
						double product = e1->face->normal.Dot(vert_pos[tv2] - vert_pos[tv1]);

						std::pair<size_t, size_t> tmp_pair(minfid - 1, maxfid - 1);
						auto it = fp2product.find(tmp_pair);
						if (it == fp2product.end())
						{
							fp2product[tmp_pair] = product;
						}
						else
						{
							fp2product[tmp_pair] += product;
						}
						
						double tmp_cos = e1->face->normal.Dot(e2->face->normal);

						int gid = he2gid[i];
						if (fp2ge.find(tmp_pair) == fp2ge.end())
						{
							fp2ge[tmp_pair] = std::set<int>();
						}
						fp2ge[tmp_pair].insert(gid);
						
						if (product < 0.0)
						{
							//convex
							if (tmp_cos < th_smooth_cos_value)
							{
								ge2count[gid][1]++;
							}
							else
							{
								ge2count[gid][0]++;
							}
						}
						else
						{
							//concave
							if (tmp_cos < th_smooth_cos_value)
							{
								ge2count[gid][2]++;
							}
							else
							{
								ge2count[gid][0]++;
							}
						}
					}
				}

				//init connectivity_v
				connectivity_v = connectivity;
				for (size_t i = 0; i < feature_v2he.size(); i++)
				{
					if (feature_v2he[i].size() > 2)
					{
						//vertex with degree larger than 3
						HE_edge<double>* ve_begin = mesh.get_vertices_list()->at(i)->edge;
						assert(ve_begin->pair->vert->id == i);
						HE_edge<double>* ve_iter = ve_begin;
						std::set<size_t> surounding_cs;
						do
						{
							surounding_cs.insert(face_color_init[ve_iter->face->id] -1);
							ve_iter = ve_iter->pair->next;
						} while (ve_iter != ve_begin);
						std::vector<size_t> surounding_cs_vector(surounding_cs.begin(), surounding_cs.end());
						for (size_t it = 0; it < surounding_cs_vector.size() - 1; it++)
						{
							size_t cur_fid = surounding_cs_vector[it];
							for (size_t it1 = it + 1; it1 < surounding_cs_vector.size(); it1++)
							{
								size_t n_fid = surounding_cs_vector[it1];
								connectivity_v[cur_fid].insert(n_fid);
								connectivity_v[n_fid].insert(cur_fid);
							}
						}
					}
				}
				

				if (flag_strict)
				{
					for (auto& p : fp2product)
					{
						if (p.second < 0.0)
						{
							flag_fpconvex[p.first] = 1;
						}
						else
						{
							flag_fpconvex[p.first] = 2;

						}
					}
				}
				else
				{
					//get face pair connectivity by grouped edges
					bool flag_valid_seg = true;
					std::set<size_t> invalid_seg_patches;
					std::vector<int> ge2convex(grouped_features.size(), 0);
					for (size_t i = 0; i < grouped_features.size(); i++)
					{
						int maxid = -1, max_num = -1;
						for (size_t ii = 0; ii < 3; ii++)
						{
							if (ge2count[i][ii] > max_num)
							{
								max_num = ge2count[i][ii];
								maxid = ii;
							}
						}
						ge2convex[i] = maxid;
					}
					
					for (auto& p : fp2ge)
					{
						std::set<int> cur_strict_convex_types;

						for (auto gid : p.second)
						{
							if (ge2convex[gid] != 0)
								cur_strict_convex_types.insert(ge2convex[gid]);
						}


						if (cur_strict_convex_types.size() >= 2)
						{
							flag_valid_seg = false;
							invalid_seg_patches.insert(p.first.first);
							invalid_seg_patches.insert(p.first.second);
						}
						else if (cur_strict_convex_types.size() == 0)
						{
							flag_fpconvex[p.first] = 0; //smooth
						}
						else
						{
							//only one left
							flag_fpconvex[p.first] = *cur_strict_convex_types.begin();
						}
					}

					if (!flag_valid_seg)
					{
						std::ofstream ofs(inputfile + "treefail");
						ofs.close();
						std::cout << "invalid segmentation " << output_prefix << std::endl;
						repair_tree_features_maxflow(mesh, face_color, std::vector<size_t>(invalid_seg_patches.begin(), invalid_seg_patches.end()), ungrouped_features);

						ofs.open(output_prefix + "_fixtree.fea");
						ofs << ungrouped_features.size() << std::endl;
						for (auto& pp : ungrouped_features)
						{
							ofs << pp.first << " " << pp.second << std::endl;
						}
						ofs.close();
						mesh.write_obj((output_prefix + "_fixtree.obj").c_str());

						return 1;
					}

				}

				

				if (flag_verbose)
				{
					std::cout << "dual graph ori:" << std::endl;
					for (size_t i = 0; i < connectivity.size(); i++)
					{
						std::cout << i << ": ";
						for (auto v : connectivity[i])
						{
							std::cout << v << " ";
						}
						std::cout << std::endl;
					}

					std::cout << "edge convex flag: " << std::endl;
					for (auto& p : flag_fpconvex)
					{
						std::cout << "edge: " << p.first.first << "-" << p.first.second << " : " << p.second << std::endl;
					}
				}
				//print graph

				TreeNode<size_t> *tree = new TreeNode<size_t>;
				;
				bool flag_convex = true;
				//convex flag is set to true by default, but if the model contains multiple component, then it is set to concave

				std::vector<std::set<size_t>> components;
				get_graph_component(connectivity, components);

				bool flag_construct_tree = true;
				std::vector<size_t> invalid_subgraph;
				if (components.size() == 1)
				{ 
					flag_construct_tree = get_tree_from_convex_graph(connectivity, flag_fpconvex, true, tree, 0, invalid_subgraph);
					
					if (!flag_construct_tree)
					{
						delete tree;
						tree = new TreeNode<size_t>;
						invalid_subgraph.clear();
						flag_construct_tree = get_tree_from_convex_graph(connectivity, flag_fpconvex, false, tree, 0, invalid_subgraph);

					}
				}
				else
				{
					flag_convex = false;
					for (size_t ic = 0; ic < components.size(); ic++)
					{
						std::vector<std::set<size_t>> subgraph(connectivity.size());
						for (auto v : components[ic])
						{
							for (auto vn : connectivity[v])
							{
								if (std::find(components[ic].begin(), components[ic].end(), vn) != components[ic].end())
								{
									subgraph[v].insert(vn);
								}
							}
						}
						TreeNode<size_t>* child = new TreeNode<size_t>;
						flag_construct_tree = get_tree_from_convex_graph(subgraph, flag_fpconvex, !flag_convex, child, 1, invalid_subgraph);
						if (!flag_construct_tree)
							break;
						tree->children.push_back(child);
					}
				}

				if (!flag_construct_tree)
				{
					std::ofstream ofs(inputfile + "treefail");
					ofs.close();
					std::cout << "tree construction failed for model " << inputfile << std::endl;
					//failure case
					if (flag_repair_tree_features) //to check for all feature
					{
						repair_tree_features_maxflow(mesh, face_color, invalid_subgraph, ungrouped_features);
						ofs.open(output_prefix + "_fixtree.fea");
						ofs << ungrouped_features.size() << std::endl;
						for (auto& pp : ungrouped_features)
						{
							ofs << pp.first << " " << pp.second << std::endl;
						}
						ofs.close();

						mesh.write_obj((output_prefix + "_fixtree.obj").c_str());
					}

					return 1;
				}

				std::cout << "convex status: " << flag_convex << std::endl;
				
				if (flag_coloring)
				{
					int max_patch_per_cluster = -1;
					if (result.count("mp"))
						max_patch_per_cluster = result["mp"].as<int>();
					std::vector<size_t> cluster_color(connectivity_v.size(), -1);
					n_color = tree_coloring<size_t>(tree, connectivity_v, cluster_color, 0, max_patch_per_cluster) + 1;
					for (size_t i = 0; i < face_clusters.size(); i++)
					{
						size_t cc = cluster_color[i];
						for (size_t j = 0; j < face_clusters[i].size(); j++)
						{
							face_color[face_clusters[i][j]] = cc + 1;
						}
					}
					
					update_tree_color(cluster_color, tree);

				}

				std::string tree_str = convert_tree_to_string<size_t>(tree);
				save_conf_file((output_prefix + "_csg.conf").c_str(), tree_str, flag_convex);
			}
			
			//sampling on triangles
			std::vector<double> tri_mean_curvature_normalize(tri_verts.size(), 0.0);
			if (result.count("cot"))
			{
				//compute tri_mean_curvature
				std::vector<double> vert_curvature;
				compute_vert_mean_curvature(vert_pos, tri_verts, vert_curvature);
				for (size_t i = 0; i < tri_verts.size(); i++)
				{
					for (size_t j = 0; j < 3; j++)
					{
						tri_mean_curvature_normalize[i] += vert_curvature[tri_verts[i][j]];
					}
				}
				auto maxele = std::max_element(begin(tri_mean_curvature_normalize), end(tri_mean_curvature_normalize));
				auto minele = std::min_element(begin(tri_mean_curvature_normalize), end(tri_mean_curvature_normalize));
				std::cout << "min curvature: " << *minele << " max curvature: " << *maxele << std::endl;
				double diff = *maxele - *minele;
				for (size_t i = 0; i < tri_verts.size(); i++)
				{
					tri_mean_curvature_normalize[i] = (tri_mean_curvature_normalize[i] - *minele) / diff;
				}
			}
			//first sample at least min_pts_per_patch pts from each patch
			if (flag_sample_pts)
			{
				int max_face_color = *std::max_element(face_color.begin(), face_color.end());
				std::cout << "num of patches: " << max_face_color << std::endl;
				for (size_t i = 1; i < max_face_color + 1; i++)
				{
					std::vector<TinyVector<size_t, 3>> cur_faces;
					std::vector<TinyVector<double, 3>> cur_normals;
					for (size_t j = 0; j < face_color.size(); j++)
					{
						if (face_color[j] == i)
						{
							cur_faces.push_back(tri_verts[j]);
							cur_normals.push_back(tri_normals[j]);
						}
					}

					std::vector<int> cur_face_mask(cur_faces.size(), i);
					std::vector<TinyVector<double, 3>> cur_sample_pts, cur_sample_normals;
					std::vector<int> cur_sample_masks;
					sample_pts_from_mesh(vert_pos, cur_faces, cur_normals, cur_face_mask, min_sample_perpatch, cur_sample_pts, cur_sample_normals, cur_sample_masks, sigma, sigma_n);

					sample_pts.insert(sample_pts.end(), cur_sample_pts.begin(), cur_sample_pts.end());
					sample_pt_normals.insert(sample_pt_normals.end(), cur_sample_normals.begin(), cur_sample_normals.end());
					sample_mask.insert(sample_mask.end(), cur_sample_masks.begin(), cur_sample_masks.end());
				}

				if (sample_mask.size() < n_nonfeature_sample)
				{
					//sample the rest
					std::vector<TinyVector<double, 3>> cur_sample_pts, cur_sample_normals;
					std::vector<int> cur_sample_masks;
					sample_pts_from_mesh(vert_pos, tri_verts, tri_normals, face_color, n_nonfeature_sample - sample_mask.size(), cur_sample_pts, cur_sample_normals, cur_sample_masks, sigma, sigma_n);
					sample_pts.insert(sample_pts.end(), cur_sample_pts.begin(), cur_sample_pts.end());
					sample_pt_normals.insert(sample_pt_normals.end(), cur_sample_normals.begin(), cur_sample_normals.end());
					sample_mask.insert(sample_mask.end(), cur_sample_masks.begin(), cur_sample_masks.end());
				}
				else
				{
					std::cout << "too many patches for model: " << inputfile << std::endl;
				}

				std::ofstream outputsamples(outputfile.c_str());

				for (size_t i = 0; i < sample_pts.size(); i++)
				{
					outputsamples << sample_pts[i] << " " << sample_pt_normals[i] << std::endl;
				}

				outputsamples.close();
			}
			else
			{
				for (size_t i = 0; i < sample_pts_tris.size(); i++)
				{
					sample_mask.push_back(face_color[sample_pts_tris[i]]);
				}
			}
			
			//output ply
			std::vector<std::array<double, 3>> branch_color;
			for (size_t i = 0; i < n_color; i++)
			{
				branch_color.push_back(std::array<double, 3>{ {unif_dist(e2), unif_dist(e2), unif_dist(e2)}});
			}

			std::vector<std::array<double, 3>> meshVertexPositions;
			std::vector<std::array<double, 3>> meshVertexColors;

			for (size_t i = 0; i < sample_pts.size(); i++)
			{
				meshVertexPositions.push_back(std::array<double, 3>{ {sample_pts[i][0], sample_pts[i][1], sample_pts[i][2]}});
				meshVertexColors.push_back(branch_color[sample_mask[i]]);
			}


			// Create an empty object
			happly::PLYData plyOut;

			// Add mesh data (elements are created automatically)
			plyOut.addVertexPositions(meshVertexPositions);
			plyOut.addVertexColors(meshVertexColors);

			// Write the object to file
			plyOut.write(output_prefix + "_patch_"+ std::to_string(n_color - 1) +".ply", happly::DataFormat::ASCII);

			std::ofstream outputmask(outputmaskfile.c_str());
			for (size_t i = 0; i < sample_mask.size(); i++)
			{
				outputmask << sample_mask[i] << std::endl;
			}

			outputmask.close();

			//output face seg id: starting from 0
			outputmask.open(output_prefix + "_fid.txt");
			for (size_t i = 0; i < face_color.size(); i++)
			{
				outputmask << face_color[i] - 1 << std::endl;
			}

			outputmask.close();

			return 1;
		}

	}
	catch (const cxxopts::OptionException& e)
	{
		std::cout << "error parsing options: " << e.what() << std::endl;
		exit(1);
	}
	return 0;
}