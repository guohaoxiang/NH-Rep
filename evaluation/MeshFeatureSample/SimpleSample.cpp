#include <iostream>
#include <fstream>
#include "Mesh3D.h"
#include "TinyVector.h"
#include "happly/happly.h"
#include "cxxopts.hpp"

#define SHORTEST_EDGE_LENGTH 1e-6


#define th_smooth_cos_value 0.939692 //for grouping case, this angle is used to distinguish sharp and non-sharp feauture


using namespace MeshLib;

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

	//not consider grouped feature
	return true;

}

void select_features_by_angle(Mesh3d* m, double th_angle, std::vector<std::pair<int, int>> &ungrouped_features, std::vector<int> &ungrouped_feature_eid)
{
	ungrouped_features.clear();
	ungrouped_feature_eid.clear();
	const double angle = th_angle * 3.1415926535897932384626433832795 / 180;
	std::vector<bool> he_flag(m->get_num_of_edges(), false);
	for (size_t i = 0; i < m->get_num_of_edges(); i++)
	{
		HE_edge<double>* he = m->get_edge(i);
		if (!he_flag[he->id])
		{
			if (!m->is_on_boundary(he) && acos(he->face->normal.Dot(he->pair->face->normal)) > angle)
			{
				he_flag[he->id] = true;
				he_flag[he->pair->id] = true;
				ungrouped_features.push_back(std::pair<int, int>(he->vert->id, he->pair->vert->id));
				ungrouped_feature_eid.push_back(he->id);
			}
		}
	}
}

void output_pts_xyz(const std::vector<TinyVector<double, 3>>& pts, const char* filename)
{
	std::ofstream ofs(filename);
	for (size_t i = 0; i < pts.size(); i++)
	{
		ofs << pts[i] << std::endl;
	}

	ofs.close();
}

void get_grouped_edges(Mesh3d& mesh, const std::vector<bool>& he_feature_flag, const std::vector<std::vector<int>>& feature_v2he, std::vector<std::vector<int>>& grouped_features, std::vector<int>& he2gid)
{
	grouped_features.clear();
	he2gid.clear();
	he2gid.resize(mesh.get_num_of_edges(), -1);
	int groupid = 0;
	while (true)
	{
		int first_he = -1;
		for (size_t i = 0; i < he_feature_flag.size(); i++)
		{
			if (he_feature_flag[i] && he2gid[i] == -1)
			{
				first_he = i;
				break;
			}
		}
		if (first_he == -1)
		{
			break;
		}

		std::vector<int> onegroup;
		std::queue<size_t> q;
		q.push(first_he);
		he2gid[first_he] = groupid;
		he2gid[mesh.get_edges_list()->at(first_he)->pair->id] = groupid;
		while (!q.empty())
		{
			size_t curhe = q.front();
			onegroup.push_back(curhe);
			q.pop();
			size_t curhe_pair = mesh.get_edges_list()->at(curhe)->pair->id;
			std::vector<size_t> twoverts;
			twoverts.push_back(mesh.get_edges_list()->at(curhe)->vert->id);
			twoverts.push_back(mesh.get_edges_list()->at(curhe)->pair->vert->id);
			std::vector<size_t> twoedges;
			twoedges.push_back(curhe_pair);
			twoedges.push_back(curhe);
			for (size_t i = 0; i < 2; i++)
			{
				//only accept points with degree 2
				if (feature_v2he[twoverts[i]].size() == 2)
				{
					assert(twoedges[i] == feature_v2he[twoverts[i]][0] || twoedges[i] == feature_v2he[twoverts[i]][1]);
					size_t other_he = feature_v2he[twoverts[i]][0] + feature_v2he[twoverts[i]][1] - twoedges[i];
					if (he2gid[other_he] == -1)
					{
						q.push(other_he);
						he2gid[other_he] = groupid;
						he2gid[mesh.get_edges_list()->at(other_he)->pair->id] = groupid;
					}
				}
			}
		}
		grouped_features.push_back(onegroup);
		groupid++;
	}
}

void select_real_sharp_features(Mesh3d& mesh, std::vector<std::pair<int, int>>& ungrouped_features)
{
	//std::vector<bool> he_feature_flag;
	std::vector<bool> he_feature_flag(mesh.get_edges_list()->size(), false);

	//std::vector<std::vector<int>> feature_v2he;
	std::vector<std::vector<int>> feature_v2he(mesh.get_num_of_vertices()); //id of hes ematating from each vertex

	for (size_t i = 0; i < ungrouped_features.size(); i++)
	{
		int id0 = ungrouped_features[i].first;
		int id1 = ungrouped_features[i].second;
		//feature_degree_v[id0]++;
		//feature_degree_v[id1]++;

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
	}

	std::vector<std::vector<int>> grouped_features;
	std::vector<int> he2gid;

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
		//mesh.get_faces_list()->at(fid)->normal;
		tri_normals.push_back(mesh.get_faces_list()->at(i)->normal);
	}
	std::vector<TinyVector<double, 3>> vert_pos;
	for (size_t i = 0; i < mesh.get_vertices_list()->size(); i++)
	{
		vert_pos.push_back(mesh.get_vertices_list()->at(i)->pos);
	}

	//step1: get he_feature_flag
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
		//assert(edge->vert->id == id1);
		if (edge->vert->id == id1)
		{
			he_feature_flag[edge->id] = true;
			he_feature_flag[edge->pair->id] = true;
		}
	}
	//step 2: get grouped feature
	get_grouped_edges(mesh, he_feature_flag, feature_v2he, grouped_features, he2gid);
	//step 3: select real sharp features
	std::vector<std::pair<int, int>> ungrouped_features_update;
	std::vector<std::array<int, 3>> ge2count(grouped_features.size(), std::array<int, 3>{0, 0, 0});
	for (size_t i = 0; i < he_feature_flag.size(); i++)
	{
		if (he_feature_flag[i])
		{
			HE_edge<double>* e1 = mesh.get_edges_list()->at(i);
			HE_edge<double>* e2 = e1->pair;

			//triangle face
			size_t tfid1 = e1->face->id, tfid2 = e2->face->id;
			size_t ev1 = e1->vert->id, ev2 = e2->vert->id;
			size_t tv1 = tri_verts[tfid1][0] + tri_verts[tfid1][1] + tri_verts[tfid1][2] - ev1 - ev2;
			size_t tv2 = tri_verts[tfid2][0] + tri_verts[tfid2][1] + tri_verts[tfid2][2] - ev1 - ev2;
			double product = e1->face->normal.Dot(vert_pos[tv2] - vert_pos[tv1]);


			double tmp_cos = e1->face->normal.Dot(e2->face->normal);
			//if (fp2count.find(tmp_pair) == fp2count.end())


			/*if (ge2count.find(tmp_pair) == ge2count.end())
			{
				fp2count[tmp_pair] = std::array<int, 3>({ 0, 0, 0 });
			}*/

			int gid = he2gid[i];

			if (product < 0.0)
			{
				//convex
				if (tmp_cos < th_smooth_cos_value)
				{
					//fp2count[tmp_pair][1]++;
					ge2count[gid][1]++;
				}
				else
				{
					//fp2count[tmp_pair][0]++;
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

		//grouped feature no duplicate
	}

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
		if (maxid != 0)
		//if (true)
		{
			//sharp features
			//std::cout << i << " grouped feature size: " << grouped_features[i].size() << std::endl;
			for (auto eid : grouped_features[i])
			{
				int v0 = mesh.get_edge(eid)->vert->id;
				int v1 = mesh.get_edge(eid)->pair->vert->id;
				ungrouped_features_update.push_back(std::pair<int, int>(v0, v1));
			}
		}
	}

	ungrouped_features = ungrouped_features_update;
}

int main(int argc, char** argv)
{
	try
	{
		cxxopts::Options options("SimpleSample", "Sampling feature points from points on mesh (author: Haoxiang Guo, Email: guohaoxiangxiang@gmail.com)");
		options
			.positional_help("[optional args]")
			.show_positional_help()
			.allow_unrecognised_options()
			.add_options()
			("i,input", "input mesh (obj/off/ply format)", cxxopts::value<std::string>())
			("f,feature", "input feature file (fea format)", cxxopts::value<std::string>())
			("o,output", "output points (ptangle format)", cxxopts::value<std::string>())
			("a", "angle threshold in degree for detecting features, default(30)", cxxopts::value<double>())
			("s", "length of line segment for sampling, default(4e-3)", cxxopts::value<double>())
			("g,group", "group feature edge");
			("h,help", "print help");

		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help({ "", "Group" }) << std::endl;
			exit(0);
		}
		bool flag_feature_flag = true;
		double len_seg = 4e-3;
		double th_angle = 30;
		bool flag_group_features = false; //group input feature to decide its convexity
		if (result.count("s"))
		{
			len_seg = result["s"].as<double>();
		}
		if (result.count("a"))
		{
			//angle is used only when no feature files are given
			th_angle = result["a"].as<double>();
		}

		if (result.count("g"))
		{
			flag_group_features = true;
		}

		auto& inputfile = result["i"].as<std::string>();
		auto& outputfile = result["o"].as<std::string>();

		int last_dot = (int)outputfile.find_last_of(".");
		auto output_prefix = outputfile.substr(0, last_dot);
		
		std::string inputext = GetFileExtension(inputfile);
		Mesh3d mesh;
		if (inputext == "obj")
			mesh.load_obj(inputfile.c_str());
		else if (inputext == "off")
			mesh.load_off(inputfile.c_str());
		else if (inputext == "ply")
		{
			happly::PLYData plyIn(inputfile);
			std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
			std::vector<std::vector<size_t>> fInd = plyIn.getFaceIndices<size_t>();
			mesh.load_mesh(vPos, fInd);

			//test code below
			//mesh.write_obj("test.obj");
		}
		std::cout << "verts: " << mesh.get_vertices_list()->size() << " face:  " << mesh.get_faces_list()->size() << std::endl;

		std::vector<std::pair<int, int>> ungrouped_features;
		std::vector<int> ungrouped_feature_eid;
		if (result.count("f"))
		{
			auto& inputfeaturefile = result["f"].as<std::string>();
			load_feature_file(inputfeaturefile.c_str(), ungrouped_features);

			if (flag_group_features)
			{
				//repair and group features
				select_real_sharp_features(mesh, ungrouped_features);
			}
		}
		else
		{
			//select feature by angle
			select_features_by_angle(&mesh, th_angle, ungrouped_features, ungrouped_feature_eid);
		}
		
		//sample points
		std::vector<TinyVector<double, 3>> sample_pts;
		std::vector<double> pts_angle;
		//for (auto& pair : ungrouped_features)
		for (size_t i = 0; i < ungrouped_features.size(); i++)
		{
			auto pair = ungrouped_features[i];
			TinyVector<double, 3> pos0 = mesh.get_vertex(pair.first)->pos;
			TinyVector<double, 3> pos1 = mesh.get_vertex(pair.second)->pos;
			double len = (pos0 - pos1).Length();
			if (len > SHORTEST_EDGE_LENGTH)
			{
				//sample at least one points
				double angle = 0.0;
				if (ungrouped_feature_eid.empty())
				{
					bool flag = false;

					HE_edge<double>* begin_edge = mesh.get_vertex(pair.first)->edge;
					HE_edge<double>* edge = begin_edge;
					do
					{
						if (edge->vert->id == pair.second)
						{
							flag = true;
							angle = acos(edge->face->normal.Dot(edge->pair->face->normal)) * 180.0 / 3.1415926535897932384626433832795;
							break;
						}

						edge = edge->pair->next;
					} while (edge != begin_edge);

					assert(flag == true);
				}
				else
				{
					HE_edge<double>* edge = mesh.get_edge(ungrouped_feature_eid[i]);
					angle = acos(edge->face->normal.Dot(edge->pair->face->normal)) * 180.0 / 3.1415926535897932384626433832795;
				}
				
				if (len < len_seg / 2.0)
				{
					sample_pts.push_back((pos0 + pos1) / 2.0);
					pts_angle.push_back(angle);
				}
				else
				{
					int n_split = std::ceil((len - len_seg / 2.0) / len_seg);
					TinyVector<double, 3> vec =  (pos1 - pos0)/(pos1 - pos0).Length();
					for (size_t i = 0; i < n_split; i++)
					{
						double tmp_len = len_seg / 2.0 + i * len_seg;
						sample_pts.push_back(pos0 + tmp_len * vec);
						pts_angle.push_back(angle);
					}
				}
			}
		}
		assert(pts_angle.size() == sample_pts.size());

		//output_xyz
		output_pts_xyz(sample_pts, (output_prefix + ".xyz").c_str());
		
		std::ofstream ofs(outputfile);
		for (size_t i = 0; i < pts_angle.size(); i++)
		{
			ofs << sample_pts[i] << " " << pts_angle[i] << std::endl;
		}

		ofs.close();

		//smoothness term
	}
	catch (const cxxopts::OptionException& e)
	{
		std::cout << "error parsing options: " << e.what() << std::endl;
		exit(1);
	}

	return 0;
}