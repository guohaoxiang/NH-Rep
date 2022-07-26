#include "helper.h"
#include <algorithm>
#include <queue>
#include <map>
#include <igl/gaussian_curvature.h>
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/readOFF.h>

#include "MaxFlow.h"

#define DENOMINATOR_EPS 1e-6

typedef TinyVector<double, 3> vec3d;

void greedy_graph_coloring(const size_t num_vertices, const std::vector<std::set<size_t>>& edges, std::vector<std::vector<size_t>>& colored_vertices)
{
	std::vector<int> result(num_vertices, -1);
	result[0] = 0;
	std::vector<bool> available(num_vertices, false);

	int max_color = 0;
	for (size_t u = 1; u < num_vertices; u++)
	{
		for (typename std::set<size_t>::iterator iter = edges[u].begin(); iter != edges[u].end(); iter++)
		{
			if (result[*iter] != -1)
				available[result[*iter]] = true;
		}
		int cr;
		for (cr = 0; cr < (int)num_vertices; cr++)
		{
			if (available[cr] == false)
				break;
		}

		result[u] = cr;
		max_color = std::max(cr, max_color);

		for (typename std::set<size_t>::iterator iter = edges[u].begin(); iter != edges[u].end(); iter++)
		{
			if (result[*iter] != -1)
				available[result[*iter]] = false;
		}
	}
	colored_vertices.resize(max_color + 1);
	for (size_t i = 0; i < num_vertices; i++)
	{
		colored_vertices[result[i]].push_back(i);
	}

}

void get_graph_component(const std::vector<std::set<size_t>>& edges, std::vector<std::set<size_t>>& components)
{
	//edges id starting from 0
	components.clear();
	std::vector<int> vert2color(edges.size(), -1);
	int cid = 0;
	int startid = -1;
	for (size_t i = 0; i < edges.size(); i++)
	{
		if (!edges[i].empty())
		{
			startid = i;
			break;
		}
	}
	assert(startid != -1);
	while (startid != -1)
	{
		std::set<size_t> onecluser;
		std::queue<size_t> q;
		q.push(startid);
		vert2color[startid] = cid;
		while (!q.empty())
		{
			size_t front = q.front();
			q.pop();
			onecluser.insert(front);
			for (auto v : edges[front])
			{
				if (vert2color[v] == -1)
				{
					q.push(v);
					vert2color[v] = cid;
				}
			}
		}
		components.push_back(onecluser);
		cid++;
		startid = -1;
		for (size_t i = 0; i < edges.size(); i++)
		{
			if (!edges[i].empty() && vert2color[i] == -1)
			{
				startid = i;
				break;
			}
		}
	}
}

void compute_vert_mean_curvature(const std::vector<TinyVector<double, 3>>& pos, const std::vector<TinyVector<size_t, 3>>& faces, std::vector<double>& curvature)
{
	using namespace Eigen;
	using namespace std;
	MatrixXd V;
	MatrixXi F;
	curvature.clear();
	curvature.resize(pos.size(), 0.0);
	//igl::readOFF(TUTORIAL_SHARED_PATH "/bumpy.off",V,F);
	//igl::readOFF("E:\\code\\PaperCode\\VolumeMeshProcessing\\x64\\Release\\AllModels\\fandisk\\fandisk.off", V, F);
	
	
	//remove dup verts
	std::vector<size_t> o2n(pos.size(), (size_t)-1), n2o;
	size_t count = 0;
	for (size_t i = 0; i < faces.size(); i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			size_t id = faces[i][j];
			if (o2n[id] == (size_t)-1)
			{
				o2n[id] = count;
				count = count + 1;
				n2o.push_back(id);
			}
		}
	}

	std::vector<std::vector<double>> vV(n2o.size(), std::vector<double>(3));
	std::vector<std::vector<int>> vF(faces.size(), std::vector<int>(3));
	for (size_t i = 0; i < n2o.size(); i++)
	{
		//V(3 *i, 1) = pos[i][0];
		/*V(3 * i + 1) = pos[i][1];
		V(3 * i + 2) = pos[i][2];*/
		for (size_t j = 0; j < 3; j++)
		{
			vV[i][j] = pos[n2o[i]][j];
		}
	}
	for (size_t i = 0; i < faces.size(); i++)
	{
		/*F(3 * i) = faces[i][0];
		F(3 * i + 1) = faces[i][1];
		F(3 * i + 2) = faces[i][2];*/
		for (size_t j = 0; j < 3; j++)
		{
			vF[i][j] = o2n[faces[i][j]];
		}
	}
	igl::list_to_matrix(vV, V);
	igl::list_to_matrix(vF, F);
	std::cout << "V size: " << V.size() << std::endl;
	std::cout << "F size: " << F.size() << std::endl;
	std::cout << "pos & faces size: " << pos.size() << " " << faces.size() << std::endl;


	SparseMatrix<double> M, Minv;
	MatrixXd HN;
	SparseMatrix<double> L;
	igl::cotmatrix(V, F, L);
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
	igl::invert_diag(M, Minv);
	HN = -Minv * (L * V);
	VectorXd H = HN.rowwise().norm(); //up to sign
	std::cout << "H min max: " << H.minCoeff() << " " << H.maxCoeff() << std::endl;
	for (size_t i = 0; i < n2o.size(); i++)
	{
		curvature[n2o[i]] = H[i];
	}
}

void update_tree_color(const std::vector<size_t>& new_color, TreeNode<size_t>* t)
{
	std::set<size_t> new_keys;
	for (auto v : t->keys)
	{
		new_keys.insert(new_color[v]);
	}
	t->keys = new_keys;
	for (size_t i = 0; i < t->children.size(); i++)
	{
		update_tree_color(new_color, t->children[i]);
	}

}



bool get_tree_from_convex_graph(const std::vector<std::set<size_t>> &graph , const std::map<std::pair<size_t, size_t>, int>& flag_fpconvex, bool flag_convex_bool, TreeNode<size_t>* tn, int layer, std::vector<size_t> &invalid_cluster)
{
	//set tn
	int target_convex = 2 - (int)flag_convex_bool;
	std::set<size_t> counter_nodes;
	for (size_t i = 0; i < graph.size(); i++)
	{
		for (auto v : graph[i])
		{
			if (v > i)
			{
				auto it = flag_fpconvex.find(std::pair<size_t, size_t>(i, v));
				assert(it != flag_fpconvex.end());
				if (it->second != 0 && it->second != target_convex)
				{
					counter_nodes.insert(i);
					counter_nodes.insert(v);
				}
			}
			
		}
	}

	if (!counter_nodes.empty())
	{
		std::vector<std::set<size_t>> counter_graph(graph.size()); //graph made of counter nodes
		for (auto cv : counter_nodes)
		{
			for (auto cvn : graph[cv])
			{
				if (std::find(counter_nodes.begin(), counter_nodes.end(), cvn) != counter_nodes.end())
				{
					auto it = flag_fpconvex.find(std::pair<size_t, size_t>(std::min(cv, cvn), std::max(cv, cvn)));
					assert(it != flag_fpconvex.end());
					if (it->second != target_convex) //including smooth one
						counter_graph[cv].insert(cvn);
				}
			}
		}
		std::vector<std::set<size_t>> counter_clusters;
		get_graph_component(counter_graph, counter_clusters);
		//set tn.children

		//set nodes that are not counter nodes to tn.keys
		for (size_t i = 0; i < graph.size(); i++)
		{
			if (!graph[i].empty() && std::find(counter_nodes.begin(), counter_nodes.end(), i) == counter_nodes.end())
			{
				tn->keys.insert(i);
			}
		}

		//one more criteria: cluster == 1 and no other component
		if (tn->keys.size() == 0 && counter_clusters.size() == 1)
		{
			//assume only one cluster is invalid
			std::cout << "tree construction failed" << std::endl;
			//exit(EXIT_FAILURE);
			invalid_cluster = std::vector<size_t>(counter_clusters[0].begin(), counter_clusters[0].end());
			return false;
		}

		for (size_t i = 0; i < counter_clusters.size(); i++)
		{
			std::vector<std::set<size_t>> subgraph(graph.size());
			for (auto v : counter_clusters[i])
			{
				for (auto vn : graph[v])
				{
					if (std::find(counter_clusters[i].begin(), counter_clusters[i].end(), vn) != counter_clusters[i].end())
					{
						subgraph[v].insert(vn);
					}
				}
			}
			TreeNode<size_t>* child = new TreeNode<size_t>;
			//if (layer == 10)
			//{
			//	std::cout << "Layers over 10!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			//	//exit(EXIT_FAILURE);
			//	return false;
			//}
			bool tmp_flag = get_tree_from_convex_graph(subgraph, flag_fpconvex, !flag_convex_bool, child, layer + 1, invalid_cluster);
			if (!tmp_flag)
			{
				return false;
			}
			tn->children.push_back(child);
		}
	}
	else
	{
		//set nodes that are not counter nodes to tn.keys
		for (size_t i = 0; i < graph.size(); i++)
		{
			if (!graph[i].empty() && std::find(counter_nodes.begin(), counter_nodes.end(), i) == counter_nodes.end())
			{
				tn->keys.insert(i);
			}
		}
	}

	
	return true;
}

using namespace MeshLib;
void get_mesh_vert_faces(Mesh3d& mesh, std::vector<std::array<double, 3>>& pos, std::vector<std::vector<size_t>>& faces)
{
	pos.clear();
	faces.clear();
	
	for (size_t i = 0; i < mesh.get_faces_list()->size(); i++)
	{
		HE_edge<double>* begin_edge = mesh.get_faces_list()->at(i)->edge;
		HE_edge<double>* edge = mesh.get_faces_list()->at(i)->edge;
		std::vector<size_t> oneface;
		do
		{
			//tri_verts[i][local_id++] = edge->pair->vert->id;
			oneface.push_back(edge->pair->vert->id);
			edge = edge->next;
		} while (edge != begin_edge);
		//mesh.get_faces_list()->at(fid)->normal;
		faces.push_back(oneface);
	}
	//std::vector<TinyVector<double, 3>> vert_pos;
	for (size_t i = 0; i < mesh.get_vertices_list()->size(); i++)
	{
		//vert_pos.push_back(mesh.get_vertices_list()->at(i)->pos);
		vec3d one_pt = mesh.get_vertices_list()->at(i)->pos;
		pos.push_back(std::array<double, 3>({ one_pt[0], one_pt[1], one_pt[2] }));
	}
	
}

int get_mesh_edge_type(Mesh3d& m, int heid) //0: smooth, 1: convex, 2: concave
{
	HE_edge<double>* e1 = m.get_edges_list()->at(heid);
	HE_edge<double>* e2 = e1->pair;
	if (e1->face == NULL || e2->face == NULL)
	{
		return 0;
	}

	double product = e1->face->normal.Dot(e2->next->vert->pos - e1->next->vert->pos);
	
	double tmp_cos = e1->face->normal.Dot(e2->face->normal);

	int res = 0;
	
	if (product < 0.0)
	{
		if (tmp_cos < th_smooth_cos_value)
			res = 1;
	}
	else
	{
		if (tmp_cos < th_smooth_cos_value)
		{
			res = 2;
		}
	}

	return res;
}

void get_all_tri_area(Mesh3d& mesh, std::vector<double>& tri_area)
{
	tri_area.clear();
	tri_area.resize(mesh.get_num_of_faces(), 0.0);
	std::vector<TinyVector<size_t, 3>> tri_verts(mesh.get_faces_list()->size());
	//std::vector<TinyVector<double, 3>> tri_normals;
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
		//tri_normals.push_back(mesh.get_faces_list()->at(i)->normal);
	}
	std::vector<TinyVector<double, 3>> vert_pos;
	for (size_t i = 0; i < mesh.get_vertices_list()->size(); i++)
	{
		vert_pos.push_back(mesh.get_vertices_list()->at(i)->pos);
	}

	for (size_t i = 0; i < tri_area.size(); i++)
	{
		//tri_area[i] = std::abs(compute_tri_area<double>(tri_verts[tri_faces[i][0]], tri_verts[tri_faces[i][1]], tri_verts[tri_faces[i][2]]));
		//total_tri_area += tri_area[i];
		tri_area[i] = std::abs(compute_tri_area<double>(vert_pos[tri_verts[i][0]], vert_pos[tri_verts[i][1]], vert_pos[tri_verts[i][2]]));
	}
	
}

bool dijkstra_mesh(Mesh3d* m, size_t start_vert, std::vector<size_t>& prev_map, std::vector<double>& dist)
{
	prev_map.clear();
	dist.clear();
	dist.resize(m->get_num_of_vertices(), DBL_MAX);
	prev_map.resize(m->get_num_of_vertices(), 0);
	dist[start_vert] = 0.0;
	std::set<std::pair<double, HE_vert<double>*>> vqueue;
	for (size_t i = 0; i < m->get_num_of_vertices(); i++)
	{
		HE_vert<double>* hv = m->get_vertex(i);
		vqueue.insert(std::pair<double, HE_vert<double>*>(dist[hv->id], hv));
	}

	while (!vqueue.empty())
	{
		std::set<std::pair<double, HE_vert<double>*>>::iterator iter = vqueue.begin();
		HE_vert<double>* u = iter->second;
		vqueue.erase(iter);
		if (u->edge == NULL)
			continue;

		HE_edge<double>* he = u->edge;
		do
		{
			double alt = dist[u->id] + (u->pos - he->vert->pos).Length();
			if (alt < dist[he->vert->id])
			{
				vqueue.erase(std::pair<double, HE_vert<double>*>(dist[he->vert->id], he->vert));
				dist[he->vert->id] = alt;
				prev_map[he->vert->id] = u->id;
				vqueue.insert(std::pair<double, HE_vert<double>*>(dist[he->vert->id], he->vert));
			}
			he = he->pair->next;
		} while (he != u->edge);
	}

	return true;
}

void get_he_list_dijkstra(Mesh3d& m, size_t start_vert, size_t end_vert, const std::vector<size_t>& prev_map, std::vector<size_t>& he_list)
{
	//get he list for path start_vert -> end_vert
	he_list.clear();
	size_t cur = end_vert;
	while (cur != start_vert)
	{
		size_t prev = prev_map[cur];
		HE_edge<double>* start_edge = m.get_vertex(cur)->edge;
		HE_edge<double>* edge = m.get_vertex(cur)->edge;
		do
		{
			if (edge->vert->id == prev)
			{
				he_list.push_back(edge->id);
				he_list.push_back(edge->pair->id);
				break;
			}

			edge = edge->pair->next;
		} while (edge != start_edge);

		cur = prev;
	}
}

void repair_tree_features(Mesh3d& m, const std::vector<int>& face_color, const std::vector<size_t>& invalid_colors, std::vector<std::pair<int, int>>& ungrouped_features)
{
	//get sub mesh firstly
	std::vector<bool> invalid_color_map(face_color.size(), false);
	for (auto cid : invalid_colors)
		//invalid_color_set.insert(cid);
		invalid_color_map[cid] = true;

	std::vector<std::array<double, 3>> all_pts;
	std::vector<std::vector<size_t>> all_faces, invalid_faces;
	get_mesh_vert_faces(m, all_pts, all_faces);
	for (size_t i = 0; i < face_color.size(); i++)
	{
		if (invalid_color_map[face_color[i] - 1] == true)
		{
			invalid_faces.push_back(all_faces[i]);
		}
	}

	Mesh3d sub_mesh;
	sub_mesh.load_mesh(all_pts, invalid_faces);
	sub_mesh.write_obj("invalid.obj");

	//to change: ungrouped features
	//step1: load feature, step2: select convex-concavefeature, step3: add shorted path to other corner points or boundary
	
	std::vector<std::vector<int>> feature_v2he(sub_mesh.get_num_of_vertices()); //id of hes ematating from each vertex
	
	std::vector<std::pair<int, int>> ungrouped_features_new;
	std::vector<bool> flag_he_feature(sub_mesh.get_num_of_edges(), false);
	for (size_t i = 0; i < ungrouped_features.size(); i++)
	{
		int id0 = ungrouped_features[i].first;
		int id1 = ungrouped_features[i].second;
		//feature_degree_v[id0]++;
		//feature_degree_v[id1]++;

		HE_edge<double>* begin_edge = sub_mesh.get_vertices_list()->at(id0)->edge;
		HE_edge<double>* edge = sub_mesh.get_vertices_list()->at(id0)->edge;
		bool flag_found = false;
		//if (edge != NULL && !sub_mesh.is_on_boundary(edge)) //not boundary
		if (edge != NULL)
		{
			do
			{
				if (id1 == edge->vert->id)
				{
					feature_v2he[id0].push_back(edge->id);
					feature_v2he[id1].push_back(edge->pair->id);
					flag_he_feature[edge->id] = true;
					flag_he_feature[edge->pair->id] = true;
					flag_found = true;
					break;
				}
				edge = edge->pair->next;
			} while (edge != begin_edge);
		}
		//assert(flag_found == true);
		if (flag_found == true)
		{
			ungrouped_features_new.push_back(ungrouped_features[i]);
		}
	}

	//for debugging: visualize flag_he_feature
	//auto flag_he_feature_debug = flag_he_feature;
	//std::vector<std::pair<int, int>> debug_feas;
	//std::ofstream ofs("debug.fea");
	//for (size_t i = 0; i < flag_he_feature.size(); i++)
	//{
	//	if (flag_he_feature_debug[i])
	//	{
	//		flag_he_feature_debug[sub_mesh.get_edge(i)->pair->id] = false;
	//		//ofs << sub_mesh.get_edge(i)->vert->id << " " << sub_mesh.get_edge(i)->pair->vert->id << std::endl;
	//		debug_feas.push_back(std::pair<int, int>(sub_mesh.get_edge(i)->vert->id, sub_mesh.get_edge(i)->pair->vert->id));
	//	}
	//}

	//ofs << debug_feas.size() << std::endl;
	//for (auto& pp : debug_feas)
	//{
	//	ofs << pp.first << " " << pp.second << std::endl;
	//}

	//ofs.close();

	//int max_v_degree = 0;
	std::vector<int> feature_vtype(sub_mesh.get_num_of_vertices(), 0); //0: normal case, 1: corner, 2: turn vertex
	std::vector<int> feature_vcolor(sub_mesh.get_num_of_vertices(), false);

	for (size_t i = 0; i < feature_v2he.size(); i++)
	{
		/*if (max_v_degree < feature_v2he[i].size())
			max_v_degree = feature_v2he[i].size();*/
		if (feature_v2he[i].size() == 1)
		{
			feature_vtype[i] = 1;
			//feature_vcolor[i] = true;
		}
		else if (feature_v2he[i].size() >= 3)
		{
			//normal or turn
			int convex_count = 0, concave_count = 0;
			for (auto heid : feature_v2he[i])
			{
				int hetype = get_mesh_edge_type(sub_mesh, heid);
				if (hetype == 1)
					convex_count += 1;
				if (hetype == 2)
					concave_count += 1;
			}

			if (convex_count > 0 && concave_count > 0)
			{
				feature_vtype[i] = 2; //turn
				//std::cout << "turn vertex id: " << i << std::endl;
			}
			else
			{
				feature_vtype[i] = 1;
			}
		}
	}

	std::set<int> ep_cand; //corner or boundary
	for (size_t i = 0; i < feature_v2he.size(); i++)
	{
		if (feature_vtype[i] > 0)
			ep_cand.insert(i);
	}

	/*for (size_t i = 0; i < sub_mesh.get_num_of_vertices(); i++)
	{
		if (sub_mesh.is_on_boundary(sub_mesh.get_vertex(i)))
		{
			ep_cand.insert(i);
		}
	}*/
	
	for (size_t i = 0; i < sub_mesh.get_num_of_edges(); i++)
	{
		if (sub_mesh.is_on_boundary(sub_mesh.get_edge(i)))
		{
			ep_cand.insert(sub_mesh.get_edge(i)->vert->id);
			ep_cand.insert(sub_mesh.get_edge(i)->pair->vert->id);
		}
	}

	//add features for each turn vertex
	//std::vector<bool> flag_he_feature_add = flag_he_feature;
	std::vector<bool> flag_he_feature_add(flag_he_feature.size(), false);

	for (size_t i = 0; i < feature_v2he.size(); i++)
	{
		if (feature_vtype[i] == 2 && !feature_vcolor[i])
		{
			std::vector<size_t> prev_map;
			std::vector<double> dist;
			dijkstra_mesh(&sub_mesh, i, prev_map, dist);

			int target = -1;
			std::vector<std::pair<double, int>> dist_ep;
			for (auto ep : ep_cand)
			{
				if (ep != i)
					dist_ep.push_back(std::pair<double, int>(dist[ep], ep));
			}

			std::sort(dist_ep.begin(), dist_ep.end());

			for (size_t j = 0; j < dist_ep.size(); j++)
			{
				int cur_vert = dist_ep[j].second;
				bool path_valid = false; //if one of the edge in the path is not feature, set as true
				
				std::vector<size_t> he_list;
				get_he_list_dijkstra(sub_mesh, i, cur_vert, prev_map, he_list);
				for (auto he : he_list)
				{
					if (flag_he_feature[he] == false)
						path_valid = true;
				}

				if (path_valid)
				{
					//update flag_he_feature_add
					for (auto he : he_list)
					{
						flag_he_feature_add[he] = true;
					}
					target = cur_vert;
					break;
				}
			}
			assert(target != -1);
			feature_vcolor[i] = true;
			//target also set as true
			feature_vcolor[target] = true;
		}
	}
	
	//update ungrouped_features
	std::set<std::pair<int, int>> ungrouped_features_set;
	for (auto& pp : ungrouped_features)
	{
		ungrouped_features_set.insert(std::pair<int, int>(std::min(pp.first, pp.second), std::max(pp.first, pp.second)));
	}

	

	//ungrouped_features.clear();
	for (size_t i = 0; i < flag_he_feature_add.size(); i++)
	{
		if (flag_he_feature_add[i])
		{
			//set opposite as true
			flag_he_feature_add[sub_mesh.get_edge(i)->pair->id] = false;
			int v1 = sub_mesh.get_edge(i)->vert->id, v2 = sub_mesh.get_edge(i)->pair->vert->id;
			ungrouped_features_set.insert(std::pair<int, int>(std::min(v1, v2), std::max(v1, v2)));
			//ungrouped_features.push_back(std::pair<int, int>(sub_mesh.get_edge(i)->vert->id, sub_mesh.get_edge(i)->pair->vert->id));
		}
	}

	//ungrouped_features.clear();
	ungrouped_features = std::vector<std::pair<int, int>>(ungrouped_features_set.begin(), ungrouped_features_set.end());
}

void repair_tree_features_maxflow(Mesh3d& m, const std::vector<int>& face_color, const std::vector<size_t>& invalid_colors, std::vector<std::pair<int, int>>& ungrouped_features)
{
	//bool flag_debug = true;
	bool flag_debug = false;
	std::set<std::pair<int, int>> ungrouped_features_set;
	for (auto& pp : ungrouped_features)
	{
		ungrouped_features_set.insert(std::pair<int, int>(std::min(pp.first, pp.second), std::max(pp.first, pp.second)));
	}

	std::vector<std::array<double, 3>> all_pts;
	std::vector<std::vector<size_t>> all_faces, all_invalid_faces;
	get_mesh_vert_faces(m, all_pts, all_faces);
	std::vector<bool> invalid_color_map(face_color.size(), false);
	for (auto cid : invalid_colors)
		//invalid_color_set.insert(cid);
		invalid_color_map[cid] = true;

	for (size_t i = 0; i < face_color.size(); i++)
	{
		if (invalid_color_map[face_color[i] - 1])
		{
			all_invalid_faces.push_back(all_faces[i]);
		}
	}
	
	Mesh3d invalid_mesh;
	invalid_mesh.load_mesh(all_pts, all_invalid_faces);

	if (flag_debug)
	{
		invalid_mesh.write_obj("invalid_all.obj");
	}

	std::map<std::pair<int, int>, int> edgepair2type;

	for (size_t i = 0; i < ungrouped_features.size(); i++)
	{
		int id0 = ungrouped_features[i].first;
		int id1 = ungrouped_features[i].second;

		HE_edge<double>* begin_edge = invalid_mesh.get_vertices_list()->at(id0)->edge;
		HE_edge<double>* edge = invalid_mesh.get_vertices_list()->at(id0)->edge;
		//if (edge != NULL && !invalid_mesh.is_on_boundary(edge)) //not boundary
		if (edge != NULL)
		{
			do
			{
				if (id1 == edge->vert->id)
				{
					int edge_type = get_mesh_edge_type(invalid_mesh, edge->id); //if boundary, type is 0
					edgepair2type[ungrouped_features[i]] = edge_type;
					break;
				}
				edge = edge->pair->next;
			} while (edge != begin_edge);
		}
	}
	

	bool flag_patchbypatch = true;

	//flip edge
	std::vector<std::pair<int, int>> flipped_edges;
	std::vector<int> flipped_corner;

	//repair patch by patch
	if (flag_patchbypatch)
	{
		for (auto cid : invalid_colors)
		{
			std::vector<std::vector<size_t>> cur_faces;
			for (size_t i = 0; i < face_color.size(); i++)
			{
				if (face_color[i] - 1 == cid)
				{
					cur_faces.push_back(all_faces[i]);
				}
			}
			Mesh3d sub_mesh;
			sub_mesh.load_mesh(all_pts, cur_faces);
			if (flag_debug)
				sub_mesh.write_obj("curpatch.obj");
			//submesh might need to be adjusted
			std::vector<bool> flag_he_feature(sub_mesh.get_num_of_edges(), false);
			std::vector<int> flag_he_type(sub_mesh.get_num_of_edges(), 0);
			for (size_t i = 0; i < ungrouped_features.size(); i++)
			{
				int id0 = ungrouped_features[i].first;
				int id1 = ungrouped_features[i].second;

				HE_edge<double>* begin_edge = sub_mesh.get_vertices_list()->at(id0)->edge;
				HE_edge<double>* edge = sub_mesh.get_vertices_list()->at(id0)->edge;
				bool flag_found = false;
				//if (edge != NULL && !sub_mesh.is_on_boundary(edge)) //not boundary
				if (edge != NULL)
				{
					do
					{
						if (id1 == edge->vert->id)
						{
							flag_he_feature[edge->id] = true;
							flag_he_feature[edge->pair->id] = true;
							//int edge_type = get_mesh_edge_type(sub_mesh, edge->id);
							int edge_type = edgepair2type[ungrouped_features[i]];
							flag_he_type[edge->id] = edge_type;
							flag_he_type[edge->pair->id] = edge_type;
							flag_found = true;
							break;
						}
						edge = edge->pair->next;
					} while (edge != begin_edge);
				}
			}

			//check mesh validness: a triangle should not have two nb edge with different convexity
			bool flag_mesh_valid = true;
			for (size_t i = 0; i < sub_mesh.get_num_of_faces(); i++)
			{
				if (sub_mesh.is_on_boundary(sub_mesh.get_face(i)))
				{
					int convex_count = 0, concave_count = 0;
					HE_edge<double>* edge = sub_mesh.get_face(i)->edge;
					do
					{
						if (flag_he_feature[edge->id])
						{
							if (flag_he_type[edge->id] == 1)
							{
								convex_count += 1;
							}
							else if (flag_he_type[edge->id] == 2)
							{
								concave_count += 1;
							}
						}
						edge = edge->next;
					} while (edge != sub_mesh.get_face(i)->edge);


					if (convex_count > 0 && concave_count > 0)
					{
						flag_mesh_valid = false;
						//edge flip
						edge = sub_mesh.get_face(i)->edge; //find no feature curve
						do
						{
							if (!flag_he_feature[edge->id])
							{
								break;
							}
							edge = edge->next;
						} while (edge != sub_mesh.get_face(i)->edge);

						if (sub_mesh.is_on_boundary(edge))
						{
							std::cout << "all 3 edge of triangle " << i << " are features" << std::endl;
						}
						else
						{
							int v0 = edge->vert->id, v1 = edge->pair->vert->id;
							flipped_edges.push_back(std::pair<int, int>(std::min(v0, v1), std::max(v0, v1)));
							flipped_corner.push_back(edge->next->vert->id);

							int oppo_face_id = edge->pair->face->id;
							std::vector<size_t> newf1, newf2;
							//edge flip
							newf1.push_back(edge->vert->id);
							newf1.push_back(edge->next->vert->id);
							newf1.push_back(edge->pair->next->vert->id);
							edge = edge->pair;
							newf2.push_back(edge->vert->id);
							newf2.push_back(edge->next->vert->id);
							newf2.push_back(edge->pair->next->vert->id);

							cur_faces[i] = newf1;
							cur_faces[oppo_face_id] = newf2;
						}
					}

				}
			}

			if (!flag_mesh_valid)
			{
				std::cout << "input mesh for patch " << cid << " is invalid" << std::endl;
				/*Mesh3d new_mesh;
				new_mesh.load_mesh(all_pts, cur_faces);
				std::swap(sub_mesh, new_mesh);*/

				sub_mesh.load_mesh(all_pts, cur_faces);
				if (flag_debug)
					sub_mesh.write_obj("invalidpatch_repair.obj");
				//submesh might need to be adjusted

				flag_he_feature.clear();
				flag_he_type.clear();
				flag_he_feature.resize(sub_mesh.get_num_of_edges(), false);
				flag_he_type.resize(sub_mesh.get_num_of_edges(), 0);
				for (size_t i = 0; i < ungrouped_features.size(); i++)
				{
					int id0 = ungrouped_features[i].first;
					int id1 = ungrouped_features[i].second;
					//feature_degree_v[id0]++;
					//feature_degree_v[id1]++;

					HE_edge<double>* begin_edge = sub_mesh.get_vertices_list()->at(id0)->edge;
					HE_edge<double>* edge = sub_mesh.get_vertices_list()->at(id0)->edge;
					bool flag_found = false;
					//if (edge != NULL && !sub_mesh.is_on_boundary(edge)) //not boundary
					if (edge != NULL)
					{
						do
						{
							if (id1 == edge->vert->id)
							{
								flag_he_feature[edge->id] = true;
								flag_he_feature[edge->pair->id] = true;
								//int edge_type = get_mesh_edge_type(sub_mesh, edge->id);
								int edge_type = edgepair2type[ungrouped_features[i]];
								flag_he_type[edge->id] = edge_type;
								flag_he_type[edge->pair->id] = edge_type;
								flag_found = true;
								break;
							}
							edge = edge->pair->next;
						} while (edge != begin_edge);
					}
				}
			}

			//mesh constructed, build max cut algorithm
			int n_node = sub_mesh.get_num_of_faces() + 2;
			std::set<int> convex_node, concave_node;
			for (size_t i = 0; i < sub_mesh.get_num_of_edges(); i++)
			{
				if (!flag_he_feature[i]) continue;
				if (flag_he_type[i] == 1)
				{
					//convex
					if (sub_mesh.get_edge(i)->face != NULL)
					{
						convex_node.insert(sub_mesh.get_edge(i)->face->id);
					}
				}
				else if (flag_he_type[i] == 2)
				{
					//concave
					if (sub_mesh.get_edge(i)->face != NULL)
					{
						concave_node.insert(sub_mesh.get_edge(i)->face->id);
					}
				}
			}

			assert(!convex_node.empty() && !concave_node.empty());

			if (!convex_node.empty() && !concave_node.empty())
			{
				int s = sub_mesh.get_num_of_faces();
				int t = s + 1;
				double max_cap = 1.0 * sub_mesh.get_num_of_faces();
				std::vector<std::pair<int, int>> graph_edges;
				std::vector<double> graph_edgecap;


				//s to convex
				for (auto node : convex_node)
				{
					graph_edges.push_back(std::pair<int, int>(s, node));
					graph_edgecap.push_back(max_cap);
				}

				//concave to t
				for (auto node : concave_node)
				{
					graph_edges.push_back(std::pair<int, int>(node, t));
					graph_edgecap.push_back(max_cap);
				}

				if (flag_debug)
				{
					std::vector<std::vector<size_t>> convex_faces, concave_faces;
					for (auto node : convex_node)
					{
						convex_faces.push_back(cur_faces[node]);
					}
					for (auto node : concave_node)
					{
						concave_faces.push_back(cur_faces[node]);
					}
					
					Mesh3d convexmesh, concavemesh;
					convexmesh.load_mesh(all_pts, convex_faces);
					concavemesh.load_mesh(all_pts, concave_faces);
					convexmesh.write_obj("convex.obj");
					concavemesh.write_obj("concave.obj");

				}

				//other parts
				for (size_t i = 0; i < sub_mesh.get_num_of_edges(); i++)
				{
					if (!sub_mesh.is_on_boundary(sub_mesh.get_edge(i)) && flag_he_type[i] == 0)
					{
						//not boundary, not convex or concave
						graph_edges.push_back(std::pair<int, int>(sub_mesh.get_edge(i)->face->id, sub_mesh.get_edge(i)->pair->face->id));
						graph_edgecap.push_back(1.0);
					}
				}

				std::vector<int> part1, part2;
				MinCutfromMaxFlow(n_node, graph_edges, graph_edgecap, s, t, part1, part2);

				//for debugging:
				//visualze part1 and part2
				if (flag_debug)
				{
					Mesh3d mesh_part1, mesh_part2;
					std::vector<std::vector<size_t>> part1_faces, part2_faces;
					for (auto f : part1)
					{
						part1_faces.push_back(cur_faces[f]);
					}

					for (auto f : part2)
					{
						part2_faces.push_back(cur_faces[f]);
					}
					mesh_part1.load_mesh(all_pts, part1_faces);
					mesh_part2.load_mesh(all_pts, part2_faces);
					mesh_part1.write_obj("part1.obj");
					mesh_part2.write_obj("part2.obj");
				}
				
				
				std::vector<bool> flag_part1(n_node - 2, false), flag_part2(n_node - 2, false);
				for (auto f : part1)
					flag_part1[f] = true;

				for (auto f : part2)
					flag_part2[f] = true;

				for (size_t i = 0; i < sub_mesh.get_num_of_edges(); i++)
				{
					auto edge = sub_mesh.get_edge(i);
					if (!sub_mesh.is_on_boundary(edge))
					{
						int f0 = edge->face->id, f1 = edge->pair->face->id;
						if (flag_part1[f0] && flag_part2[f1])
						{
							int id0 = edge->vert->id, id1 = edge->pair->vert->id;
							ungrouped_features_set.insert(std::pair<int, int>(std::min(id0, id1), std::max(id0, id1)));
						}
					}
				}
			}

		}
	}
	else
	{
	//for (auto cid : invalid_colors)
	if (true)
	{
		
		std::vector<std::vector<size_t>> cur_faces;
		for (size_t i = 0; i < face_color.size(); i++)
		{
			if (invalid_color_map[face_color[i] - 1])
			{
				cur_faces.push_back(all_faces[i]);
			}
		}
		Mesh3d sub_mesh;
		sub_mesh.load_mesh(all_pts, cur_faces);
		sub_mesh.write_obj("invalidpatch.obj");

		//submesh might need to be adjusted
		std::vector<bool> flag_he_feature(sub_mesh.get_num_of_edges(), false);
		std::vector<int> flag_he_type(sub_mesh.get_num_of_edges(), 0);
		for (size_t i = 0; i < ungrouped_features.size(); i++)
		{
			int id0 = ungrouped_features[i].first;
			int id1 = ungrouped_features[i].second;

			HE_edge<double>* begin_edge = sub_mesh.get_vertices_list()->at(id0)->edge;
			HE_edge<double>* edge = sub_mesh.get_vertices_list()->at(id0)->edge;
			bool flag_found = false;
			if (edge != NULL && !sub_mesh.is_on_boundary(edge)) //not boundary
			//if (edge != NULL)
			{
				do
				{
					if (id1 == edge->vert->id)
					{
						flag_he_feature[edge->id] = true;
						flag_he_feature[edge->pair->id] = true;
						int edge_type = get_mesh_edge_type(sub_mesh, edge->id);
						flag_he_type[edge->id] = edge_type;
						flag_he_type[edge->pair->id] = edge_type;
						flag_found = true;
						break;
					}
					edge = edge->pair->next;
				} while (edge != begin_edge);
			}
		}

		//check mesh validness: a triangle should not have two nb edge with different convexity
		bool flag_mesh_valid = true;
		for (size_t i = 0; i < sub_mesh.get_num_of_faces(); i++)
		{
			if (sub_mesh.is_on_boundary(sub_mesh.get_face(i)))
			{
				int convex_count = 0, concave_count = 0;
				HE_edge<double>* edge = sub_mesh.get_face(i)->edge;
				do
				{
					if (flag_he_feature[edge->id])
					{
						if (flag_he_type[edge->id] == 1)
						{
							convex_count += 1;
						}
						else if (flag_he_type[edge->id] == 2)
						{
							concave_count += 1;
						}
					}
					edge = edge->next;
				} while (edge != sub_mesh.get_face(i)->edge);


				if (convex_count > 0 && concave_count > 0)
				{
					flag_mesh_valid = false;
					//edge flip
					edge = sub_mesh.get_face(i)->edge; //find no feature curve
					do
					{
						if (!flag_he_feature[edge->id])
						{
							break;
						}
					} while (edge != sub_mesh.get_face(i)->edge);

					if (sub_mesh.is_on_boundary(edge))
					{
						std::cout << "all 3 edge of triangle " << i << " are features" << std::endl;
						int a = 0;
					}
					else
					{
						int oppo_face_id = edge->pair->face->id;
						std::vector<size_t> newf1, newf2;
						//edge flip
						newf1.push_back(edge->vert->id);
						newf1.push_back(edge->next->vert->id);
						newf1.push_back(edge->pair->next->id);

						edge = edge->pair;
						newf2.push_back(edge->vert->id);
						newf2.push_back(edge->next->vert->id);
						newf2.push_back(edge->pair->next->id);

						cur_faces[i] = newf1;
						cur_faces[oppo_face_id] = newf2;
					}
				}

			}
		}

		if (!flag_mesh_valid)
		{
			//std::cout << "input mesh for patch " << cid << " is invalid" << std::endl;
			sub_mesh.load_mesh(all_pts, cur_faces);
			sub_mesh.write_obj("invalidpatch_repair.obj");
			//submesh might need to be adjusted
			std::vector<bool> flag_he_feature(sub_mesh.get_num_of_edges(), false);
			std::vector<int> flag_he_type(sub_mesh.get_num_of_edges(), 0);

			flag_he_feature.clear();
			flag_he_type.clear();
			flag_he_feature.resize(sub_mesh.get_num_of_edges(), false);
			flag_he_type.resize(sub_mesh.get_num_of_edges(), 0);
			for (size_t i = 0; i < ungrouped_features.size(); i++)
			{
				int id0 = ungrouped_features[i].first;
				int id1 = ungrouped_features[i].second;
				//feature_degree_v[id0]++;
				//feature_degree_v[id1]++;

				HE_edge<double>* begin_edge = sub_mesh.get_vertices_list()->at(id0)->edge;
				HE_edge<double>* edge = sub_mesh.get_vertices_list()->at(id0)->edge;
				bool flag_found = false;
				if (edge != NULL && !sub_mesh.is_on_boundary(edge)) //not boundary
				//if (edge != NULL)
				{
					do
					{
						if (id1 == edge->vert->id)
						{
							flag_he_feature[edge->id] = true;
							flag_he_feature[edge->pair->id] = true;
							int edge_type = get_mesh_edge_type(sub_mesh, edge->id);
							flag_he_type[edge->id] = edge_type;
							flag_he_type[edge->pair->id] = edge_type;
							flag_found = true;
							break;
						}
						edge = edge->pair->next;
					} while (edge != begin_edge);
				}
			}
		}

		//mesh constructed, build max cut algorithm
		int n_node = sub_mesh.get_num_of_faces() + 2;
		std::set<int> convex_node, concave_node;
		for (size_t i = 0; i < sub_mesh.get_num_of_edges(); i++)
		{
			if (flag_he_type[i] == 1)
			{
				//convex
				if (sub_mesh.get_edge(i)->face != NULL)
				{
					convex_node.insert(sub_mesh.get_edge(i)->face->id);
				}
			}
			else if (flag_he_type[i] == 2)
			{
				//concave
				if (sub_mesh.get_edge(i)->face != NULL)
				{
					concave_node.insert(sub_mesh.get_edge(i)->face->id);
				}
			}
		}

		assert(!convex_node.empty() && !concave_node.empty());

		if (!convex_node.empty() && !concave_node.empty())
		{
			int s = sub_mesh.get_num_of_faces();
			int t = s + 1;
			double max_cap = 1.0 * sub_mesh.get_num_of_faces();
			std::vector<std::pair<int, int>> graph_edges;
			std::vector<double> graph_edgecap;


			//s to convex
			for (auto node : convex_node)
			{
				graph_edges.push_back(std::pair<int, int>(s, node));
				graph_edgecap.push_back(max_cap);
			}

			//concave to t
			for (auto node : concave_node)
			{
				graph_edges.push_back(std::pair<int, int>(node, t));
				graph_edgecap.push_back(max_cap);
			}

			//other parts
			for (size_t i = 0; i < sub_mesh.get_num_of_edges(); i++)
			{
				if (!sub_mesh.is_on_boundary(sub_mesh.get_edge(i)) && flag_he_type[i] == 0)
				{
					//not boundary, not convex or concave
					graph_edges.push_back(std::pair<int, int>(sub_mesh.get_edge(i)->face->id, sub_mesh.get_edge(i)->pair->face->id));
					graph_edgecap.push_back(1.0);
				}
			}

			std::vector<int> part1, part2; 
			MinCutfromMaxFlow(n_node, graph_edges, graph_edgecap, s, t, part1, part2);

			//get min cut by neighbor info
			std::vector<bool> flag_part1(n_node - 2, false), flag_part2(n_node - 2, false);
			for (auto f : part1)
				flag_part1[f] = true;

			for (auto f : part2)
				flag_part2[f] = true;

			for (size_t i = 0; i < sub_mesh.get_num_of_edges(); i++)
			{
				auto edge = sub_mesh.get_edge(i);
				if (!sub_mesh.is_on_boundary(edge))
				{
					int f0 = edge->face->id, f1 = edge->pair->face->id;
					if (flag_part1[f0] && flag_part2[f1])
					{
						int id0 = edge->vert->id, id1 = edge->pair->vert->id;
						ungrouped_features_set.insert(std::pair<int, int>(std::min(id0, id1), std::max(id0, id1)));
					}
				}
			}

			//visualze part1 and part2
			std::vector<std::vector<size_t>> part1_faces, part2_faces;
			for (auto f : part1)
			{
				part1_faces.push_back(cur_faces[f]);
			}

			for (auto f : part2)
			{
				part2_faces.push_back(cur_faces[f]);
			}

			Mesh3d mesh_part1, mesh_part2;
			mesh_part1.load_mesh(all_pts, part1_faces);
			mesh_part2.load_mesh(all_pts, part2_faces);
			mesh_part1.write_obj("part1.obj");
			mesh_part2.write_obj("part2.obj");

		}

	}
	}
	
	
	ungrouped_features = std::vector<std::pair<int, int>>(ungrouped_features_set.begin(), ungrouped_features_set.end());

	if (!flipped_edges.empty())
	{
		//update all faces
		//for (auto& pp : flipped_edges)
		for (size_t i = 0; i < flipped_edges.size(); i++)
		{
			auto pp = flipped_edges[i];
			int id0 = pp.first;
			int id1 = pp.second;
			auto begin_edge = m.get_vertices_list()->at(id0)->edge;
			auto edge = m.get_vertices_list()->at(id0)->edge;
			do
			{
				if (id1 == edge->vert->id)
				{
					if (edge->next->vert->id != flipped_corner[i])
						edge = edge->pair;

					assert(edge->next->vert->id == flipped_corner[i]);
					int cur_face_id = edge->face->id;
					int oppo_face_id = edge->pair->face->id;
					std::vector<size_t> newf1, newf2;
					//edge flip
					newf1.push_back(edge->vert->id);
					newf1.push_back(edge->next->vert->id);
					newf1.push_back(edge->pair->next->vert->id);
					edge = edge->pair;
					newf2.push_back(edge->vert->id);
					newf2.push_back(edge->next->vert->id);
					newf2.push_back(edge->pair->next->vert->id);

					all_faces[cur_face_id] = newf1;
					all_faces[oppo_face_id] = newf2;

					break;
				}
				//edge = edge->next;
				edge = edge->pair->next;
			} while (edge != begin_edge);
		}

		m.load_mesh(all_pts, all_faces);
		
	}

	return;
}

bool load_xyz_file(const char* filename, std::vector<vec3d>& pts, std::vector<vec3d>& normals)
{
	//assume normal is given
	int number_of_lines = 0;
	std::string line;
	std::ifstream myfile(filename);

	while (std::getline(myfile, line))
		++number_of_lines;
	//return 0;
	myfile.close();

	pts.clear();
	normals.clear();
	pts.resize(number_of_lines);
	normals.resize(number_of_lines);
	myfile.open(filename);

	for (size_t i = 0; i < number_of_lines; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			myfile >> pts[i][j];
		}
		
		for (size_t j = 0; j < 3; j++)
		{
			myfile >> normals[i][j];
		}
	}
	myfile.close();
	return true;
}

TinyVector<double, 3> perturb_normal(const TinyVector<double, 3> normal, double angle_noise1, double angle_noise2)
{
	//get angle from normal
	//assume normal is normalized
	double theta = acos(normal[2]);
	double phi = acos(normal[0] / (sqrt(1 - normal[2] * normal[2]) + DENOMINATOR_EPS));
	double phi_ref = asin(normal[1] / (sqrt(1 - normal[2] * normal[2]) + DENOMINATOR_EPS));
	if (phi_ref < 0)
		phi = 2 * M_PI - phi;
	theta = theta + angle_noise1 / 180.0 * M_PI;
	phi = phi + angle_noise2 / 180.0 * M_PI;
	TinyVector<double, 3> normal_new;
	normal_new[0] = sin(theta) * cos(phi);
	normal_new[1] = sin(theta) * sin(phi);
	normal_new[2] = cos(theta);
	return normal_new;
}

#include <random>
bool sample_pts_from_mesh(const std::vector<TinyVector<double, 3>>& tri_verts, const std::vector<TinyVector<size_t, 3>>& tri_faces, const std::vector<TinyVector<double, 3>>& tri_normals, const std::vector<int>& tri_face_masks, int n_sample, std::vector<TinyVector<double, 3>>& output_pts, std::vector<TinyVector<double, 3>>& output_normals, std::vector<int>& output_masks, double sigma, double sigma_n)
{
	if (tri_faces.empty())
	{
		std::cout << "!!!no face input for sampling" << std::endl;
		return false;
	}
	output_pts.clear();
	output_normals.clear();
	output_masks.clear();
	bool flag_pos_noise = (sigma > 0.0);
	bool flag_normal_noise = (sigma_n > 0.0);
	//noise to be set
	if (!flag_pos_noise) sigma = 0.0;
	if (!flag_normal_noise) sigma_n = 0.0;
	//feature parts first
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<double> unif_dist(0, 1);
	//std::normal_distribution<double> normal_dist(0, sigma);
	std::uniform_real_distribution<double> normal_dist(-sigma, sigma);
	std::uniform_real_distribution<double> angle_unif_dist(-sigma_n, sigma_n);

	//compute triangles
	std::vector<double> tri_area(tri_faces.size(), 0.0);
	double total_tri_area = 0.0;
	for (size_t i = 0; i < tri_area.size(); i++)
	{
		tri_area[i] = std::abs(compute_tri_area<double>(tri_verts[tri_faces[i][0]], tri_verts[tri_faces[i][1]], tri_verts[tri_faces[i][2]]));
		total_tri_area += tri_area[i];
	}
	
	std::vector<double> tri_bound(tri_area.size() + 1, 0.0);
	for (size_t i = 0; i < tri_area.size(); i++)
	{
		tri_bound[i + 1] = tri_bound[i] + tri_area[i] / total_tri_area;
	}
	
	for (size_t i = 0; i < n_sample; i++)
	{
		double u = unif_dist(e2);
		auto iter = std::upper_bound(tri_bound.begin(), tri_bound.end(), u);
		int fid = (int)std::distance(tri_bound.begin(), iter);
		//assert(fid != tri_verts.size() + 1);
		fid = std::max(0, fid - 1);
		fid = std::min(fid, (int)tri_faces.size() - 1);
		//sample
		//int id0 = ungrouped_features[fid].first;
		//int id1 = ungrouped_features[fid].second;
		double s = unif_dist(e2);
		double t = unif_dist(e2);
		if (s + t > 1)
		{
			s = 1 - s;
			t = 1 - t;
		}
		TinyVector<double, 3> facenormal = tri_normals[fid];
		/*if (result.count("s"))
			sample_pts.push_back((1.0 - s - t) * vert_pos[tri_verts[fid][0]] + s * vert_pos[tri_verts[fid][1]] + t * vert_pos[tri_verts[fid][2]] + facenormal * normal_dist(e2));
		else*/
		if (flag_pos_noise)
		{
			output_pts.push_back((1.0 - s - t) * tri_verts[tri_faces[fid][0]] + s * tri_verts[tri_faces[fid][1]] + t * tri_verts[tri_faces[fid][2]] + facenormal * normal_dist(e2));
		}
		else
			output_pts.push_back((1.0 - s - t) * tri_verts[tri_faces[fid][0]] + s * tri_verts[tri_faces[fid][1]] + t * tri_verts[tri_faces[fid][2]]);
//			sample_pts.push_back((1.0 - s - t) * vert_pos[tri_verts[fid][0]] + s * vert_pos[tri_verts[fid][1]] + t * vert_pos[tri_verts[fid][2]]);
		//sample_pt_normals.push_back(facenormal);
		/*if (result.count("sn"))
		{
			sample_pt_normals.push_back(perturb_normal(facenormal, angle_unif_dist(e2), angle_unif_dist(e2)));
		}
		else*/
		if (flag_normal_noise)
		{
			output_normals.push_back(perturb_normal(facenormal, angle_unif_dist(e2), angle_unif_dist(e2)));
		}
		else
		{
			//sample_pt_normals.push_back(facenormal);
			output_normals.push_back(facenormal);

		}
		//sample mask to be added
		//assert(face_color[fid] <= n_color);
		//sample_mask[n_feature_sample + i] = face_color[fid];
		//sample_mask.push_back(face_color[fid]);
		output_masks.push_back(tri_face_masks[fid]);
	}

}

using namespace MeshLib;
void sort_grouped_features(Mesh3d* m, std::vector<std::vector<int>>& grouped_features)
{
	std::vector<std::vector<int>> grouped_features_new;
	for (size_t i = 0; i < grouped_features.size(); i++)
	{
		std::map<int, std::vector<int>> v2hes;
		std::map<int, std::vector<int>> v2vs;
		for (size_t j = 0; j < grouped_features[i].size(); j++)
		{
			int heid = grouped_features[i][j];
			int vid[2];
			vid[0] = m->get_edges_list()->at(heid)->vert->id;
			vid[1] = m->get_edges_list()->at(heid)->pair->vert->id;
			for (size_t k = 0; k < 2; k++)
			{
				if (v2hes.find(vid[k]) == v2hes.end())
				{
					std::vector<int> hes, vs;
					hes.push_back(heid);
					vs.push_back(vid[1 - k]);
					/*v2hes[vid[k]] = std::vector<int>(heid);
					v2vs[vid[k]] = std::vector<int>(vid[1 - k]);*/
					v2hes[vid[k]] = hes;
					v2vs[vid[k]] = vs;
				}
				else
				{
					v2hes[vid[k]].push_back(heid);
					v2vs[vid[k]].push_back(vid[1 - k]);
				}
			}
		}

		int cur_edge = -1;
		int cur_vert = -1;
		int prev_edge = -1;
		for (auto it : v2hes)
		{
			if (it.second.size() == 1)
			{
				cur_vert = it.first;
			}
		}
		//assert(cur_edge != -1);
		if (cur_vert == -1)
		{
			//circular case
			cur_vert = v2hes.begin()->first;
		}
		std::vector<int> one_group;
		//from start to end
		std::queue<int> q;
		std::vector<bool> edge_color(m->get_num_of_edges(), false);
		while (true)
		{
			//get cur_edge
			cur_edge = -1;
			for (auto e : v2hes[cur_vert])
			{
				if (e != prev_edge && edge_color[e] == false)
				{
					cur_edge = e;
					break;
				}
			}
			if (cur_edge == -1)
				break;

			edge_color[cur_edge] = true;
			//get next_vert
			assert(cur_vert == m->get_edges_list()->at(cur_edge)->vert->id || cur_vert == m->get_edges_list()->at(cur_edge)->pair->vert->id);

			int next_vert = m->get_edges_list()->at(cur_edge)->vert->id + m->get_edges_list()->at(cur_edge)->pair->vert->id - cur_vert;
			
			if (next_vert == m->get_edges_list()->at(cur_edge)->vert->id)
			{
				one_group.push_back(cur_edge);
			}
			else
			{
				one_group.push_back(m->get_edges_list()->at(cur_edge)->pair->id);
			}

			cur_vert = next_vert;
			prev_edge = cur_edge;
			
			
		}
		
		assert(one_group.size() == grouped_features[i].size());
		grouped_features_new.push_back(one_group);
	}
	grouped_features = grouped_features_new;
}

int cluster_mesh_faces(Mesh3d* m, const std::vector<bool>& he_feature_flag, std::vector<std::vector<int>> &grouped_features, int cluster_begin_id, std::vector<int>& face2cluster, std::vector<std::pair<int, int>>& feature_twoface_colors)
{
	//return n_cluster + cluster_start_id
	face2cluster.clear();
	feature_twoface_colors.clear();
	face2cluster.resize(m->get_num_of_faces(), -1);
	std::vector<TinyVector<double, 3>> face_centers(m->get_num_of_faces(), TinyVector<double, 3>(0,0,0));
	for (size_t i = 0; i < m->get_num_of_faces(); i++)
	{
		HE_edge<double>* begin_edge = m->get_faces_list()->at(i)->edge;
		HE_edge<double>* edge = begin_edge;
		do
		{
			face_centers[i] = face_centers[i] + edge->vert->pos;
			edge = edge->next;
		} while (edge != begin_edge);
		face_centers[i] = face_centers[i] / 3.0;
	}


	class face_cluster
	{
	public:
		face_cluster(int faceid = 0,  TinyVector<double, 3> ori = TinyVector<double, 3>(0,0,0), TinyVector<double, 3> cur = TinyVector<double, 3>(0, 0, 0))
			:fid(faceid), ori_face_center(ori)
		{
			e = (cur - ori).Length();
			//e = energy;
		}
	public:
		int fid;
		TinyVector<double, 3> ori_face_center;
		double e;
	};
	
	class compare_face_cluster
	{
	public:
		compare_face_cluster()
		{
		}
		bool operator()(const face_cluster& fc1, const face_cluster& fc2) const
		{
			return fc1.e > fc2.e;
		}
	};
	
	std::vector<double> energy(m->get_num_of_faces(), DBL_MAX);
	std::priority_queue<face_cluster, std::deque<face_cluster>, compare_face_cluster> m_queue;
	sort_grouped_features(m, grouped_features);
	std::vector<bool> flag_face_unchangable(m->get_num_of_faces(), false);
	for (size_t i = 0; i < grouped_features.size(); i++)
	{
		int color1 = cluster_begin_id;
		int color2 = cluster_begin_id + 1;
		feature_twoface_colors.push_back(std::pair<int, int>(color1, color2));
		for (size_t j = 0; j < grouped_features[i].size(); j++)
		{
			int heid = grouped_features[i][j];
			int fid1 = m->get_edges_list()->at(heid)->face->id;
			int fid2 = m->get_edges_list()->at(heid)->pair->face->id;

			if (face2cluster[fid1] == -1)
			{
				face2cluster[fid1] = color1;
				m_queue.push(face_cluster(fid1, face_centers[fid1], face_centers[fid1]));
				energy[fid1] = 0;
				/*energy[fid1] = 1 - std::abs(m->get_faces_list()->at(fid1)->normal.Dot(m->get_faces_list()->at(fid2)->normal));
				m_queue.push(face_cluster(fid1, face_centers[fid1], face_centers[fid1], energy[fid1]));*/
				flag_face_unchangable[fid1] = true;
			}
			if (face2cluster[fid2] == -1)
			{
				face2cluster[fid2] = color2;
				m_queue.push(face_cluster(fid2, face_centers[fid2], face_centers[fid2]));
				energy[fid2] = 0;
				//energy[fid2] = energy[fid1];
				//m_queue.push(face_cluster(fid2, face_centers[fid2], face_centers[fid2], energy[fid2]));

				flag_face_unchangable[fid2] = true;
			}
		}
		cluster_begin_id = cluster_begin_id + 2;
	}

	while (!m_queue.empty())
	{
		face_cluster fc = m_queue.top();
		m_queue.pop();
		HE_edge<double>* begin_edge = m->get_faces_list()->at(fc.fid)->edge;
		HE_edge<double>* edge = begin_edge;
		do
		{
			if (he_feature_flag[edge->id])
			{
				edge = edge->next;
				continue;
			}
			assert(fc.fid == edge->face->id || fc.fid == edge->pair->face->id);
			int otherface = edge->face->id + edge->pair->face->id - fc.fid;
			if (!flag_face_unchangable[otherface])
			{
				double tmp_energy = (face_centers[otherface] - fc.ori_face_center).Length();
				//double tmp_energy = 1.0 - std::abs(edge->face->normal.Dot(edge->pair->face->normal));
				if (face2cluster[otherface] == -1)
				{
					face2cluster[otherface] = face2cluster[fc.fid];
					m_queue.push(face_cluster(otherface, fc.ori_face_center, face_centers[otherface]));
					energy[otherface] = tmp_energy;
					//m_queue.push(face_cluster(otherface, fc.ori_face_center, face_centers[otherface], tmp_energy));
				}
				else
				{
					//colored
					if (tmp_energy < energy[otherface])
					{
						face2cluster[otherface] = face2cluster[fc.fid];
						//m_queue.push(face_cluster(otherface, tmp_energy));
						m_queue.push(face_cluster(otherface, fc.ori_face_center, face_centers[otherface]));
						energy[otherface] = tmp_energy;
						//m_queue.push(face_cluster(otherface, fc.ori_face_center, face_centers[otherface], tmp_energy));

					}
				}
			}
			edge = edge->next;
		} while (edge != begin_edge);
	}
	return cluster_begin_id;
}

void get_cluster_from_coloring(const std::vector<int>& face_color, int color_start_id, std::vector<std::vector<int>>& face_clusters)
{
	int max_color = -1;
	for (auto c : face_color)
	{
		assert(c >= color_start_id);
		if (max_color < c)
		{
			max_color = c;
		}
	}
	face_clusters.clear();
	for (size_t i = color_start_id; i <= max_color; i++)
	{
		std::vector<int> one_cluster;
		for (size_t j = 0; j < face_color.size(); j++)
		{
			if (face_color[j] == i)
			{
				one_cluster.push_back(j);
			}
		}
		face_clusters.push_back(one_cluster);
	}
}

int merge_clusters(Mesh3d* m, const std::vector<bool>& he_feature_flag, int cluster_begin_id, int n_cluster, const std::vector<std::pair<int, int>>& feature_twoface_colors, std::vector<int>& face_color)
{
	//return num_clusters + cluster_begin_id
	std::vector<int> feature_pair_mapping(n_cluster, -1);
	for (auto& p : feature_twoface_colors)
	{
		feature_pair_mapping[p.first - 1] = p.second - 1;
		feature_pair_mapping[p.second - 1] = p.first - 1;
	}
	std::vector<int> cluster_o2n(n_cluster, -1); //starting from 0->0
	std::vector<std::set<int>> con(n_cluster); //split by features
	int cluster_id = 0;

	for (size_t i = 0; i < m->get_num_of_edges(); i++)
	{
		if (he_feature_flag[i])
			continue;
		int color1 = face_color[m->get_edges_list()->at(i)->face->id] - 1;
		int color2 = face_color[m->get_edges_list()->at(i)->pair->face->id] - 1;
		if (color1 != color2)
		{
			con[color1].insert(color2);
			con[color2].insert(color1);
		}
	}
	
	for (size_t i = 0; i < n_cluster; i++)
	{
		if (cluster_o2n[i] != -1)
			continue;
		std::queue<int> q;
		q.push(i);
		cluster_o2n[i] = cluster_id;
		std::set<int> mset;
		while (!q.empty())
		{
			int front = q.front();
			q.pop();
			for (auto nn : con[front])
			{
				if (cluster_o2n[nn] != -1)
					continue;
				int pair_cluster = feature_pair_mapping[nn];
				if (cluster_o2n[pair_cluster] == cluster_id)
					continue;
				cluster_o2n[nn] = cluster_id;
				q.push(nn);
			}
		}
		cluster_id = cluster_id + 1;
	}

	for (size_t i = 0; i < face_color.size(); i++)
	{
		face_color[i] = cluster_o2n[face_color[i] - 1] + 1;
	}

	return cluster_id + 1;
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