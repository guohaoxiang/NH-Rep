#include <iostream>
#include "Tree.h"
#include "helper.h"

template<typename T>
std::string convert_tree_to_string(const TreeNode<T>* tree)
{
	std::string str = "[";
	for (auto v : tree->keys)
	{
		str = str + std::to_string(v) + ",";
	}

	for (size_t i = 0; i < tree->children.size(); i++)
	{
		str = str + convert_tree_to_string<T>(tree->children[i]);
	}

	str = str + "],";
	
	return str;
}

template<typename T>
T tree_coloring(const TreeNode<T>* tree, const std::vector<std::set<T>>& connectivity_v, std::vector<T>& node_color, T start_id, int max_patch_per_cluster)
{
	//return the number of node coloring
	std::vector<T> o2n(connectivity_v.size(), T(-1));
	std::vector<T> n2o;
	T ncount = 0;
	for (auto v : tree->keys)
	{
		o2n[v] = ncount++;
		n2o.push_back(v);
	}
	std::vector<std::set<T>> local_con(ncount);
	for (auto v : tree->keys)
	{
		for (auto vn : connectivity_v[v])
		{
			if (std::find(tree->keys.begin(), tree->keys.end(), vn) != tree->keys.end())
			{
				//both are found in keys
				local_con[o2n[v]].insert(o2n[vn]);
			}
		}
	}
	std::vector<std::vector<T>> colored_vertices;
	if (ncount != 0)
		greedy_graph_coloring(ncount, local_con, colored_vertices);
	//update colored_vertices if mp is set
	if (max_patch_per_cluster != -1)
	{
		std::vector<std::vector<T>> colored_vertices_new;
		for (size_t i = 0; i < colored_vertices.size(); i++)
		{
			if (colored_vertices[i].size() <= max_patch_per_cluster)
			{
				colored_vertices_new.push_back(colored_vertices[i]);
			}
			else
			{
				//split colored_vertices
				int n_split = std::ceil(1.0 * colored_vertices[i].size() / max_patch_per_cluster);
				for (int j = 0; j < n_split; j++)
				{
					int begin = j * max_patch_per_cluster;
					int end = (j + 1) * max_patch_per_cluster;
					if (end > colored_vertices[i].size())
						end = colored_vertices[i].size();
					colored_vertices_new.push_back(std::vector<T>{colored_vertices[i].begin() + begin, colored_vertices[i].begin() + end});
				}
			}
		}

		colored_vertices = colored_vertices_new;
	}


	//std::vector<T> cluster_color(connectivity.size()); //starting from 0
	for (size_t i = 0; i < colored_vertices.size(); i++)
	{
		for (size_t j = 0; j < colored_vertices[i].size(); j++)
		{
			colored_vertices[i][j] = n2o[colored_vertices[i][j]];
			node_color[colored_vertices[i][j]] = i + start_id;
		}
	}
	start_id += colored_vertices.size();
	for (size_t i = 0; i < tree->children.size(); i++)
	{
		T n_color = tree_coloring(tree->children[i], connectivity_v, node_color, start_id);
		start_id = n_color;
	}
	return start_id;
	

}

template std::string convert_tree_to_string<size_t>(const TreeNode<size_t>*);

template size_t tree_coloring(const TreeNode<size_t>* tree, const std::vector<std::set<size_t>>& connectivity_v, std::vector<size_t>& node_color, size_t start_id, int mp);
