// multi-branch tree
#pragma once
#include <vector>
#include <string>
#include <set>

template<typename T>
struct TreeNode
{
	std::set<T> keys;
	std::vector<TreeNode*> children;
	int layer;
};

template<typename T>
std::string convert_tree_to_string(const TreeNode<T> *tree);

template<typename T>
T tree_coloring(const TreeNode<T>* tree, const std::vector<std::set<T>> &connectivity_v,std::vector<T>& node_color, T start_id = 0, int max_patch_per_cluster = -1);