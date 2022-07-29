#include <boost/config.hpp>
#include <iostream>
#include <string>
#include <map>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/read_dimacs.hpp>
#include <boost/graph/graph_utility.hpp>

#include "MaxFlow.h"



void MaxFlow(int n_vert, const vector<std::pair<int, int>> &edge_array, const vector<double> &cap, int sink, int tar, vector<double>& output_flow)
{
	//neighboring information need to be considered carefully
	output_flow.clear();
	assert(edge_array.size() == cap.size());
	assert(sink >= 0 && sink < n_vert && tar >= 0 && tar < n_vert);
	using namespace boost;

	typedef adjacency_list_traits < vecS, vecS, directedS > Traits;
	typedef adjacency_list < vecS, vecS, directedS,
		property < vertex_name_t, std::string,
		property < vertex_index_t, long,
		property < vertex_color_t, boost::default_color_type,
		property < vertex_distance_t, double,
		property < vertex_predecessor_t, Traits::edge_descriptor > > > > >,

		property < edge_capacity_t, double,
		property < edge_residual_capacity_t, double,
		property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;

	typedef Traits::vertex_descriptor Vertex;
	typedef Traits::edge_descriptor Edge;

	Graph g;

	property_map < Graph, edge_capacity_t >::type
		capacity = get(edge_capacity, g);
	property_map < Graph, edge_residual_capacity_t >::type
		residual_capacity = get(edge_residual_capacity, g);
	property_map < Graph, edge_reverse_t >::type rev = get(edge_reverse, g);

	property_map <Graph, vertex_color_t>::type
		node_color = get(vertex_color, g);

	Traits::vertex_descriptor s, t;

	vector<Vertex> v_vert;
	vector<Edge> v_edge;

	//initialization
	for (size_t i = 0; i < n_vert; i++)
	{
		v_vert.push_back(add_vertex(g));
	}
	s = v_vert[sink];
	t = v_vert[tar];
	
	int num_arcs = edge_array.size();

	for (size_t i = 0; i < num_arcs; i++)
	{
		Edge e1, e2, e3;
		bool in1, in2;
		boost::tie(e1, in1) = add_edge(v_vert[edge_array[i].first], v_vert[edge_array[i].second], g);
		boost::tie(e2, in2) = add_edge(v_vert[edge_array[i].second], v_vert[edge_array[i].first], g);

		if (!in1 || !in2) {
			std::cerr << "unable to add edge ("
				<< std::endl;
		}

		capacity[e1] = cap[i];
		capacity[e2] = -1.0;
		rev[e1] = e2;
		rev[e2] = e1;

	}

	std::vector<default_color_type> color(num_vertices(g));
	std::vector<long> distance(num_vertices(g));
	double flow = boykov_kolmogorov_max_flow(g, s, t);

	std::cout << "c  The total flow:" << std::endl;
	std::cout << "s " << flow << std::endl << std::endl;

	graph_traits < Graph >::vertex_iterator u_iter, u_end;
	graph_traits <Graph>::edge_iterator ei, e_end;
	
	std::map<std::pair<int, int>, double> edge2flow;
	
	for (boost::tie(ei, e_end) = edges(g); ei != e_end; ++ei)
	{
		if (capacity[*ei] > -0.5)
		{
			edge2flow[std::pair<int, int>(source(*ei, g), target(*ei, g))] = capacity[*ei] - residual_capacity[*ei];
		} 
	}

	assert(edge2flow.size() == edge_array.size());

	for (size_t i = 0; i < edge_array.size(); i++)
	{
		output_flow.push_back(edge2flow[edge_array[i]]);
	}


	std::cout << "flow computing finished" << std::endl;
}

void MinCutfromMaxFlow(int n_vert, const vector<std::pair<int, int>>& edge_array, const vector<double>& cap, int sink, int tar, vector<int>& part1, vector<int>& part2)
{
	//input should be a directed graph
	//get min cut from max flow result, store them in part1, part2
	//part1 -> sink, part2->tar, excluded

	assert(edge_array.size() == cap.size());
	assert(sink >= 0 && sink < n_vert&& tar >= 0 && tar < n_vert);
	//std::cout << "max flow" << std::endl;
	using namespace boost;

	typedef adjacency_list_traits < vecS, vecS, directedS > Traits;
	typedef adjacency_list < vecS, vecS, directedS,
		property < vertex_name_t, std::string,
		property < vertex_index_t, long,
		property < vertex_color_t, boost::default_color_type,
		property < vertex_distance_t, double,
		property < vertex_predecessor_t, Traits::edge_descriptor > > > > >,

		property < edge_capacity_t, double,
		property < edge_residual_capacity_t, double,
		property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;

	typedef Traits::vertex_descriptor Vertex;
	typedef Traits::edge_descriptor Edge;

	Graph g;

	property_map < Graph, edge_capacity_t >::type
		capacity = get(edge_capacity, g);
	property_map < Graph, edge_residual_capacity_t >::type
		residual_capacity = get(edge_residual_capacity, g);
	property_map < Graph, edge_reverse_t >::type rev = get(edge_reverse, g);
	Traits::vertex_descriptor s, t;

	property_map <Graph, vertex_color_t>::type
		node_color = get(vertex_color, g);

	vector<Vertex> v_vert;
	vector<Edge> v_edge;

	//initialization
	for (size_t i = 0; i < n_vert; i++)
	{
		v_vert.push_back(add_vertex(g));
	}
	s = v_vert[sink];
	t = v_vert[tar];

	int num_arcs = edge_array.size();

	for (size_t i = 0; i < num_arcs; i++)
	{
		Edge e1, e2, e3;
		bool in1, in2;
		boost::tie(e1, in1) = add_edge(v_vert[edge_array[i].first], v_vert[edge_array[i].second], g);
		boost::tie(e2, in2) = add_edge(v_vert[edge_array[i].second], v_vert[edge_array[i].first], g);

		if (!in1 || !in2) {
			std::cerr << "unable to add edge ("
				<< std::endl;
			//return -1;
		}

		capacity[e1] = cap[i];
		capacity[e2] = -1.0;
		rev[e1] = e2;
		rev[e2] = e1;

	}

	std::vector<default_color_type> color(num_vertices(g));
	std::vector<long> distance(num_vertices(g));
	double flow = boykov_kolmogorov_max_flow(g, s, t);

	graph_traits < Graph >::vertex_iterator u_iter, u_end;
	graph_traits < Graph >::out_edge_iterator ei, e_end;

	//direct get color
	std::vector<int> label0(n_vert, 0);
	//graph_traits < Graph >::vertex_iterator u_iter, u_end;
	for (boost::tie(u_iter, u_end) = vertices(g); u_iter != u_end; ++u_iter)
	{
		if (node_color[*u_iter] == boost::default_color_type::black_color)
		{
			label0[*u_iter] = 1;
		}
	}

	for (int i = 0; i < n_vert; i++)
	{
		if (i == sink || i == tar)
			continue;
		if (label0[i] == 1)
			part1.push_back(i);
		else
			part2.push_back(i);
	}
	return;
}




#include <boost/graph/graph_traits.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/typeof/typeof.hpp>

typedef std::pair<int, int> edge_t;

void MinCut(int n_vert, const vector<std::pair<int, int>> &edge, const vector<double> &cap, vector<int> &part1, vector<int> &part2)
{
	using namespace std;

	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
		boost::no_property, boost::property<boost::edge_weight_t, double> > undirected_graph;
	typedef boost::property_map<undirected_graph, boost::edge_weight_t>::type weight_map_type;
	typedef boost::property_traits<weight_map_type>::value_type weight_type;

	const edge_t *edges = &edge[0];
	const weight_type *ws = &cap[0];
	undirected_graph g(edges, edges + edge.size(), ws, n_vert, edge.size());
	BOOST_AUTO(parities, boost::make_one_bit_color_map(num_vertices(g), get(boost::vertex_index, g)));

	int w = boost::stoer_wagner_min_cut(g, get(boost::edge_weight, g), boost::parity_map(parities));

	cout << "The min-cut weight of G is " << w << ".\n" << endl;

	//two parts
	part1.clear();
	part2.clear();
	cout << "One set of vertices consists of:" << endl;
	size_t i;
	for (i = 0; i < num_vertices(g); ++i) {
		if (get(parities, i))
			part1.push_back(i);
	}

	cout << "The other set of vertices consists of:" << endl;
	for (i = 0; i < num_vertices(g); ++i) {
		if (!get(parities, i))
			part2.push_back(i);
	}
}