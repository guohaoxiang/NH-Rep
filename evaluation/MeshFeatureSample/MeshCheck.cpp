#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <direct.h>
#include "cxxopts.hpp"
#include "Mesh3D.h"

using namespace MeshLib;

std::string GetFileExtension(const std::string& FileName)
{
	if (FileName.find_last_of(".") != std::string::npos)
		return FileName.substr(FileName.find_last_of(".") + 1);
	return "";
}

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Real_timer.h>
#include <CGAL/tags.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3>                      Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor          face_descriptor;
namespace PMP = CGAL::Polygon_mesh_processing;

bool CheckMeshIntersection(const char* filename)
{
	std::ifstream input(filename);
	Mesh mesh;
	if (!input || !(input >> mesh) || !CGAL::is_triangle_mesh(mesh))
	{
		std::cerr << "Not a valid input file." << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "Using parallel mode? " << std::is_same<CGAL::Parallel_if_available_tag, CGAL::Parallel_tag>::value << std::endl;
	CGAL::Real_timer timer;
	timer.start();
	bool intersecting = PMP::does_self_intersect<CGAL::Parallel_if_available_tag>(mesh, CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, mesh)));
	std::cout << (intersecting ? "There are self-intersections." : "There is no self-intersection.") << std::endl;
	std::cout << "Elapsed time (does self intersect): " << timer.time() << std::endl;
	timer.reset();
	std::vector<std::pair<face_descriptor, face_descriptor> > intersected_tris;
	PMP::self_intersections<CGAL::Parallel_if_available_tag>(faces(mesh), mesh, std::back_inserter(intersected_tris));
	std::cout << intersected_tris.size() << " pairs of triangles intersect." << std::endl;
	std::cout << "Elapsed time (self intersections): " << timer.time() << std::endl;

	return intersecting;
}


int main(int argc, char** argv)
{
	cxxopts::Options options("MeshCheck", "Point Sampling program for featured CAD models (author: Haoxiang Guo, Email: guohaoxiangxiang@gmail.com)");
	options
		.positional_help("[optional args]")
		.show_positional_help()
		.allow_unrecognised_options()
		.add_options()
		("i,input", "input mesh (obj/off format)", cxxopts::value<std::string>())
		//("o,output", "output file", cxxopts::value<std::string>())
		("f,feature", "input feature (obj/off format)", cxxopts::value<std::string>())
		("h,help", "print help");
	
	auto result = options.parse(argc, argv);

	if (result.count("help"))
	{
		std::cout << options.help({ "", "Group" }) << std::endl;
		exit(0);
	}
	std::string inputfile = result["i"].as<std::string>();
	std::string inputext = GetFileExtension(inputfile);
	//auto& outputfile = result["o"].as<std::string>();
	Mesh3d mesh;
	if (inputext == "obj")
		mesh.load_obj(inputfile.c_str());
	else if (inputext == "off")
		mesh.load_off(inputfile.c_str());
	std::cout << "verts: " << mesh.get_vertices_list()->size() << " face:  " << mesh.get_faces_list()->size() << std::endl;
	//mesh.write_obj("tmp.obj");
	if (inputext == "obj")
	{
		//save to off
		inputfile = inputfile.substr(0, inputfile.length() - 3) + "off";
		mesh.write_off(inputfile.c_str());
	}

	int flag_legal = 1;

	for (size_t i = 0; i < mesh.get_num_of_edges(); i++)
	{
		HE_edge<double>* edge = mesh.get_edges_list()->at(i);
		if (edge->face == NULL || edge->pair->face == NULL)
		{
			std::cout << "openmesh detected" << std::endl;
			flag_legal = 0;
			break;
		}
	}

	if (flag_legal)
	{
		//intersection
		flag_legal = !CheckMeshIntersection(inputfile.c_str());
	}

	//ofs
	/*std::ofstream ofs;
	ofs.open(outputfile, std::ofstream::out | std::ofstream::app);
	ofs << inputfile << std::endl;
	ofs << flag_legal << std::endl;*/

	//modified 1203
	if (!flag_legal)
	{
		std::ofstream ofs(inputfile + "mesh_invalid");
		ofs.close();
	}

	return 1;
}