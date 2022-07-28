#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include "cxxopts.hpp"
#include "yaml.h"
#include "MyObjLoader.h"
#include "MyCurve.h"
#include "MySurf.h"
#include "Helper.h"
#include "Mesh3D.h"
#include "nanoflann.hpp"

//for distance queries
#include <igl/writeOBJ.h>
#include <igl/AABB.h>

using namespace std;
using namespace MeshLib;

typedef TinyVector<double, 3> vec3;

void output_info(const std::string &str, bool flag_close, bool flag_legal_curve, bool flag_legal_patch, bool flag_consistency, bool flag_idvalid)
{
    std::ofstream ofs1(str);

    ofs1 << "watertight " << flag_close << std::endl;
    ofs1 << "curvelegal " << flag_legal_curve << std::endl;
    ofs1 << "patchlegal " << flag_legal_patch << std::endl;
    ofs1 << "consistency " << flag_consistency << std::endl;
    ofs1 << "idvalid " << flag_idvalid << std::endl;
    ofs1.close();
}

class PointSet;
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointSet>, PointSet, 3> my_kd_tree_t;
class PointSet
{
public:
    PointSet()
    {
        m_kdtree = nullptr;
    }
    ~PointSet()
    {
        if (m_kdtree != nullptr)
            delete m_kdtree;
    }
    void build()
    {
        if (points.empty()) return;
        if (m_kdtree != nullptr)
            delete m_kdtree;
        int dim = 3;
        m_kdtree = new my_kd_tree_t(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        m_kdtree->buildIndex();
    }
    ///////////////////////////////////////////////////////
    inline size_t kdtree_get_point_count() const { return points.size(); }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim < 3)
            return points[idx][(int)dim];
        else
            return 0;
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    ///////////////////////////////////////////////////////
    void push(const vec3& point)
    {
        points.push_back(point);
    }
    
    void push_normal(const vec3& normal)
    {
        normals.push_back(normal);
    }

    const size_t get_nearest_point_id(const vec3& vp) const
    {
        nanoflann::KNNResultSet<double> resultSet(1);
        size_t nearest_point_id;
        double distance;
        resultSet.init(&nearest_point_id, &distance);
        m_kdtree->findNeighbors(resultSet, &vp[0], nanoflann::SearchParams(10));
        return nearest_point_id;
    }


//protected:
    my_kd_tree_t* m_kdtree;
    std::vector<vec3> points;
    std::vector<vec3> normals;
};

class face_info
{
public:
    std::string surf_type;
    int surf_index;
    size_t face_index;
    std::vector<double> u, v;
    bool reverse_normal;
    std::vector<vec3d> para_normals; // normals of parameter points
};

std::string GetFileExtension(const std::string &FileName)
{
    if (FileName.find_last_of(".") != std::string::npos)
        return FileName.substr(FileName.find_last_of(".") + 1);
    return "";
}

template<typename T>
T compute_tri_area(const TinyVector<T, 3>& v0, const TinyVector<T, 3>& v1, const TinyVector<T, 3>& v2)
{
    return ((v1 - v0).Cross(v2 - v0)).Length() * (T)0.5;
}

bool sample_pts_from_mesh_parametric(const std::vector<TinyVector<double, 3>>& tri_verts, const std::vector<TinyVector<size_t, 3>>& tri_faces, const std::vector<TinyVector<double, 3>>& tri_normals, int n_sample, const std::vector<MySurf*>& allsurfs, const std::vector<face_info>& face_info_for_resampling, std::vector<TinyVector<double, 3>>& output_pts, std::vector<TinyVector<double, 3>>& output_normals, std::vector<int>& output_pts_faces)
{
    if (tri_faces.empty())
    {
        std::cout << "!!!no face input for sampling" << std::endl;
        return false;
    }
    
    if (tri_faces.size() != face_info_for_resampling.size())
    {
        std::cout << "tri faces and fis size df" << std::endl;
        return false;
    }

    output_pts.clear();
    output_normals.clear();
    output_pts_faces.clear();
    //feature parts first
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<double> unif_dist(0, 1);

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
        fid = std::max(0, fid - 1);
        double s = unif_dist(e2);
        double t = unif_dist(e2);
        if (s + t > 1)
        {
            s = 1 - s;
            t = 1 - t;
        }
        TinyVector<double, 3> facenormal = tri_normals[fid];
        const auto& type = face_info_for_resampling[fid].surf_type;
        int surf_index = face_info_for_resampling[fid].surf_index;
        bool reverse_normal = face_info_for_resampling[fid].reverse_normal;
        if (type == "Other")
        {
            //direct sample from mesh
            output_pts.push_back((1.0 - s - t) * tri_verts[tri_faces[fid][0]] + s * tri_verts[tri_faces[fid][1]] + t * tri_verts[tri_faces[fid][2]]);
            output_normals.push_back(facenormal);
        }
        else
        {
            const std::vector<double>& us = face_info_for_resampling[fid].u;
            const std::vector<double>& vs = face_info_for_resampling[fid].v;
            double au = 0.0, av = 0.0;
            au = (1.0 - s - t) * us[0] + s * us[1] + t * us[2];
            av = (1.0 - s - t) * vs[0] + s * vs[1] + t * vs[2];
            vec3d p = allsurfs[surf_index]->GetPosition(au, av);
            vec3d pn = allsurfs[surf_index]->GetNormal(au, av);
            if (reverse_normal)
            {
                pn = -pn;
            }
            output_pts.push_back(p);
            output_normals.push_back(pn);
        }

        output_pts_faces.push_back(face_info_for_resampling[fid].face_index);
    }

    return true;
}

int main(int argc, char** argv)
{
    double max_point_mesh_dist = 0.05;
    double target_bb_scale = 1.8;
    int min_sample_perpatch = 50;
    int n_nonfeature_sample = 50000;

    try
    {
        cxxopts::Options options("PatchGen", "PatchGen (author: Haoxiang Guo)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()("y,yaml", "yaml filename (*.yaml)", cxxopts::value<std::string>())
            ("o,obj", "obj filename (*.obj)", cxxopts::value<std::string>())
            ("ns", "number of samples on non-feature faces(default: 50000)", cxxopts::value<int>())
            ("h,help", "Print help");

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({ "", "Group" }) << std::endl;
            exit(0);
        }
        
        std::string yamlfilename, objfilename;
        if (result.count("y"))
        {
            yamlfilename = result["y"].as<std::string>();
        }
        else
        {
            std::cout << "No input yaml file!\n";
            exit(0);
        }
        std::string filenamewithoutext(yamlfilename);
        filenamewithoutext = filenamewithoutext.substr(0, filenamewithoutext.find_last_of("."));
        if (result.count("o"))
        {
            objfilename = result["o"].as<std::string>();
        }
        else
        {
            objfilename = filenamewithoutext + ".obj";
        }
        if (result.count("ns"))
            n_nonfeature_sample = result["ns"].as<int>();
        Mesh3d mesh;
        mesh.load_obj(objfilename.c_str());
        mesh.compute_boundingbox();
        double mesh_scale = std::max(mesh.xmax - mesh.xmin, std::max(mesh.ymax - mesh.ymin, mesh.zmax - mesh.zmin));
        mesh_scale = target_bb_scale / mesh_scale;
        vec3d mesh_center = vec3d((mesh.xmax + mesh.xmin) / 2.0, (mesh.ymax + mesh.ymin) / 2.0, (mesh.zmax + mesh.zmin) / 2.0);
        int n_mesh_faces = mesh.get_num_of_faces();
        std::set<MySortedTuple<2>> sharp_edges;
        vector<vec3> vertices;
        vector<vector<size_t>> facets;
        double sharp_angle_in_degree = 30;
        if (!obj_loader(objfilename.c_str(), vertices, sharp_edges, facets, sharp_angle_in_degree))
        {
            std::cout << "Failed to load " << objfilename << " correctly!\n";
            exit(0);
        }

        bool flag_close = true;
        auto el = mesh.get_edges_list();
        for (size_t i = 0; i < el->size(); i++)
        {
            if (mesh.is_on_boundary(el->at(i)))
            {
                flag_close = false;
                break;
            }
        }

        //if not close, return
        if (!flag_close)
        {
            std::cout << "open mesh detected" << std::endl;
            std::ofstream ofs(filenamewithoutext + ".fail");
            ofs.close();
            return 0;
        }
        
        bool flag_idvalid = true; //whether face id valid
        bool flag_distclose = true; //whether distance is close enough
        bool flag_paramvalid = true; //whether vert param is valid

        std::vector<std::string> surf_types;
        std::vector<MySurf*> allsurfs;


        YAML::Node abcfile = YAML::LoadFile(yamlfilename);
        //process patches
        YAML::Node surfaces = abcfile["surfaces"];
        std::cout << "# surfaces: " << surfaces.size() << std::endl;

        
        std::vector<face_info> face_info_for_resampling;
        std::vector<std::vector<size_t>> patch2fis;
        bool with_spline = false; //with spline/extrusion/revolution/other
        for (auto s : surfaces)
        {
            auto type = s["type"].as<std::string>();
            surf_types.push_back(type);
            if (type == "BSpline" || type == "Extrusion" || type == "Revolution" || type == "Other")
            {
                with_spline = true;
            }
            //const auto& vert_indices = s["vert_indices"]; //not corrected
            const auto& vert_parameters = s["vert_parameters"];
            const auto& face_indices = s["face_indices"];
            bool u_is_closed = false, v_is_closed = false;
            double valid_max_u = -DBL_MAX, valid_min_u = DBL_MAX, valid_max_v = -DBL_MAX, valid_min_v = DBL_MAX;
            std::vector<double> us, vs;
            for (size_t i = 0; i < vert_parameters.size(); i++)
            {
                double u = vert_parameters[i][0].as<double>();
                double v = vert_parameters[i][1].as<double>();
                if (!std::isinf(u))
                {
                    valid_max_u = std::max(u, valid_max_u), valid_min_u = std::min(u, valid_min_u);
                    us.push_back(u);
                }
                else
                {
                    flag_paramvalid = false;
                }
                if (!std::isinf(v))
                {
                    valid_max_v = std::max(v, valid_max_v), valid_min_v = std::min(v, valid_min_v);
                    vs.push_back(v);
                }
                else 
                {
                    flag_paramvalid = false;
                }
            }

            if (!flag_paramvalid)
            {
                break;
            }
            double valid_u_diff = valid_max_u - valid_min_u, valid_v_diff = valid_max_v - valid_min_v;
            for (size_t i = 0; i < face_indices.size(); i++)
            {
                size_t fid = face_indices[i].as<size_t>();
                if (fid >= n_mesh_faces)
                {
                    flag_idvalid = false;
                    break;
                }

                HE_edge<double>* cur = mesh.get_face(fid)->edge;
                HE_edge<double>* iter = cur;
                do
                {
                    iter = iter->next;
                } while (iter != cur);
            }

            if (!flag_idvalid)
            {
                break;
            }

            int num_loop = 1;
            if (!(type == "Plane"))
            {
                std::vector<size_t> selected_facets(face_indices.size());
                for (size_t i = 0; i < face_indices.size(); i++)
                {
                    selected_facets[i] = face_indices[i].as<size_t>();
                }
                std::cout << type << " ";
                num_loop = detect_num_loops(facets, selected_facets);

                
            }


            double u_scale = 1, v_scale = 1;

            if (type == "Plane")
            {
                const auto& location = s["location"];
                const auto& x_axis = s["x_axis"];
                const auto& y_axis = s["y_axis"];
                const auto& z_axis = s["z_axis"];
                //const auto &coefficients = s["coefficients"];
                double max_u = valid_max_u, min_u = valid_min_u, max_v = valid_max_v, min_v = valid_min_v;

                vec3 xdir(x_axis[0].as<double>(), x_axis[1].as<double>(), x_axis[2].as<double>());
                vec3 ydir(y_axis[0].as<double>(), y_axis[1].as<double>(), y_axis[2].as<double>());
                vec3 loc(location[0].as<double>(), location[1].as<double>(), location[2].as<double>());
                
                MySurf* tmp = new MyPlane(loc, xdir, ydir, min_u, max_u, min_v, max_v);

                allsurfs.push_back(tmp);

            }
            else if (type == "Cylinder")
            {
                const auto& location = s["location"];
                const auto& x_axis = s["x_axis"];
                const auto& y_axis = s["y_axis"];
                const auto& z_axis = s["z_axis"];
                const auto radius = s["radius"].as<double>();

                u_scale = 0.001;
                double max_u = 0.001 * valid_max_u, min_u = 0.001 * valid_min_u, max_v = valid_max_v, min_v = valid_min_v;

                vec3 xdir(x_axis[0].as<double>(), x_axis[1].as<double>(), x_axis[2].as<double>());
                vec3 ydir(y_axis[0].as<double>(), y_axis[1].as<double>(), y_axis[2].as<double>());
                vec3 zdir(z_axis[0].as<double>(), z_axis[1].as<double>(), z_axis[2].as<double>());
                vec3 loc(location[0].as<double>(), location[1].as<double>(), location[2].as<double>());
                bool u_closed = u_is_closed;

                MySurf* tmp = new MyCylinder(loc, xdir, ydir, zdir, radius, u_closed, min_u, max_u, min_v, max_v);
                allsurfs.push_back(tmp);
                
            }
            else if (type == "Cone")
            {
                //todo: check oneside closeness
                const auto& location = s["location"];
                const auto& x_axis = s["x_axis"];
                const auto& y_axis = s["y_axis"];
                const auto& z_axis = s["z_axis"];
                const auto radius = s["radius"].as<double>();
                const auto angle = s["angle"].as<double>();
                //const auto &coefficients = s["coefficients"];

                double max_u = 0.001 * valid_max_u, min_u = 0.001 * valid_min_u, max_v = valid_max_v, min_v = valid_min_v;
                u_scale = 0.001;

                vec3 xdir(x_axis[0].as<double>(), x_axis[1].as<double>(), x_axis[2].as<double>());
                vec3 ydir(y_axis[0].as<double>(), y_axis[1].as<double>(), y_axis[2].as<double>());
                vec3 zdir(z_axis[0].as<double>(), z_axis[1].as<double>(), z_axis[2].as<double>());
                vec3 loc(location[0].as<double>(), location[1].as<double>(), location[2].as<double>());
                //bool u_closed = (num_loop >= 2);
                bool u_closed = u_is_closed;
                if (!u_closed)
                {
                    if (max_u - min_u > 2 * M_PI)
                    {
                        std::cout << "angle diff: " << max_u - min_u << std::endl;
                        update_minmax(us, min_u, max_u);
                    }
                }

                MySurf* tmp = new MyCone(loc, xdir, ydir, zdir, radius, angle, u_closed, min_u, max_u, min_v, max_v);
                allsurfs.push_back(tmp);

            }
            else if (type == "Torus")
            {
                const auto& location = s["location"];
                const auto& x_axis = s["x_axis"];
                const auto& y_axis = s["y_axis"];
                const auto& z_axis = s["z_axis"];
                const auto max_radius = s["max_radius"].as<double>();
                const auto min_radius = s["min_radius"].as<double>();
                //const auto radius = s["radius"].as<double>();

                double max_u = 0.001 * valid_max_u, min_u = 0.001 * valid_min_u, max_v = 0.001 * valid_max_v, min_v = 0.001 * valid_min_v;
                u_scale = 0.001, v_scale = 0.001;

                vec3 xdir(x_axis[0].as<double>(), x_axis[1].as<double>(), x_axis[2].as<double>());
                vec3 ydir(y_axis[0].as<double>(), y_axis[1].as<double>(), y_axis[2].as<double>());
                vec3 zdir(z_axis[0].as<double>(), z_axis[1].as<double>(), z_axis[2].as<double>());
                vec3 loc(location[0].as<double>(), location[1].as<double>(), location[2].as<double>());

                bool u_closed = u_is_closed, v_closed = v_is_closed;
                if (u_closed && v_closed && num_loop != 0)
                {
                    v_closed = false;
                }

                if (!u_closed)
                {
                    if (max_u - min_u > 2 * M_PI)
                    {
                        std::cout << "angle diff: " << max_u - min_u << std::endl;
                        update_minmax(us, min_u, max_u);
                    }
                }

                if (!v_closed)
                {
                    if (max_v - min_v > 2 * M_PI)
                    {
                        std::cout << "v angle diff: " << max_v - min_v << std::endl;
                        update_minmax(vs, min_v, max_v);
                    }
                }

                MySurf* tmp = new MyTorus(loc, xdir, ydir, zdir, max_radius, min_radius, u_closed, v_closed, min_u, max_u, min_v, max_v);
                allsurfs.push_back(tmp);

            }
            else if (type == "Sphere")
            {
                //recheck
                u_is_closed = false;
                v_is_closed = false;
                std::vector<size_t> selected_facets(face_indices.size());
                for (size_t i = 0; i < face_indices.size(); i++)
                {
                    selected_facets[i] = face_indices[i].as<size_t>();
                }
                std::cout << type << " ";
                num_loop = detect_num_loops(facets, selected_facets);

                //todo: check oneside closeness
                const auto& location = s["location"];
                const auto& x_axis = s["x_axis"];
                const auto& y_axis = s["y_axis"];
                const auto radius = s["radius"].as<double>();
                double max_u = 0.001 * valid_max_u, min_u = 0.001 * valid_min_u, max_v = 0.001 * valid_max_v, min_v = 0.001 * valid_min_v;
                u_scale = 0.001, v_scale = 0.001;

                vec3 xdir(x_axis[0].as<double>(), x_axis[1].as<double>(), x_axis[2].as<double>());
                vec3 ydir(y_axis[0].as<double>(), y_axis[1].as<double>(), y_axis[2].as<double>());
                vec3 zdir = xdir.UnitCross(ydir);
                vec3 loc(location[0].as<double>(), location[1].as<double>(), location[2].as<double>());

                bool u_closed = u_is_closed, v_closed = v_is_closed;
                if (u_closed && v_closed && num_loop != 0)
                {
                    v_closed = false;
                }
                MySurf* tmp = new MySphere(loc, xdir, ydir, zdir, radius, u_closed, v_closed, min_u, max_u, min_v, max_v);
                allsurfs.push_back(tmp);

            }
            else if (type == "Revolution")
            {
                const auto& profile = s["curve"];
                const auto& z_axis = s["z_axis"];
                vec3 zvec(z_axis[0].as<double>(), z_axis[1].as<double>(), z_axis[2].as<double>());
                const auto location = s["location"];
                vec3 loc(location[0].as<double>(), location[1].as<double>(), location[2].as<double>());
                const string type = profile["type"].as<string>();

                //uv swap
                double max_v = 0.001 * valid_max_u, min_v = 0.001 * valid_min_u, max_u = 0.001 * valid_max_v, min_u = 0.001 * valid_min_v;

                u_scale = 0.001, v_scale = 0.001;

                bool u_closed = v_is_closed, v_closed = u_is_closed;

                if (num_loop == 0)
                {
                    v_closed = true;
                }

                if (v_closed)
                {
                    min_v = 0, max_v = 2 * M_PI;
                }

                if (type == "BSpline")
                {
                    const auto closed = profile["closed"].as<bool>();
                    const auto continuity = profile["continuity"].as<int>();
                    const auto degree = profile["degree"].as<int>();
                    const auto& knots = profile["knots"];
                    const auto& poles = profile["poles"];
                    const bool rational = profile["rational"].as<bool>();
                    const auto& weights = profile["weights"];
                    std::vector<double> myknots(knots.size());
                    for (size_t i = 0; i < knots.size(); i++)
                        myknots[i] = knots[i].as<double>();

                    std::vector<gte::Vector<3, double>> controls(poles.size());
                    std::vector<double> myweights(poles.size(), 1);
                    for (size_t i = 0; i < poles.size(); i++)
                    {
                        controls[i][0] = poles[i][0].as<double>();
                        controls[i][1] = poles[i][1].as<double>();
                        controls[i][2] = poles[i][2].as<double>();
                    }
                    if (rational)
                        for (size_t i = 0; i < weights.size(); i++)
                            myweights[i] = weights[i].as<double>();

                    if (closed)
                    {
                        min_u = myknots[degree], max_u = myknots[myknots.size() - degree - 1];
                    }

                    MySurf* tmp = new MyRevolutionSurf(degree, loc, zvec, controls, myknots, myweights, closed, v_closed, min_u, max_u, min_v, max_v);
                    allsurfs.push_back(tmp);

                }
                else if (type == "Line")
                {
                    const auto& location = profile["location"];
                    const auto& vert_indices = profile["vert_indices"];
                    const auto& vert_parameters = profile["vert_parameters"];
                    const auto& direction = profile["direction"];
                    vec3 c_loc(location[0].as<double>(), location[1].as<double>(), location[2].as<double>());
                    vec3 dir(direction[0].as<double>(), direction[1].as<double>(), direction[2].as<double>());
                    vec3 start = c_loc + vert_parameters[0].as<double>() * dir;
                    vec3 end = c_loc + vert_parameters[vert_parameters.size() - 1].as<double>() * dir;

                    allsurfs.push_back(new MyRevolutionSurf(start, end, loc, zvec, v_closed, min_u, max_u, min_v, max_v));
                }
                else if (type == "Circle")
                {
                    const auto& location = profile["location"];
                    const auto& radius = profile["radius"].as<double>();
                    const auto& x_axis = profile["x_axis"];
                    const auto& y_axis = profile["y_axis"];
                    vec3 dirx(x_axis[0].as<double>(), x_axis[1].as<double>(), x_axis[2].as<double>());
                    vec3 diry(y_axis[0].as<double>(), y_axis[1].as<double>(), y_axis[2].as<double>());
                    bool is_closed = u_closed; //(vert_indices[0].as<int>() == vert_indices[vert_indices.size() - 1].as<int>());
                    vec3 c_loc(location[0].as<double>(), location[1].as<double>(), location[2].as<double>());

                    if (is_closed)
                    {
                        min_u = 0;
                        max_u = M_PI * 2;
                    }
                    
                    allsurfs.push_back(new MyRevolutionSurf(c_loc, dirx, diry, radius, is_closed, v_closed, loc, zvec, min_u, max_u, min_v, max_v));
                }
                else if (type == "Ellipse")
                {
                    const auto& location = profile["location"];
                    const auto& x_radius = profile["maj_radius"].as<double>();
                    const auto& y_radius = profile["min_radius"].as<double>();
                    const auto& x_axis = profile["x_axis"];
                    const auto& y_axis = profile["y_axis"];
                    vec3 dirx(x_axis[0].as<double>(), x_axis[1].as<double>(), x_axis[2].as<double>());
                    vec3 diry(y_axis[0].as<double>(), y_axis[1].as<double>(), y_axis[2].as<double>());

                    bool is_closed = u_closed; //(vert_indices[0].as<int>() == vert_indices[vert_indices.size() - 1].as<int>());
                    const auto& focus1 = profile["focus1"];
                    const auto& focus2 = profile["focus2"];
                    vec3 c_loc(focus1[0].as<double>() + focus2[0].as<double>(),
                        focus1[1].as<double>() + focus2[1].as<double>(),
                        focus1[2].as<double>() + focus2[2].as<double>());
                    c_loc *= 0.5;

                    if (is_closed)
                    {
                        min_u = 0;
                        max_u = M_PI * 2;
                    }
                    
                    allsurfs.push_back(new MyRevolutionSurf(c_loc, dirx, diry, x_radius, y_radius, is_closed, v_closed, loc, zvec, min_u, max_u, min_v, max_v));
                }
            }
            else if (type == "Extrusion")
            {
                const auto& profile = s["curve"];
                const string type = profile["type"].as<string>(); //"Bspline"
                const auto& direction = s["direction"];
                vec3 zvec(direction[0].as<double>(), direction[1].as<double>(), direction[2].as<double>());

                double max_u = 0.001 * valid_max_u, min_u = 0.001 * valid_min_u, max_v = valid_max_v, min_v = valid_min_v;

                u_scale = 0.001;

                bool u_closed = u_is_closed;

                if (type == "BSpline")
                {
                    const auto closed = profile["closed"].as<bool>();
                    const auto continuity = profile["continuity"].as<int>();
                    const auto degree = profile["degree"].as<int>();
                    const auto& knots = profile["knots"];
                    const auto& poles = profile["poles"];
                    const bool rational = profile["rational"].as<bool>();
                    const auto& weights = profile["weights"];

                    std::vector<double> myknots(knots.size());
                    for (size_t i = 0; i < knots.size(); i++)
                        myknots[i] = knots[i].as<double>();

                    std::vector<gte::Vector<3, double>> controls(poles.size());
                    std::vector<double> myweights(poles.size(), 1);
                    for (size_t i = 0; i < poles.size(); i++)
                    {
                        controls[i][0] = poles[i][0].as<double>();
                        controls[i][1] = poles[i][1].as<double>();
                        controls[i][2] = poles[i][2].as<double>();
                    }

                    if (rational)
                        for (size_t i = 0; i < weights.size(); i++)
                            myweights[i] = weights[i].as<double>();
                    if (closed)
                    {
                        min_u = myknots[degree], max_u = myknots[myknots.size() - degree - 1];
                    }
                    
                    allsurfs.push_back(new MyExtrusionSurf(degree, zvec, controls, myknots, myweights, closed, min_u, max_u, min_v, max_v));

                }
                else if (type == "Line")
                {
                    const auto& location = profile["location"];
                    const auto& vert_indices = profile["vert_indices"];
                    const auto& vert_parameters = profile["vert_parameters"];
                    const auto& direction = profile["direction"];
                    vec3 loc(location[0].as<double>(), location[1].as<double>(), location[2].as<double>());
                    vec3 dir(direction[0].as<double>(), direction[1].as<double>(), direction[2].as<double>());
                    vec3 start = loc + vert_parameters[0].as<double>() * dir;
                    vec3 end = loc + vert_parameters[vert_parameters.size() - 1].as<double>() * dir;
                    
                    allsurfs.push_back(new MyExtrusionSurf(start, end, zvec, min_u, max_u, min_v, max_v));
                }
                else if (type == "Circle")
                {
                    const auto& location = profile["location"];
                    const auto& radius = profile["radius"].as<double>();
                    const auto& x_axis = profile["x_axis"];
                    const auto& y_axis = profile["y_axis"];
                    vec3 dirx(x_axis[0].as<double>(), x_axis[1].as<double>(), x_axis[2].as<double>());
                    vec3 diry(y_axis[0].as<double>(), y_axis[1].as<double>(), y_axis[2].as<double>());
                    bool is_closed = u_closed; //(vert_indices[0].as<int>() == vert_indices[vert_indices.size() - 1].as<int>());
                    vec3 loc(location[0].as<double>(), location[1].as<double>(), location[2].as<double>());

                    if (is_closed)
                    {
                        min_u = 0;
                        max_u = M_PI * 2;
                    }

                    allsurfs.push_back(new MyExtrusionSurf(loc, dirx, diry, radius, is_closed, zvec, min_u, max_u, min_v, max_v));
                    
                }
                else if (type == "Ellipse")
                {
                    const auto& location = profile["location"];
                    const auto& x_radius = profile["maj_radius"].as<double>();
                    const auto& y_radius = profile["min_radius"].as<double>();
                    const auto& x_axis = profile["x_axis"];
                    const auto& y_axis = profile["y_axis"];
                    vec3 dirx(x_axis[0].as<double>(), x_axis[1].as<double>(), x_axis[2].as<double>());
                    vec3 diry(y_axis[0].as<double>(), y_axis[1].as<double>(), y_axis[2].as<double>());

                    bool is_closed = u_closed; // double check
                    const auto& focus1 = profile["focus1"];
                    const auto& focus2 = profile["focus2"];
                    vec3 loc(focus1[0].as<double>() + focus2[0].as<double>(),
                        focus1[1].as<double>() + focus2[1].as<double>(),
                        focus1[2].as<double>() + focus2[2].as<double>());
                    loc *= 0.5;

                    if (is_closed)
                    {
                        min_u = 0;
                        max_u = M_PI * 2;
                    }
                    
                    allsurfs.push_back(new MyExtrusionSurf(loc, dirx, diry, x_radius, y_radius, is_closed, zvec, min_u, max_u, min_v, max_v));
                }
            }
            else if (type == "BSpline")
            {
                const auto u_rational = s["u_rational"].as<bool>();
                const auto v_rational = s["v_rational"].as<bool>();
                const auto u_closed = s["u_closed"].as<bool>();
                const auto v_closed = s["v_closed"].as<bool>();
                const auto continuity = s["continuity"].as<int>();
                const auto u_degree = s["u_degree"].as<int>();
                const auto v_degree = s["v_degree"].as<int>();
                const auto& poles = s["poles"];
                const auto& u_knots = s["u_knots"];
                const auto& v_knots = s["v_knots"];
                const auto weights = s["weights"];

                std::vector<double> my_uknots(u_knots.size());
                for (size_t i = 0; i < u_knots.size(); i++)
                    my_uknots[i] = u_knots[i].as<double>();
                std::vector<double> my_vknots(v_knots.size());
                for (size_t i = 0; i < v_knots.size(); i++)
                    my_vknots[i] = v_knots[i].as<double>();

                std::vector<gte::Vector<3, double>> controls(poles.size() * poles[0].size());
                std::vector<double> myweights(controls.size(), 1);

                for (size_t i = 0; i < poles.size(); i++)
                {
                    for (size_t j = 0; j < poles[i].size(); j++)
                    {
                        controls[i + j * poles.size()][0] = poles[i][j][0].as<double>();
                        controls[i + j * poles.size()][1] = poles[i][j][1].as<double>();
                        controls[i + j * poles.size()][2] = poles[i][j][2].as<double>();
                    }
                }

                if (u_rational || v_rational)
                {
                    for (size_t i = 0; i < weights.size(); i++)
                    {
                        for (size_t j = 0; j < weights[i].size(); j++)
                        {
                            myweights[i + j * weights.size()] = weights[i][j].as<double>();
                        }
                    }
                }

                double max_u = 0.001 * valid_max_u, min_u = 0.001 * valid_min_u, max_v = 0.001 * valid_max_v, min_v = 0.001 * valid_min_v;
                u_scale = 0.001, v_scale = 0.001;

                if (u_is_closed)
                {
                    min_u = my_uknots[u_degree], max_u = my_uknots[my_uknots.size() - u_degree - 1];
                }

                if (v_is_closed)
                {
                    min_v = my_vknots[v_degree], max_v = my_vknots[my_vknots.size() - v_degree - 1];
                }

                allsurfs.push_back(new MySplineSurf(
                    u_degree, v_degree, controls, my_uknots, my_vknots,
                    myweights, u_closed, v_closed, min_u, max_u, min_v, max_v));
            }
            else if (type == "Other")
            {
                //put a placeholder in all_surfs
                allsurfs.push_back(new MyPlane());
            }

            PointSet vert_pointset;
            for (size_t i = 0; i < vert_parameters.size(); i++)
            {
                double u = u_scale * vert_parameters[i][0].as<double>();
                double v = v_scale * vert_parameters[i][1].as<double>();
                vec3 P, PN;
                
                if (type != "Other")
                {
                    P = allsurfs.back()->GetPosition(u, v);
                    PN = allsurfs.back()->GetNormal(u, v);
                }

                vert_pointset.push(P);
                vert_pointset.push_normal(PN);
            }

            vert_pointset.build();
            std::vector<size_t> selected_facets(face_indices.size());
            for (size_t i = 0; i < face_indices.size(); i++)
            {
                selected_facets[i] = face_indices[i].as<size_t>();
            }
            
            size_t fi_id = face_info_for_resampling.size();
            std::vector<size_t> onefis;
            for (size_t i = 0; i < selected_facets.size(); i++)
            {
                face_info fi;
                fi.surf_type = type; //including other
                fi.face_index = selected_facets[i];

                fi.surf_index = (int)allsurfs.size() - 1;

                const size_t& fid = selected_facets[i];
                for (int j = 0; j < facets[fid].size(); j++)
                {
                    auto vid = vert_pointset.get_nearest_point_id(vertices[facets[fid][j]]);
                    //check distance
                    double scaled_dist = (vertices[facets[fid][j]] - vert_pointset.points[vid]).Length() * mesh_scale;
                    if (scaled_dist > max_point_mesh_dist)
                    {
                        std::cout << "!!!parametric and mesh not consistent" << std::endl;
                        flag_distclose = false;
                        break;
                    }

                    fi.u.push_back(u_scale * vert_parameters[vid][0].as<double>());
                    fi.v.push_back(v_scale * vert_parameters[vid][1].as<double>());
                    fi.para_normals.push_back(vert_pointset.normals[vid]);
                }
                if (!flag_distclose)
                {
                    break;
                }

                onefis.push_back(face_info_for_resampling.size());
                face_info_for_resampling.push_back(fi);

            }
            
            patch2fis.push_back(onefis);

            //update uv range according to patch type, not considering extrusion and revolution
            if (type != "Other")
            {
                int counter_same_normal = 0, counter_reverse_normal = 0;
                for (size_t i = 0; i < selected_facets.size(); i++)
                {
                    //update u
                    if (type == "Cylinder" || type == "Cone" || type == "Sphere" || type == "Torus")
                    {
                        double maxu = *std::max_element(face_info_for_resampling[fi_id + i].u.begin(), face_info_for_resampling[fi_id + i].u.end());
                        double minu = *std::min_element(face_info_for_resampling[fi_id + i].u.begin(), face_info_for_resampling[fi_id + i].u.end());
                        if (maxu - minu > 2 * M_PI - M_PI / 12.0)
                        {
                            for (size_t j = 0; j < face_info_for_resampling[fi_id + i].u.size(); j++)
                            {
                                if (maxu - face_info_for_resampling[fi_id + i].u[j] > M_PI / 12.0)
                                {
                                    face_info_for_resampling[fi_id + i].u[j] += 2 * M_PI;
                                }
                            }
                        }
                    }

                    //update v
                    if (type == "Sphere" || type == "Torus")
                    {
                        double maxv = *std::max_element(face_info_for_resampling[fi_id + i].v.begin(), face_info_for_resampling[fi_id + i].v.end());
                        double minv = *std::min_element(face_info_for_resampling[fi_id + i].v.begin(), face_info_for_resampling[fi_id + i].v.end());
                        if (maxv - minv > 2 * M_PI - M_PI / 12.0)
                        {
                            for (size_t j = 0; j < face_info_for_resampling[fi_id + i].v.size(); j++)
                            {
                                if (maxv - face_info_for_resampling[fi_id + i].v[j] > M_PI / 12.0)
                                {
                                    face_info_for_resampling[fi_id + i].v[j] += 2 * M_PI;
                                }
                            }
                        }
                    }

                    //spline not impl yet, when u is closed, u_min, u_max not necessarily 0,1
                    //check normal
                    vec3d para_normal(0.0, 0.0, 0.0);
                    for (size_t j = 0; j < face_info_for_resampling[fi_id + i].para_normals.size(); j++)
                    {
                        para_normal += face_info_for_resampling[fi_id + i].para_normals[j];
                    }
                    para_normal.Normalize();
                    vec3d face_normal = mesh.get_face(face_info_for_resampling[fi_id + i].face_index)->normal;
                    if (para_normal.Dot(face_normal) < 0.0)
                    {
                        counter_reverse_normal += 1;
                    }
                    else
                    {
                        counter_same_normal += 1;
                    }
                }

                //update normal info
                bool reverse_normal = false;
                if (counter_reverse_normal > counter_same_normal)
                {
                    reverse_normal = true;
                }
                for (size_t i = 0; i < selected_facets.size(); i++)
                {
                    face_info_for_resampling[fi_id + i].reverse_normal = reverse_normal;
                }
                
            }
            

            if (!flag_distclose)
            {
                break;
            }
        }


        //face_info_size should be the same with face size
        if (face_info_for_resampling.size() != facets.size())
        {
            std::cout << "face info size not equal to face size" << std::endl;
            flag_idvalid = false;
        }

        if (with_spline)
        {
            std::ofstream ofs(filenamewithoutext + ".withspline");
            ofs.close();
        }

        if (!flag_idvalid || !flag_distclose || !flag_paramvalid)
        {
            std::cout << "flag idvalid: " << flag_idvalid << " flag_distclose: " << flag_distclose << " flag_paramvalid: " << flag_paramvalid << std::endl;
            std::ofstream ofs(filenamewithoutext + ".fail");
            ofs.close();
            return 1;
        }

        assert(surf_types.size() == allsurfs.size());

        //sampling
        std::vector<TinyVector<double, 3>> vert_pos;
        for (size_t i = 0; i < mesh.get_vertices_list()->size(); i++)
        {
            vert_pos.push_back(mesh.get_vertices_list()->at(i)->pos);
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
            //mesh.get_faces_list()->at(fid)->normal;
            tri_normals.push_back(mesh.get_faces_list()->at(i)->normal);
        }


        std::vector<vec3d> output_pts, output_normals;
        std::vector<int> output_pts_tris; //which triangle a point belongs to
        //first sample at least min_pts_per_patch pts from each patch
        for (size_t i = 0; i < patch2fis.size(); i++)
        {
            std::vector<face_info> cur_fis;
            std::vector<TinyVector<size_t, 3>> cur_faces;
            std::vector<vec3d> cur_face_normals;
            for (auto fiid : patch2fis[i])
            {
                cur_fis.push_back(face_info_for_resampling[fiid]);
                int fid = face_info_for_resampling[fiid].face_index;
                cur_faces.push_back(tri_verts[fid]);
                cur_face_normals.push_back(tri_normals[fid]);
            }

            std::vector<vec3d> cur_pts, cur_normals;
            std::vector<int> cur_pts_tris;
            sample_pts_from_mesh_parametric(vert_pos, cur_faces, cur_face_normals, min_sample_perpatch, allsurfs, cur_fis, cur_pts, cur_normals, cur_pts_tris);
            output_pts.insert(output_pts.end(), cur_pts.begin(), cur_pts.end());
            output_normals.insert(output_normals.end(), cur_normals.begin(), cur_normals.end());
            output_pts_tris.insert(output_pts_tris.end(), cur_pts_tris.begin(), cur_pts_tris.end());
        }

        //sample the rest
        if (output_pts.size() < n_nonfeature_sample)
        {
            std::vector<vec3d> cur_pts, cur_normals;
            std::vector<int> cur_pts_tris;
            sample_pts_from_mesh_parametric(vert_pos, tri_verts, tri_normals, n_nonfeature_sample - output_pts.size(), allsurfs, face_info_for_resampling, cur_pts, cur_normals, cur_pts_tris);
            output_pts.insert(output_pts.end(), cur_pts.begin(), cur_pts.end());
            output_normals.insert(output_normals.end(), cur_normals.begin(), cur_normals.end());
            output_pts_tris.insert(output_pts_tris.end(), cur_pts_tris.begin(), cur_pts_tris.end());
        }

        //check distance from output pts to mesh, if it is larger than the given threshold, then skip
        Eigen::MatrixXd aabb_V;
        Eigen::MatrixXi aabb_F;
        aabb_V.resize(vert_pos.size(), 3);
        aabb_F.resize(tri_verts.size(), 3);
        for (size_t i = 0; i < vert_pos.size(); i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                aabb_V(i, j) = vert_pos[i][j];
            }
        }

        for (size_t i = 0; i < tri_verts.size(); i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                aabb_F(i, j) = tri_verts[i][j];
            }
        }
        igl::AABB<Eigen::MatrixXd, 3>* aabb_tree = new igl::AABB<Eigen::MatrixXd, 3>();
        aabb_tree->init(aabb_V, aabb_F);

        Eigen::VectorXi I;
        Eigen::MatrixXd C;
        Eigen::VectorXd sqrD;
        Eigen::MatrixXd P(output_pts.size(), 3);
        for (size_t i = 0; i < output_pts.size(); i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                P(i, j) = output_pts[i][j];
            }
        }
        aabb_tree->squared_distance(aabb_V, aabb_F, P, sqrD, I, C);
        
        double max_dist = std::sqrt(sqrD.maxCoeff()) * mesh_scale;
        if (max_dist > max_point_mesh_dist)
        {
            std::cout << "point to mesh distance too large" << std::endl;
            std::ofstream ofs(filenamewithoutext + ".fail");
            ofs.close();
            return 1;
        }

        std::ofstream outputsamples(filenamewithoutext + "_50k.xyz");
        
        for (size_t i = 0; i < output_pts.size(); i++)
        {
            outputsamples << (output_pts[i] - mesh_center) * mesh_scale << " " << output_normals[i] << std::endl;
        }

        outputsamples.close();

        //save tris
        outputsamples.open(filenamewithoutext + "_50k_tris.txt");
        for (size_t i = 0; i < output_pts_tris.size(); i++)
        {
            outputsamples << output_pts_tris[i] << std::endl;
        }

        outputsamples.close();
        

        //output sampling face
        std::string samplingoutput = filenamewithoutext + "_sampling.xyz";
        std::ofstream samplefile(samplingoutput);
        for (const auto& fi : face_info_for_resampling)
        {
            const auto& type = fi.surf_type;
            const auto face_index = fi.face_index;
            const auto surf_index = fi.surf_index;
            const bool reverse_normal = fi.reverse_normal;

            double au = 0, av = 0;

            for (size_t k = 0; k < fi.u.size(); k++)
            {
                au += fi.u[k], av += fi.v[k];
            }
            if (!fi.u.empty())
            {
                au /= fi.u.size(), av /= fi.v.size();
            }
            vec3 p, pn;
            
            if (type != "Other")
            {
                p = allsurfs[surf_index]->GetPosition(au, av);
                pn = allsurfs[surf_index]->GetNormal(au, av);
                if (reverse_normal)
                {
                    pn = -pn;
                }
            }

            samplefile << p << ' ' << pn << std::endl;
        }
        samplefile.close();
        
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << "Error parsing options: " << e.what() << std::endl;
        exit(1);
    }

    return 0;
}