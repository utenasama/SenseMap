#include "utils_sampling.hpp"

#include "vcg_mesh.hpp"

#include <list>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>
#include <vector>
#include <boost/tuple/tuple.hpp>

// =============================================================================
namespace Poisson_sampling {
// =============================================================================

#define MAX_LINE_LENGTH 512

void poisson_disk(float radius,
                  int nb_samples,
                  const std::vector<Vec3>& verts,
                  const std::vector<Vec3>& nors,
                  const std::vector<int>& tris,
                  const std::vector<int>& labels,
                  std::vector<Vec3>& samples_pos,
                  std::vector<Vec3>& samples_nor,
                  std::vector<int>& samples_lab)
{
    assert(verts.size() == nors.size());
    assert(verts.size() > 0);
    assert(tris.size() > 0);

    vcg::MyMesh vcg_mesh, sampler;
    vcg_mesh.concat((float*)&(verts[0]), (int*)&(tris[0]), verts.size(), tris.size()/3);
    vcg_mesh.set_normals( (float*)&(nors[0]) );
    vcg_mesh.set_labels(labels.data());
    vcg_mesh.update_bb();
    
    vcg::MyAlgorithms::Poison_setup pp;
    pp._radius = radius;
    pp._nb_samples = nb_samples;
    pp._approx_geodesic_dist = false;
    vcg::MyAlgorithms::poison_disk_sampling(vcg_mesh, pp, sampler);

    const int nb_vert = sampler.vert.size();
    samples_pos.clear();
    samples_nor.clear();
    samples_pos.resize( nb_vert );
    samples_nor.resize( nb_vert );
    samples_lab.resize( nb_vert );

    vcg::MyMesh::VertexIterator vi = sampler.vert.begin();
    for(int i = 0; i < nb_vert; ++i, ++vi)
    {
        vcg::MyMesh::CoordType  p = (*vi).P();
        vcg::MyMesh::NormalType n = (*vi).N();
        samples_pos[i] = Vec3(p.X(), p.Y(), p.Z());
        samples_nor[i] = Vec3(n.X(), n.Y(), n.Z());
        samples_nor[i].normalize();
        samples_lab[i] = (*vi).L();
    }

}

void poisson_disk(float radius,
                  int nb_samples,
                  const std::vector<Vec3>& verts,
                  const std::vector<Vec3>& nors,
                  const std::vector<int>& tris,
                  std::vector<Vec3>& samples_pos,
                  std::vector<Vec3>& samples_nor)
{
    std::vector<int> labels(verts.size(), 0);
    std::vector<int> samples_lab;

    poisson_disk(radius, nb_samples, verts, nors, tris, labels, samples_pos, samples_nor, samples_lab);
}


bool load_mesh(const std::string &filename, 
                std::vector<Vec3>& vertexs,
                std::vector<Vec3>& normals,
                std::vector<int>& labels,
                std::vector<int>& faces,
                bool &has_sem,
                bool &has_rgb,
                bool &has_tex) 
{
    has_sem = has_rgb = has_tex = false;

    FILE *file = fopen(filename.c_str(), "r");
	if (!file) {
		std::cerr << "File does not exist! Please check the file path "
				  << filename << std::endl;
		return false;
	}
    char line_buf[MAX_LINE_LENGTH];
    while (fgets(line_buf, MAX_LINE_LENGTH, file))
    {
        if (!strncmp(line_buf, "v ", 2))
        {
            float vertex[3];
            float vertex_color[3];
            int vertex_label;

            int n = sscanf(line_buf + 2, "%f %f %f %f %f %f %d",
                    &vertex[0], &vertex[1], &vertex[2], 
                    &vertex_color[0], &vertex_color[1], &vertex_color[2],
					&vertex_label);
            if (n == 7) {
                has_sem = true;
                has_rgb = true;
                sscanf(line_buf + 2, "%f %f %f %f %f %f %d",
                    &vertex[0], &vertex[1], &vertex[2], 
                    &vertex_color[0], &vertex_color[1], &vertex_color[2],
					&vertex_label);
			} else if (n == 6) {
                has_rgb = true;
                has_sem = false;
                sscanf(line_buf + 2, "%f %f %f %f %f %f",
                    &vertex[0], &vertex[1], &vertex[2], 
                    &vertex_color[0], &vertex_color[1], &vertex_color[2]);
            } else {
                has_rgb = false;
                has_sem = false;
                sscanf(line_buf + 2, "%f %f %f",
                    &vertex[0], &vertex[1], &vertex[2]);
            }
            vertexs.push_back(Vec3(vertex[0], vertex[1], vertex[2]));
            labels.push_back(vertex_label);
        }
        if (!strncmp(line_buf, "vt ", 3)) {
            has_tex = true;
        }
        if (!strncmp(line_buf, "vn ", 3))
        {
            float vertex_norm[3];
            sscanf(line_buf + 3, "%f %f %f",
                   &vertex_norm[0], &vertex_norm[1], &vertex_norm[2]);
            normals.push_back(Vec3(vertex_norm[0], vertex_norm[1], vertex_norm[2]));
        }
        if (!strncmp(line_buf, "f ", 2))
        {
            int face[3];
            int tex[3];
            char c;
            // f v/vt/vn 
            if (has_tex && !normals.empty()) {
                sscanf(line_buf + 2, "%d %c %d %c %d %d %c %d %c %d %d %c %d %c %d",
					&face[0], &c, &tex[0], &c, &face[0],
					&face[1], &c, &tex[1], &c, &face[1],
					&face[2], &c, &tex[2], &c, &face[2]);
            }
            // f v // vn
			else if (normals.size() != 0) {
				sscanf(line_buf + 2, "%d %c %c %d %d %c %c %d %d %c %c %d",
					&face[0], &c, &c, &face[0],
					&face[1], &c, &c, &face[1],
					&face[2], &c, &c, &face[2]);
            // f v
			} else {
				sscanf(line_buf + 2, "%d %d %d", &face[0], &face[1], &face[2]);
			}
            face[0] -= 1;
            face[1] -= 1;
            face[2] -= 1;
            faces.push_back(face[0]);
            faces.push_back(face[1]);
            faces.push_back(face[2]);
        }
    }
    fclose(file);
    std::cout << "vtxs.size(): " << vertexs.size() << std::endl;
    std::cout << "faces.size(): " << faces.size() / 3 << std::endl;
    return true;      
}

// Get Vertex and Face Norms
void get_face_normals(
                    std::vector<Vec3> &FaceNorms,
                    const std::vector<Vec3> &Vtxs,
                    const std::vector<int> &Faces)
{
    int iFaceNum = Faces.size() / 3;
    std::vector<Vec3>(iFaceNum).swap(FaceNorms);
    for (int i = 0; i < iFaceNum; ++ i)
    {
        int f0 = Faces[3 * i + 0];
        int f1 = Faces[3 * i + 1];
        int f2 = Faces[3 * i + 2];


        const Vec3 &V1 = Vtxs[f0];
        const Vec3 &V2 = Vtxs[f1];
        const Vec3 &V3 = Vtxs[f2];

        FaceNorms[i] = ((V2 - V1).cross(V3 - V1));
        FaceNorms[i].normalize();
    }
}

void get_vtx_normals(
                std::vector<Vec3> &VtxNorms,
                const std::vector<Vec3> &FaceNorms,
                const std::vector<Vec3> &Vtxs,
                const std::vector<int> &Faces)
    {
        int iVtxNum = Vtxs.size();
        std::vector<Vec3>(iVtxNum).swap(VtxNorms);
        std::vector<std::list<int> > ORNs(iVtxNum);
        int iFaceNum = Faces.size() / 3;
        for (int i = 0; i < iFaceNum; i++)
        {
            int f0 = Faces[3 * i + 0];
            int f1 = Faces[3 * i + 1];
            int f2 = Faces[3 * i + 2];
            
            ORNs[f0].emplace_back(i);
            ORNs[f1].emplace_back(i);
            ORNs[f2].emplace_back(i);
        }
        for (int i = 0; i < iVtxNum; i++)
        {
            const std::list<int> &ORN = ORNs[i];
            Vec3 &VtxNorm = VtxNorms[i];
            VtxNorm.x = VtxNorm.y = VtxNorm.z = 0;
            for (std::list<int>::const_iterator Iter = ORN.begin(); Iter != ORN.end(); Iter++)
            {
                VtxNorm += FaceNorms[*Iter];
            }
            VtxNorm.normalize();
        }
    }

void get_normals(
    std::vector<Vec3> &v_norms,
    std::vector<Vec3> &f_norms,
    const std::vector<Vec3> &v,
    const std::vector<int> &f)
{
    get_face_normals(f_norms, v, f);
    get_vtx_normals(v_norms, f_norms, v, f);
}

double pcd_average_spacing(const std::vector<Vec3> &verts)
{
    // Types
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef Kernel::FT FT;
    typedef Kernel::Point_3 Point;
    // Data type := index, followed by the point, followed by three integers that
    // define the Red Green Blue color of the point.
    typedef boost::tuple<int, Point> IndexedPointWithColorTuple;
    // Concurrency
    typedef CGAL::Parallel_if_available_tag Concurrency_tag;

    std::vector<IndexedPointWithColorTuple> point_set;
    IndexedPointWithColorTuple tuple_p;
    for (int i = 0; i < verts.size(); ++ i) {
        tuple_p.get<0>() = i;
        tuple_p.get<1>() = Point(verts[i].x, verts[i].y, verts[i].z);
        point_set.emplace_back(tuple_p);
    }

    // Computes average spacing.
    const unsigned int nb_neighbors = 6; // 1 ring
    FT average_spacing = CGAL::compute_average_spacing<Concurrency_tag>(
                            point_set, nb_neighbors,
                            CGAL::parameters::point_map(CGAL::Nth_of_tuple_property_map<1,IndexedPointWithColorTuple>()));

    return average_spacing;
}

}// END UTILS_SAMPLING NAMESPACE ===============================================
