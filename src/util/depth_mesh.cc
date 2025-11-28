#include "depth_mesh.h"
#include <map>
#include <queue>
#include <list>
#include <string.h>
#include <stdio.h>

#define DRAW_FRAME_BUFFER_BLEND 1

namespace sensemap {

void GeometricCheck(std::vector<unsigned char> &geo_cons_map, const float *depth_buf, int stride, int width, int height, int step = 1, float depth_thres = 0.01f, float min_depth = 0.4f, float max_depth = 3.0f) {
    const int step_height = (height - 1) / step + 1;
    const int step_width = (width - 1) / step + 1;
    depth_thres *= step;

    geo_cons_map.resize(step_height * step_width);
    std::fill(geo_cons_map.begin(), geo_cons_map.end(), 255);
    float depth, depth_disc_thres, depth_n;
    for (int y = step, sy = 0; y + step < height; y += step, ++sy) {
        for (int x = step, sx = 0; x + step < width; x += step, ++sx) {
            unsigned char &geo_consist = geo_cons_map[sy * step_width + sx];
            if (!geo_consist)
                continue;

            depth = depth_buf[y * stride + x];
            if (depth < min_depth || depth > max_depth) {
                geo_consist = 0;
                continue;
            }

            depth_disc_thres = depth_thres * depth;
            // left
            depth_n = depth_buf[y * stride + x - step];
            if (depth_n > 0 && std::abs(depth_n - depth) >= depth_disc_thres) {
                geo_consist = 0;
                continue;
            }
            // right
            depth_n = depth_buf[y * stride + x + step];
            if (depth_n > 0 && std::abs(depth_n - depth) >= depth_disc_thres) {
                geo_consist = 0;
                continue;
            }
            // top
            depth_n = depth_buf[(y - step) * stride + x];
            if (depth_n > 0 && std::abs(depth_n - depth) >= depth_disc_thres) {
                geo_consist = 0;
                continue;
            }
            // bottom
            depth_n = depth_buf[(y + step) * stride + x];
            if (depth_n > 0 && std::abs(depth_n - depth) >= depth_disc_thres) {
                geo_consist = 0;
                continue;
            }
            // lefttop
            depth_n = depth_buf[(y - step) * stride + x - step];
            if (depth_n > 0 && std::abs(depth_n - depth) >= depth_disc_thres) {
                geo_consist = 0;
                continue;
            }
            // leftbottom
            depth_n = depth_buf[(y + step) * stride + x - step];
            if (depth_n > 0 && std::abs(depth_n - depth) >= depth_disc_thres) {
                geo_consist = 0;
                continue;
            }
            // righttop
            depth_n = depth_buf[(y - step) * stride + x + step];
            if (depth_n > 0 && std::abs(depth_n - depth) >= depth_disc_thres) {
                geo_consist = 0;
                continue;
            }
            // rightbottom
            depth_n = depth_buf[(y + step) * stride + x + step];
            if (depth_n > 0 && std::abs(depth_n - depth) >= depth_disc_thres) {
                geo_consist = 0;
                continue;
            }
        }
    }
}

void GetMaxMeshSize(int &max_vtx_size, int &max_facet_size, int width, int height, int step, VertexFormat vertex_format)
{
    const int step_width = (width - 1) / step + 1;
    const int step_height = (height - 1) / step + 1;

    max_vtx_size = step_width * step_height * vertex_format;
    max_facet_size = (step_width - 1) * (step_height - 1) * 3 << 1;
}

bool WriteTriangleMeshObj(const std::string &filename, 
                          const DepthMesh &mesh) {
    printf("[Writing Triangle Mesh (.Obj) ...");
    FILE *obj_file = fopen(filename.c_str(), "w");

    for(int i = 0; i < mesh.vertices_.size(); ++i){
        fprintf(obj_file, "v %f %f %f\n",
                mesh.vertices_[i](0), mesh.vertices_[i](1),
                mesh.vertices_[i](2));
    }


    for(int i = 0; i < mesh.faces_.size(); ++i){
        fprintf(obj_file, "f %d %d %d\n",
                mesh.faces_[i](0) + 1, mesh.faces_[i](1) + 1,
                mesh.faces_[i](2) + 1);
    }

    fclose(obj_file);
    printf("done]\n");
    return true;
}

bool WarpFrameBuffer(cv::Mat& depth_buffer, 
                     const std::vector<Eigen::Vector3f> & vertices, 
                     const std::vector<Eigen::Vector3i> & faces, 
                     float min_depth, float max_depth,
                     int cull_back_face
) {
    const int height = depth_buffer.rows;
    const int width = depth_buffer.cols;
    depth_buffer.setTo(std::numeric_limits<float>::max());

    const int n_faces = faces.size();
    for (int i = 0; i < n_faces; i++) {
        const Eigen::Vector3f & p0 = vertices[faces[i][0]];
        const Eigen::Vector3f & p1 = vertices[faces[i][1]];
        const Eigen::Vector3f & p2 = vertices[faces[i][2]];
        if (p0.x() < 0.0f && p1.x() < 0.0f && p2.x() < 0.0f) continue;
        if (p0.x() >= width && p1.x() >= width && p2.x() >= width) continue;
        if (p0.y() < 0.0f && p1.y() < 0.0f && p2.y() < 0.0f) continue;
        if (p0.y() >= height && p1.y() >= height && p2.y() >= height) continue;
        if (p0.z() > max_depth && p1.z() > max_depth && p2.z() > max_depth) continue;
        if (p0.z() < min_depth || p1.z() < min_depth || p2.z() < min_depth) continue;
        if (p0.z() <= 0.0f || p1.z() <= 0.0f || p2.z() <= 0.0f) continue;

        const int umin = std::max(0,          (int)std::floor(std::min(std::min(p0.x(), p1.x()), p2.x())));
        const int umax = std::min(width - 1,  (int)std::ceil (std::max(std::max(p0.x(), p1.x()), p2.x())));
        const int vmin = std::max(0,          (int)std::floor(std::min(std::min(p0.y(), p1.y()), p2.y())));
        const int vmax = std::min(height - 1, (int)std::ceil (std::max(std::max(p0.y(), p1.y()), p2.y())));

        const double uminf = umin;
        const double vminf = vmin;
        const double pa_y = vminf - p0.y();
        const double pb_y = vminf - p1.y();
        const double pc_y = vminf - p2.y();
        const double pa_x = uminf - p0.x();
        const double pb_x = uminf - p1.x();
        const double pc_x = uminf - p2.x();
        const double delta_ab_y = pb_y - pa_y;
        const double delta_bc_y = pc_y - pb_y;
        const double delta_ca_y = pa_y - pc_y;
        const double delta_ab_x = pa_x - pb_x;
        const double delta_bc_x = pb_x - pc_x;
        const double delta_ca_x = pc_x - pa_x;
        double cross_ab0 = pa_x * pb_y - pb_x * pa_y;
        double cross_bc0 = pb_x * pc_y - pc_x * pb_y;
        double cross_ca0 = pc_x * pa_y - pa_x * pc_y;
        double detA = cross_ab0 + cross_bc0 + cross_ca0;
        float back_face = 1.0f;
        if (cull_back_face) {
            if (std::abs(detA) < std::numeric_limits<double>::epsilon()) continue;
            if (cull_back_face > 0 && detA < 0.0f) back_face *= -1;
            else if (cull_back_face < 0 && detA > 0.0f) back_face *= -1;
        } else {
            if (detA > -std::numeric_limits<double>::epsilon()) continue;
        }

        #if DRAW_FRAME_BUFFER_BLEND
        double inv_detA = 1.0 / detA;
        double inv_A00 = delta_bc_y * inv_detA;
        double inv_A10 = delta_ca_y * inv_detA;
        double inv_A20 = delta_ab_y * inv_detA;
        double inv_A01 = delta_bc_x * inv_detA;
        double inv_A11 = delta_ca_x * inv_detA;
        double inv_A21 = delta_ab_x * inv_detA;
        double inv_A02 = cross_bc0 * inv_detA - uminf * inv_A00 - vminf * inv_A01;
        double inv_A12 = cross_ca0 * inv_detA - uminf * inv_A10 - vminf * inv_A11;
        double inv_A22 = cross_ab0 * inv_detA - uminf * inv_A20 - vminf * inv_A21;
        Eigen::Vector3d x(
            inv_A00 * p0.z() + inv_A10 * p1.z() + inv_A20 * p2.z(),
            inv_A01 * p0.z() + inv_A11 * p1.z() + inv_A21 * p2.z(),
            inv_A02 * p0.z() + inv_A12 * p1.z() + inv_A22 * p2.z()
        );

        // NOTE: double precision is preferred
        // Eigen::Matrix3d A;
        // A(0, 0) = p0.x(); A(0, 1) = p0.y(); A(0, 2) = 1;
        // A(1, 0) = p1.x(); A(1, 1) = p1.y(); A(1, 2) = 1;
        // A(2, 0) = p2.x(); A(2, 1) = p2.y(); A(2, 2) = 1;
        // Eigen::Vector3d b(p0.z(), p1.z(), p2.z());
        // Eigen::Vector3d x = A.inverse() * b;
        #else
        float z = (p0.z() + p1.z() + p2.z()) / 3.0;
        #endif
        for (
            int v = vmin; v <= vmax; v++, 
            cross_ab0 += delta_ab_x, cross_bc0 += delta_bc_x, cross_ca0 += delta_ca_x
        ) {
            double cross_ab = cross_ab0;
            double cross_bc = cross_bc0;
            double cross_ca = cross_ca0;
            #if DRAW_FRAME_BUFFER_BLEND
            double z0 = x[2] + x[1] * v;
            #endif
            for (
                int u = umin; u <= umax; u++, 
                cross_ab += delta_ab_y, cross_bc += delta_bc_y, cross_ca += delta_ca_y
            ) {
                if (cross_ab * cross_bc >= 0 && cross_bc * cross_ca >= 0) {
                    // is inside triangle
                    #if DRAW_FRAME_BUFFER_BLEND
                    float z = z0 + x[0] * u;
                    #endif
                    if (z < std::abs(depth_buffer.at<float>(v, u))) {
                        // back faces are marked as "negative" depths
                        // and will be filtered finally
                        depth_buffer.at<float>(v, u) = back_face * z;
                    }
                }
            }
        }
    }

    for (int y = 0; y < depth_buffer.rows; y++) {
        for (int x = 0; x < depth_buffer.cols; x++) {
            float & z = depth_buffer.at<float>(y, x);
            if (z == std::numeric_limits<float>::max()) {
                z = 0.0f;
            } else if (z < 0.0f) {
                z = 0.0f;
            } else if (z > max_depth) {
                z = 0.0f;
            } else if (z < min_depth) {
                z = 0.0f;
            }
        }
    }

    return true;
}

bool WarpFrameBuffer(cv::Mat& depth_buffer_, 
                     const Eigen::Matrix3f& K, const Eigen::Matrix3f& R, const Eigen::Vector3f& t, 
                     DepthMesh* pGLObj) {
    if (pGLObj == nullptr || pGLObj == nullptr) return false;
    std::vector<Eigen::Vector3f> vertices;
    vertices.resize(pGLObj->vertices_.size());

    const double fx = K(0, 0);
    const double cx = K(0, 2);
    const double fy = K(1, 1);
    const double cy = K(1, 2);
    const int n_vertices = vertices.size();
    for (int i = 0; i < n_vertices; i++) {
        Eigen::Vector3f p = pGLObj->vertices_[i];
        p = R * p + t;
        p.x() = fx * p.x() / p.z() + cx;
        p.y() = fy * p.y() / p.z() + cy;
        
        vertices[i] = p.cast<float>();
    }

    return WarpFrameBuffer(depth_buffer_, vertices, pGLObj->faces_);
}

bool GenerateMesh(DepthMesh& pGLObj, 
                const float *depth_buf, int stride, int width, int height,
                const Eigen::Matrix3f &K, const Eigen::Matrix3f &R, const Eigen::Vector3f &T,
                int step, float depth_thres, VertexFormat vertex_format,
                float min_depth, float max_depth) {
    if (!depth_buf || width < 1 || height < 1)
        return false;

    const int step_width = (width - 1) / step + 1;
    const int step_height = (height - 1) / step + 1;

    Eigen::Matrix3f Kinv = K.inverse();

    Eigen::Matrix3f Q_w_c = R.transpose();
    Eigen::Vector3f T_w_c = -Q_w_c * T;

    std::vector<int> grid_to_cur_vert_idx(step_height * step_width, -1);
    std::vector<unsigned char> geo_cons_map;
    GeometricCheck(geo_cons_map, depth_buf, stride, width, height, step, depth_thres, min_depth, max_depth);

    Eigen::Vector3f Xc, Xw;
    Eigen::Vector3i facet;
    int gridId, vertId;
    float depth;
    int vtx_size = vertex_format;
    int vtx_num = 0;
    //float *vtxs_data = surface_mesh_vtx_buf;

    for (int y = step, sy = 0; y < height; y += step, ++sy) {
        for (int x = step, sx = 0; x < width; x += step, ++sx) {
            if (!geo_cons_map[sy * step_width + sx])
                continue;

            depth = depth_buf[y * stride + x];
            Xc = Kinv * Eigen::Vector3f(x, y, 1.0f) * depth;

            Xw = Q_w_c * Xc + T_w_c;
            gridId = sy * step_width + sx;
            vertId = vtx_num;
            grid_to_cur_vert_idx[gridId] = vertId;

            pGLObj.vertices_.emplace_back(Xw);
            vtx_num++;
        }
    }

    int facet_num = 0;
    //int *facets_data = surface_mesh_facet_buf;
    for (int y = step, sy = 0; y < height; y += step, ++sy) {
        for (int x = step, sx = 0; x < width; x += step, ++sx) {
            if (((sy - 1) * step_width + sx - 1) < 0 || (sy * step_width + sx - 1) < 0) {
                continue;
            }
            if (geo_cons_map[sy * step_width + sx] &&
                geo_cons_map[(sy - 1) * step_width + sx] &&
                geo_cons_map[(sy - 1) * step_width + sx - 1]) {

                facet[0] = grid_to_cur_vert_idx[sy * step_width + sx];
                facet[1] = grid_to_cur_vert_idx[(sy - 1) * step_width + sx];
                facet[2] = grid_to_cur_vert_idx[(sy - 1) * step_width + sx - 1];

                //if (!(facet[0] == facet[1] || facet[0] == facet[2] || facet[1] == facet[2] || facet[2] <= 0)) {
                if (!(facet[0] < 0 || facet[1] < 0 || facet[2] < 0)) {
                    pGLObj.faces_.emplace_back(facet);
                    facet_num++;
                }
            }
            if (geo_cons_map[sy * step_width + sx] &&
                geo_cons_map[sy * step_width + sx - 1] &&
                geo_cons_map[(sy - 1) * step_width + sx - 1]) {

                facet[0] = grid_to_cur_vert_idx[sy * step_width + sx];
                facet[1] = grid_to_cur_vert_idx[(sy - 1) * step_width + sx - 1];
                facet[2] = grid_to_cur_vert_idx[sy * step_width + sx - 1];

                //if (!(facet[0] == facet[1] || facet[0] == facet[2] || facet[1] == facet[2] || facet[2] <= 0)) {
                if (!(facet[0] < 0 || facet[1] < 0 || facet[2] < 0)) {
                    pGLObj.faces_.emplace_back(facet);
                    facet_num++;
                }
            }
        }
    }

    // std::cout << "facet_num: " << facet_num << ", " << pGLObj.faces_.size() << std::endl;
    return true;
}

}
