#include <vector>
#include <string>
#include <cmath>

// =============================================================================
namespace Poisson_sampling {
// =============================================================================

struct Vec3 {
    Vec3() { x = y = z = 0; }
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) { }
    float x,y,z;
    void normalize(){
        float dist = sqrt(x * x + y * y + z * z);
        x /= dist;
        y /= dist;
        z /= dist;
    }

    Vec3 cross(const Vec3& v) {
        return Vec3(
            y*v.z - z*v.y,
            z*v.x - x*v.z,
            x*v.y - y*v.x);
    }

    inline Vec3 operator- (const Vec3& v) const {return Vec3(x-v.x, y-v.y, z-v.z); }
    inline Vec3& operator+=(const Vec3& v) { x += v.x ; y += v.y ; z += v.z ; return *this ; }
};



/// @param radius : minimal radius between every samples. If radius <= 0
/// a new radius is approximated given the targeted number of samples
/// 'nb_samples'
/// @param nb_samples : ignored if radius > 0 otherwise will try to match
/// and find  nb_samples by finding an appropriate radius.
/// @param verts : list of vertices
/// @param nors : list of normlas coresponding to each verts[].
/// @param tris : triangle indices in verts[] array.
/// @code
///     tri(v0;v1;v3) = { verts[ tris[ith_tri*3 + 0] ],
///                       verts[ tris[ith_tri*3 + 1] ],
///                       verts[ tris[ith_tri*3 + 2] ]   }
/// @endcode
/// @param [out] samples_pos : resulting samples positions
/// @param [out] samples_nors : resulting samples normals associated to samples_pos[]
/// @warning undefined behavior if (radius <= 0 && nb_samples == 0) == true
void poisson_disk(float radius,
                  int nb_samples,
                  const std::vector<Vec3>& verts,
                  const std::vector<Vec3>& nors,
                  const std::vector<int>& tris,
                  std::vector<Vec3>& samples_pos,
                  std::vector<Vec3>& samples_nor);

void poisson_disk(float radius,
                  int nb_samples,
                  const std::vector<Vec3>& verts,
                  const std::vector<Vec3>& nors,
                  const std::vector<int>& tris,
                  const std::vector<int>& labels,
                  std::vector<Vec3>& samples_pos,
                  std::vector<Vec3>& samples_nor,
                  std::vector<int>& samples_lab);


bool load_mesh(const std::string &filename, 
                std::vector<Vec3>& vertexs,
                std::vector<Vec3>& normals,
                std::vector<int>& labels,
                std::vector<int>& faces,
                bool &has_sem,
                bool &has_rgb,
                bool &has_tex);

double pcd_average_spacing(const std::vector<Vec3> &verts);

void get_normals(
    std::vector<Vec3> &v_norms,
    std::vector<Vec3> &f_norms,
    const std::vector<Vec3> &v,
    const std::vector<int> &f);
}// END UTILS_SAMPLING NAMESPACE ===============================================
