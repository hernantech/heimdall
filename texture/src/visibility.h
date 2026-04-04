#pragma once

// ============================================================================
// heimdall::texture — Visibility / occlusion testing via BVH ray casting.
//
// Builds a simple AABB-based bounding volume hierarchy over a triangle mesh
// and provides ray-intersection queries to determine whether a surface point
// is visible from a given camera position.
//
// CPU-only, offline quality path.  The BVH uses midpoint splits (no SAH) and
// is perfectly adequate for meshes under ~50K faces.
// ============================================================================

#include <cstdint>
#include <vector>

namespace heimdall::texture {

// ---------------------------------------------------------------------------
// Minimal math types — mirrors mesh::Vec3 but avoids cross-module dependency.
// ---------------------------------------------------------------------------

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s)       const { return {x * s,   y * s,   z * s};   }

    float length_sq() const { return x * x + y * y + z * z; }
    float length()    const;
    Vec3  normalized() const;
};

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

struct AABB {
    Vec3 min_pt{+1e30f, +1e30f, +1e30f};
    Vec3 max_pt{-1e30f, -1e30f, -1e30f};

    void expand(const Vec3& p);
    void expand(const AABB& other);
    int  longest_axis() const;
};

// ---------------------------------------------------------------------------
// Triangle (stored inside the BVH leaf pool).
// ---------------------------------------------------------------------------

struct Triangle {
    Vec3     v0, v1, v2;
    uint32_t face_index;   // original mesh face index (for caller reference)
};

// ---------------------------------------------------------------------------
// BVH node — flat array layout for cache friendliness.
//
// Internal nodes: left/right are indices into the nodes array.
// Leaf nodes:     left == right == 0, tri_start/tri_count index the triangle
//                 pool.
// ---------------------------------------------------------------------------

struct BVHNode {
    AABB     bounds;
    uint32_t left       = 0;
    uint32_t right      = 0;
    uint32_t tri_start  = 0;
    uint32_t tri_count  = 0;

    bool is_leaf() const { return tri_count > 0; }
};

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

struct Ray {
    Vec3  origin;
    Vec3  direction;      // need not be normalized, but must be nonzero
    float t_max = 1e30f;  // max parameter along ray
};

// ---------------------------------------------------------------------------
// VisibilityTester
//
// Build once per mesh, query many times (once per texel * camera pair).
// ---------------------------------------------------------------------------

class VisibilityTester {
public:
    // Build the BVH from triangle vertex positions and an index buffer.
    // positions: interleaved xyz (3 floats per vertex).
    // indices:   3 uint32_t per triangle.
    void build(const float* positions, size_t num_vertices,
               const uint32_t* indices, size_t num_triangles);

    // Returns true if the ray from `surface_point` to `camera_pos` is
    // unoccluded — i.e., no triangle intersects the segment strictly between
    // the two endpoints.
    //
    // `surface_face_idx` is the face the surface point lies on; that triangle
    // is excluded from the intersection test to avoid self-occlusion.
    bool is_visible(const Vec3& surface_point,
                    const Vec3& camera_pos,
                    uint32_t    surface_face_idx) const;

    bool is_built() const { return !nodes_.empty(); }

    size_t num_triangles() const { return triangles_.size(); }

private:
    static constexpr int kMaxLeafSize = 4;

    std::vector<BVHNode>  nodes_;
    std::vector<Triangle> triangles_;   // reordered triangle pool

    // Recursive BVH builder.  Returns node index.
    uint32_t build_recursive(uint32_t tri_begin, uint32_t tri_end);

    // Moeller-Trumbore ray-triangle intersection.
    // Returns parametric t, or -1 on miss.
    static float ray_triangle_intersect(const Ray& ray, const Triangle& tri);

    // Ray-AABB slab test.
    static bool ray_aabb_intersect(const Ray& ray,
                                   const Vec3& inv_dir,
                                   const AABB& box,
                                   float t_max);

    // Recursive traversal.  Returns true if any hit with t in (eps, t_max).
    bool any_hit(const Ray& ray, const Vec3& inv_dir,
                 uint32_t node_idx,
                 uint32_t exclude_face) const;
};

} // namespace heimdall::texture
