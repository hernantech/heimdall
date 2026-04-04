// ============================================================================
// heimdall::texture — BVH construction and ray-mesh visibility testing.
// ============================================================================

#include "visibility.h"

#include <algorithm>
#include <cmath>

namespace heimdall::texture {

// ---------------------------------------------------------------------------
// Vec3 non-inline members
// ---------------------------------------------------------------------------

float Vec3::length() const { return std::sqrt(length_sq()); }

Vec3 Vec3::normalized() const {
    float len = length();
    if (len < 1e-12f) return {0.0f, 0.0f, 0.0f};
    float inv = 1.0f / len;
    return {x * inv, y * inv, z * inv};
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

void AABB::expand(const Vec3& p) {
    min_pt.x = std::min(min_pt.x, p.x);
    min_pt.y = std::min(min_pt.y, p.y);
    min_pt.z = std::min(min_pt.z, p.z);
    max_pt.x = std::max(max_pt.x, p.x);
    max_pt.y = std::max(max_pt.y, p.y);
    max_pt.z = std::max(max_pt.z, p.z);
}

void AABB::expand(const AABB& other) {
    expand(other.min_pt);
    expand(other.max_pt);
}

int AABB::longest_axis() const {
    float dx = max_pt.x - min_pt.x;
    float dy = max_pt.y - min_pt.y;
    float dz = max_pt.z - min_pt.z;
    if (dx >= dy && dx >= dz) return 0;
    if (dy >= dz) return 1;
    return 2;
}

// ---------------------------------------------------------------------------
// VisibilityTester — BVH build
// ---------------------------------------------------------------------------

void VisibilityTester::build(const float* positions, size_t num_vertices,
                             const uint32_t* indices, size_t num_triangles) {
    (void)num_vertices;

    // Populate triangle pool.
    triangles_.resize(num_triangles);
    for (size_t i = 0; i < num_triangles; ++i) {
        uint32_t i0 = indices[i * 3 + 0];
        uint32_t i1 = indices[i * 3 + 1];
        uint32_t i2 = indices[i * 3 + 2];

        triangles_[i].v0 = {positions[i0 * 3 + 0],
                            positions[i0 * 3 + 1],
                            positions[i0 * 3 + 2]};
        triangles_[i].v1 = {positions[i1 * 3 + 0],
                            positions[i1 * 3 + 1],
                            positions[i1 * 3 + 2]};
        triangles_[i].v2 = {positions[i2 * 3 + 0],
                            positions[i2 * 3 + 1],
                            positions[i2 * 3 + 2]};
        triangles_[i].face_index = static_cast<uint32_t>(i);
    }

    // Build BVH.
    nodes_.clear();
    nodes_.reserve(num_triangles * 2);  // upper-bound estimate
    if (num_triangles > 0) {
        build_recursive(0, static_cast<uint32_t>(num_triangles));
    }
}

uint32_t VisibilityTester::build_recursive(uint32_t tri_begin,
                                           uint32_t tri_end) {
    uint32_t node_idx = static_cast<uint32_t>(nodes_.size());
    nodes_.emplace_back();

    uint32_t count = tri_end - tri_begin;

    // Compute bounding box of all triangles in [tri_begin, tri_end).
    AABB bounds;
    for (uint32_t i = tri_begin; i < tri_end; ++i) {
        bounds.expand(triangles_[i].v0);
        bounds.expand(triangles_[i].v1);
        bounds.expand(triangles_[i].v2);
    }

    nodes_[node_idx].bounds = bounds;

    // Leaf criterion.
    if (count <= static_cast<uint32_t>(kMaxLeafSize)) {
        nodes_[node_idx].tri_start = tri_begin;
        nodes_[node_idx].tri_count = count;
        return node_idx;
    }

    // Midpoint split along the longest axis.
    int axis = bounds.longest_axis();

    // Compute centroid-based split value.
    auto centroid_component = [&](const Triangle& t, int ax) -> float {
        float cx = (t.v0.x + t.v1.x + t.v2.x) / 3.0f;
        float cy = (t.v0.y + t.v1.y + t.v2.y) / 3.0f;
        float cz = (t.v0.z + t.v1.z + t.v2.z) / 3.0f;
        if (ax == 0) return cx;
        if (ax == 1) return cy;
        return cz;
    };

    float axis_min = (axis == 0) ? bounds.min_pt.x
                   : (axis == 1) ? bounds.min_pt.y
                                 : bounds.min_pt.z;
    float axis_max = (axis == 0) ? bounds.max_pt.x
                   : (axis == 1) ? bounds.max_pt.y
                                 : bounds.max_pt.z;
    float mid = (axis_min + axis_max) * 0.5f;

    // Partition triangles around the midpoint.
    auto* begin_ptr = triangles_.data() + tri_begin;
    auto* end_ptr   = triangles_.data() + tri_end;
    auto* pivot = std::partition(begin_ptr, end_ptr,
        [&](const Triangle& t) {
            return centroid_component(t, axis) < mid;
        });

    uint32_t split = static_cast<uint32_t>(pivot - triangles_.data());

    // If partition failed (all triangles on one side), force an even split.
    if (split == tri_begin || split == tri_end) {
        split = tri_begin + count / 2;
    }

    // Recurse.  Note: nodes_ may reallocate, so we must re-index after each
    // recursive call — but node_idx is stable because we reserved.
    uint32_t left_idx  = build_recursive(tri_begin, split);
    uint32_t right_idx = build_recursive(split, tri_end);

    nodes_[node_idx].left  = left_idx;
    nodes_[node_idx].right = right_idx;
    // tri_count remains 0 → not a leaf

    return node_idx;
}

// ---------------------------------------------------------------------------
// Ray-triangle intersection — Moeller-Trumbore
// ---------------------------------------------------------------------------

float VisibilityTester::ray_triangle_intersect(const Ray& ray,
                                               const Triangle& tri) {
    constexpr float kEpsilon = 1e-8f;

    Vec3 e1 = tri.v1 - tri.v0;
    Vec3 e2 = tri.v2 - tri.v0;
    Vec3 h  = cross(ray.direction, e2);
    float a = dot(e1, h);

    if (a > -kEpsilon && a < kEpsilon)
        return -1.0f;  // parallel

    float f = 1.0f / a;
    Vec3 s  = ray.origin - tri.v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return -1.0f;

    Vec3 q  = cross(s, e1);
    float v = f * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f)
        return -1.0f;

    float t = f * dot(e2, q);
    return t;
}

// ---------------------------------------------------------------------------
// Ray-AABB slab test
// ---------------------------------------------------------------------------

bool VisibilityTester::ray_aabb_intersect(const Ray& ray,
                                          const Vec3& inv_dir,
                                          const AABB& box,
                                          float t_max) {
    float t1 = (box.min_pt.x - ray.origin.x) * inv_dir.x;
    float t2 = (box.max_pt.x - ray.origin.x) * inv_dir.x;
    float tmin = std::min(t1, t2);
    float tmax = std::max(t1, t2);

    t1 = (box.min_pt.y - ray.origin.y) * inv_dir.y;
    t2 = (box.max_pt.y - ray.origin.y) * inv_dir.y;
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));

    t1 = (box.min_pt.z - ray.origin.z) * inv_dir.z;
    t2 = (box.max_pt.z - ray.origin.z) * inv_dir.z;
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));

    // The ray intersects the AABB if tmax >= max(tmin, 0) and tmin < t_max.
    return tmax >= std::max(tmin, 0.0f) && tmin < t_max;
}

// ---------------------------------------------------------------------------
// Visibility query
// ---------------------------------------------------------------------------

bool VisibilityTester::is_visible(const Vec3& surface_point,
                                  const Vec3& camera_pos,
                                  uint32_t surface_face_idx) const {
    if (nodes_.empty()) return true;

    Vec3 dir = camera_pos - surface_point;
    float dist = dir.length();
    if (dist < 1e-10f) return true;

    // Small epsilon to offset the ray origin off the surface.
    constexpr float kRayEps = 1e-4f;

    Ray ray;
    ray.origin    = surface_point + dir * (kRayEps / dist);
    ray.direction = dir;
    ray.t_max     = 1.0f - 2.0f * kRayEps / dist;  // stop just before camera

    Vec3 inv_dir{1.0f / ray.direction.x,
                 1.0f / ray.direction.y,
                 1.0f / ray.direction.z};

    // If any triangle is hit, the point is occluded.
    return !any_hit(ray, inv_dir, 0, surface_face_idx);
}

bool VisibilityTester::any_hit(const Ray& ray, const Vec3& inv_dir,
                               uint32_t node_idx,
                               uint32_t exclude_face) const {
    const BVHNode& node = nodes_[node_idx];

    if (!ray_aabb_intersect(ray, inv_dir, node.bounds, ray.t_max))
        return false;

    if (node.is_leaf()) {
        for (uint32_t i = node.tri_start;
             i < node.tri_start + node.tri_count; ++i) {
            if (triangles_[i].face_index == exclude_face)
                continue;

            float t = ray_triangle_intersect(ray, triangles_[i]);
            if (t > 0.0f && t < ray.t_max)
                return true;
        }
        return false;
    }

    // Traverse children.  Could sort by distance for early-out, but for an
    // any-hit query the order matters less.
    if (any_hit(ray, inv_dir, node.left, exclude_face))
        return true;
    return any_hit(ray, inv_dir, node.right, exclude_face);
}

} // namespace heimdall::texture
