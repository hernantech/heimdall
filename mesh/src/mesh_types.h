#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace heimdall::mesh {

// ---------------------------------------------------------------------------
// Basic math types (avoid pulling in a full math library for this module).
// ---------------------------------------------------------------------------

struct Vec2 {
    float x = 0.0f, y = 0.0f;
};

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s)       const { return {x * s,   y * s,   z * s};   }

    float length() const { return std::sqrt(x * x + y * y + z * z); }

    Vec3 normalized() const {
        float len = length();
        if (len < 1e-12f) return {0.0f, 0.0f, 0.0f};
        float inv = 1.0f / len;
        return {x * inv, y * inv, z * inv};
    }
};

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ---------------------------------------------------------------------------
// Vertex color (8-bit per channel, matching MeshFrame::vertex_colors layout).
// ---------------------------------------------------------------------------

struct Color3u8 {
    uint8_t r = 0, g = 0, b = 0;
};

// ---------------------------------------------------------------------------
// TriMesh — the primary mesh representation within the mesh module.
//
// Conventions:
//   - CCW winding order for front-facing triangles.
//   - positions, normals, texcoords, and vertex_colors are per-vertex arrays;
//     their sizes must be consistent (see vertex_count()).
//   - indices contains three uint32_t entries per triangle.
//   - normals, texcoords, and vertex_colors are optional and may be empty.
// ---------------------------------------------------------------------------

struct TriMesh {
    std::vector<Vec3>     positions;
    std::vector<Vec3>     normals;        // may be empty
    std::vector<Vec2>     texcoords;      // may be empty
    std::vector<uint32_t> indices;
    std::vector<Color3u8> vertex_colors;  // may be empty

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    size_t vertex_count() const { return positions.size(); }
    size_t face_count()   const { return indices.size() / 3; }
    bool   has_normals()  const { return normals.size() == positions.size(); }
    bool   has_texcoords()const { return texcoords.size() == positions.size(); }
    bool   has_colors()   const { return vertex_colors.size() == positions.size(); }
    bool   empty()        const { return positions.empty(); }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    bool is_valid() const {
        if (indices.size() % 3 != 0) return false;
        for (uint32_t idx : indices) {
            if (idx >= static_cast<uint32_t>(positions.size())) return false;
        }
        if (!normals.empty()       && normals.size()       != positions.size()) return false;
        if (!texcoords.empty()     && texcoords.size()     != positions.size()) return false;
        if (!vertex_colors.empty() && vertex_colors.size() != positions.size()) return false;
        return true;
    }

    // -----------------------------------------------------------------------
    // Conversion from flat float arrays (fusion / marching-cubes output).
    //
    //   flat_positions : xyz interleaved, length = vertex_count * 3
    //   flat_normals   : xyz interleaved, length = vertex_count * 3 (or empty)
    //   flat_texcoords : uv  interleaved, length = vertex_count * 2 (or empty)
    //   flat_colors    : rgb interleaved, length = vertex_count * 3 (or empty)
    // -----------------------------------------------------------------------

    static TriMesh from_flat(const std::vector<float>&    flat_positions,
                             const std::vector<uint32_t>& tri_indices,
                             const std::vector<float>&    flat_normals   = {},
                             const std::vector<float>&    flat_texcoords = {},
                             const std::vector<uint8_t>&  flat_colors    = {}) {
        assert(flat_positions.size() % 3 == 0);
        assert(tri_indices.size() % 3 == 0);

        TriMesh m;
        size_t nv = flat_positions.size() / 3;

        m.positions.resize(nv);
        std::memcpy(m.positions.data(), flat_positions.data(), nv * sizeof(Vec3));

        m.indices = tri_indices;

        if (flat_normals.size() == nv * 3) {
            m.normals.resize(nv);
            std::memcpy(m.normals.data(), flat_normals.data(), nv * sizeof(Vec3));
        }

        if (flat_texcoords.size() == nv * 2) {
            m.texcoords.resize(nv);
            std::memcpy(m.texcoords.data(), flat_texcoords.data(), nv * sizeof(Vec2));
        }

        if (flat_colors.size() == nv * 3) {
            m.vertex_colors.resize(nv);
            std::memcpy(m.vertex_colors.data(), flat_colors.data(), nv * sizeof(Color3u8));
        }

        return m;
    }

    // -----------------------------------------------------------------------
    // Conversion to flat float arrays (for MeshFrame / glTF writer).
    // -----------------------------------------------------------------------

    std::vector<float> flat_positions() const {
        std::vector<float> out(positions.size() * 3);
        std::memcpy(out.data(), positions.data(), out.size() * sizeof(float));
        return out;
    }

    std::vector<float> flat_normals() const {
        std::vector<float> out(normals.size() * 3);
        std::memcpy(out.data(), normals.data(), out.size() * sizeof(float));
        return out;
    }

    std::vector<float> flat_texcoords() const {
        std::vector<float> out(texcoords.size() * 2);
        std::memcpy(out.data(), texcoords.data(), out.size() * sizeof(float));
        return out;
    }

    std::vector<uint8_t> flat_colors() const {
        std::vector<uint8_t> out(vertex_colors.size() * 3);
        std::memcpy(out.data(), vertex_colors.data(), out.size() * sizeof(uint8_t));
        return out;
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /// Remove degenerate (zero-area) faces in-place.
    /// Returns the number of faces removed.
    size_t remove_degenerate_faces() {
        size_t write = 0;
        size_t removed = 0;
        for (size_t f = 0; f < face_count(); ++f) {
            uint32_t i0 = indices[f * 3 + 0];
            uint32_t i1 = indices[f * 3 + 1];
            uint32_t i2 = indices[f * 3 + 2];

            // Degenerate if any two indices are the same.
            if (i0 == i1 || i1 == i2 || i0 == i2) {
                ++removed;
                continue;
            }

            // Degenerate if the triangle has zero area.
            Vec3 e1 = positions[i1] - positions[i0];
            Vec3 e2 = positions[i2] - positions[i0];
            float area2 = cross(e1, e2).length();
            if (area2 < 1e-12f) {
                ++removed;
                continue;
            }

            indices[write * 3 + 0] = i0;
            indices[write * 3 + 1] = i1;
            indices[write * 3 + 2] = i2;
            ++write;
        }
        indices.resize(write * 3);
        return removed;
    }

    /// Compute area-weighted per-vertex normals from face geometry.
    /// Overwrites the normals array.  Preserves CCW winding convention.
    void recompute_normals() {
        normals.assign(positions.size(), {0.0f, 0.0f, 0.0f});

        for (size_t f = 0; f < face_count(); ++f) {
            uint32_t i0 = indices[f * 3 + 0];
            uint32_t i1 = indices[f * 3 + 1];
            uint32_t i2 = indices[f * 3 + 2];

            Vec3 e1 = positions[i1] - positions[i0];
            Vec3 e2 = positions[i2] - positions[i0];
            Vec3 fn = cross(e1, e2); // length = 2 * triangle area

            // Accumulate area-weighted face normal to each vertex.
            normals[i0] = normals[i0] + fn;
            normals[i1] = normals[i1] + fn;
            normals[i2] = normals[i2] + fn;
        }

        for (auto& n : normals) {
            n = n.normalized();
        }
    }
};

} // namespace heimdall::mesh
