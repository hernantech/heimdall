// ============================================================================
// heimdall::texture — Multi-view texture atlas generation (implementation).
// ============================================================================

#include "texturing.h"

#include <algorithm>
#include <cmath>

namespace heimdall::texture {

// ===========================================================================
// configure
// ===========================================================================

void TextureMapper::configure(const TextureConfig& config) {
    config_ = config;
}

// ===========================================================================
// set_mesh
// ===========================================================================

void TextureMapper::set_mesh(const float* positions, size_t num_vertices,
                             const float* normals,
                             const float* texcoords,
                             const uint32_t* indices, size_t num_triangles) {
    num_vertices_  = num_vertices;
    num_triangles_ = num_triangles;

    positions_.assign(positions, positions + num_vertices * 3);
    texcoords_.assign(texcoords, texcoords + num_vertices * 2);
    indices_.assign(indices, indices + num_triangles * 3);

    // Store or compute normals.
    if (normals != nullptr) {
        normals_.assign(normals, normals + num_vertices * 3);
    } else {
        // Compute area-weighted per-vertex normals.
        normals_.assign(num_vertices * 3, 0.0f);
        for (size_t f = 0; f < num_triangles; ++f) {
            uint32_t i0 = indices_[f * 3 + 0];
            uint32_t i1 = indices_[f * 3 + 1];
            uint32_t i2 = indices_[f * 3 + 2];

            Vec3 p0{positions_[i0 * 3], positions_[i0 * 3 + 1], positions_[i0 * 3 + 2]};
            Vec3 p1{positions_[i1 * 3], positions_[i1 * 3 + 1], positions_[i1 * 3 + 2]};
            Vec3 p2{positions_[i2 * 3], positions_[i2 * 3 + 1], positions_[i2 * 3 + 2]};

            Vec3 fn = cross(p1 - p0, p2 - p0);  // length proportional to area

            for (uint32_t vi : {i0, i1, i2}) {
                normals_[vi * 3 + 0] += fn.x;
                normals_[vi * 3 + 1] += fn.y;
                normals_[vi * 3 + 2] += fn.z;
            }
        }
        // Normalize.
        for (size_t i = 0; i < num_vertices; ++i) {
            Vec3 n{normals_[i * 3], normals_[i * 3 + 1], normals_[i * 3 + 2]};
            n = n.normalized();
            normals_[i * 3 + 0] = n.x;
            normals_[i * 3 + 1] = n.y;
            normals_[i * 3 + 2] = n.z;
        }
    }

    // Build BVH for visibility testing.
    visibility_.build(positions_.data(), num_vertices_,
                      indices_.data(), num_triangles_);

    // Pre-build UV triangles for atlas rasterization.
    build_uv_triangles();
}

// ===========================================================================
// set_camera
// ===========================================================================

void TextureMapper::set_camera(const CameraView& view) {
    cameras_.push_back(view);
}

// ===========================================================================
// build_uv_triangles
// ===========================================================================

void TextureMapper::build_uv_triangles() {
    uv_triangles_.resize(num_triangles_);

    float w = static_cast<float>(config_.atlas_width);
    float h = static_cast<float>(config_.atlas_height);

    for (size_t f = 0; f < num_triangles_; ++f) {
        uint32_t i0 = indices_[f * 3 + 0];
        uint32_t i1 = indices_[f * 3 + 1];
        uint32_t i2 = indices_[f * 3 + 2];

        UVTriangle& ut = uv_triangles_[f];

        // UV in texel space.  Texcoords are [0,1] range from xatlas.
        ut.u0 = texcoords_[i0 * 2 + 0] * w;
        ut.v0 = texcoords_[i0 * 2 + 1] * h;
        ut.u1 = texcoords_[i1 * 2 + 0] * w;
        ut.v1 = texcoords_[i1 * 2 + 1] * h;
        ut.u2 = texcoords_[i2 * 2 + 0] * w;
        ut.v2 = texcoords_[i2 * 2 + 1] * h;

        // 3D positions.
        ut.p0 = {positions_[i0 * 3], positions_[i0 * 3 + 1], positions_[i0 * 3 + 2]};
        ut.p1 = {positions_[i1 * 3], positions_[i1 * 3 + 1], positions_[i1 * 3 + 2]};
        ut.p2 = {positions_[i2 * 3], positions_[i2 * 3 + 1], positions_[i2 * 3 + 2]};

        // 3D normals.
        ut.n0 = {normals_[i0 * 3], normals_[i0 * 3 + 1], normals_[i0 * 3 + 2]};
        ut.n1 = {normals_[i1 * 3], normals_[i1 * 3 + 1], normals_[i1 * 3 + 2]};
        ut.n2 = {normals_[i2 * 3], normals_[i2 * 3 + 1], normals_[i2 * 3 + 2]};

        ut.face_index = static_cast<uint32_t>(f);
    }
}

// ===========================================================================
// face_normal
// ===========================================================================

Vec3 TextureMapper::face_normal(uint32_t face_idx) const {
    uint32_t i0 = indices_[face_idx * 3 + 0];
    uint32_t i1 = indices_[face_idx * 3 + 1];
    uint32_t i2 = indices_[face_idx * 3 + 2];

    Vec3 p0{positions_[i0 * 3], positions_[i0 * 3 + 1], positions_[i0 * 3 + 2]};
    Vec3 p1{positions_[i1 * 3], positions_[i1 * 3 + 1], positions_[i1 * 3 + 2]};
    Vec3 p2{positions_[i2 * 3], positions_[i2 * 3 + 1], positions_[i2 * 3 + 2]};

    return cross(p1 - p0, p2 - p0).normalized();
}

// ===========================================================================
// camera_world_position
// ===========================================================================

Vec3 TextureMapper::camera_world_position(const CameraView& cam) {
    // Extrinsics: X_cam = R * X_world + t
    // Camera position in world = -R^T * t
    const float* R = cam.rotation;
    const float* t = cam.translation;

    // R^T * t (R is row-major, so R^T columns are R's rows).
    float wx = R[0] * t[0] + R[3] * t[1] + R[6] * t[2];
    float wy = R[1] * t[0] + R[4] * t[1] + R[7] * t[2];
    float wz = R[2] * t[0] + R[5] * t[1] + R[8] * t[2];

    return {-wx, -wy, -wz};
}

// ===========================================================================
// project_point
// ===========================================================================

bool TextureMapper::project_point(const CameraView& cam,
                                  const Vec3& world_pt,
                                  float& px, float& py) {
    const float* R = cam.rotation;
    const float* t = cam.translation;

    // Transform to camera coordinates: P_cam = R * P_world + t
    float cx = R[0] * world_pt.x + R[1] * world_pt.y + R[2] * world_pt.z + t[0];
    float cy = R[3] * world_pt.x + R[4] * world_pt.y + R[5] * world_pt.z + t[1];
    float cz = R[6] * world_pt.x + R[7] * world_pt.y + R[8] * world_pt.z + t[2];

    // Point must be in front of the camera.
    if (cz <= 0.0f)
        return false;

    // Project using pinhole model: px = fx * (cx/cz) + cx_offset
    float inv_z = 1.0f / cz;
    px = cam.fx * cx * inv_z + cam.cx;
    py = cam.fy * cy * inv_z + cam.cy;

    // Check if within image bounds (with 0.5px margin).
    if (px < 0.5f || px >= static_cast<float>(cam.image_width) - 0.5f)
        return false;
    if (py < 0.5f || py >= static_cast<float>(cam.image_height) - 0.5f)
        return false;

    return true;
}

// ===========================================================================
// sample_image (bilinear interpolation)
// ===========================================================================

void TextureMapper::sample_image(const CameraView& cam,
                                 float px, float py,
                                 float& r, float& g, float& b) {
    // Pixel center convention: integer coords are at pixel centers.
    float fx = px - 0.5f;
    float fy = py - 0.5f;

    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float sx = fx - static_cast<float>(x0);
    float sy = fy - static_cast<float>(y0);

    // Clamp to image bounds.
    x0 = std::max(0, std::min(x0, cam.image_width - 1));
    x1 = std::max(0, std::min(x1, cam.image_width - 1));
    y0 = std::max(0, std::min(y0, cam.image_height - 1));
    y1 = std::max(0, std::min(y1, cam.image_height - 1));

    int stride = cam.image_width * 3;

    auto fetch = [&](int ix, int iy, float& cr, float& cg, float& cb) {
        int idx = iy * stride + ix * 3;
        cr = cam.image_linear_rgb[idx + 0];
        cg = cam.image_linear_rgb[idx + 1];
        cb = cam.image_linear_rgb[idx + 2];
    };

    float r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;
    fetch(x0, y0, r00, g00, b00);
    fetch(x1, y0, r10, g10, b10);
    fetch(x0, y1, r01, g01, b01);
    fetch(x1, y1, r11, g11, b11);

    float w00 = (1.0f - sx) * (1.0f - sy);
    float w10 = sx * (1.0f - sy);
    float w01 = (1.0f - sx) * sy;
    float w11 = sx * sy;

    r = r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11;
    g = g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11;
    b = b00 * w00 + b10 * w10 + b01 * w01 + b11 * w11;
}

// ===========================================================================
// apply_color_matrix
// ===========================================================================

void TextureMapper::apply_color_matrix(const CameraView& cam,
                                       float& r, float& g, float& b) {
    if (!cam.color_matrix.has_value())
        return;

    const auto& m = cam.color_matrix.value();
    float nr = m[0] * r + m[1] * g + m[2] * b;
    float ng = m[3] * r + m[4] * g + m[5] * b;
    float nb = m[6] * r + m[7] * g + m[8] * b;
    r = nr;
    g = ng;
    b = nb;
}

// ===========================================================================
// linear_to_srgb
//
// Standard sRGB transfer function (IEC 61966-2-1):
//   sRGB = 12.92 * linear                   if linear <= 0.0031308
//   sRGB = 1.055 * linear^(1/2.4) - 0.055   otherwise
// ===========================================================================

uint8_t TextureMapper::linear_to_srgb(float linear) {
    linear = std::max(0.0f, std::min(1.0f, linear));

    float srgb;
    if (linear <= 0.0031308f) {
        srgb = 12.92f * linear;
    } else {
        srgb = 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
    }

    return static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, srgb * 255.0f + 0.5f)));
}

// ===========================================================================
// Barycentric coordinate helpers for UV-space rasterization
// ===========================================================================

namespace {

// Compute barycentric coordinates of point (px, py) with respect to
// triangle (ax,ay), (bx,by), (cx,cy).
// Returns (u, v, w) where point = u*A + v*B + w*C.
// u + v + w = 1 if the point is on the triangle plane.
struct Bary {
    float u, v, w;
};

Bary barycentric(float px, float py,
                 float ax, float ay,
                 float bx, float by,
                 float cx, float cy) {
    float v0x = bx - ax, v0y = by - ay;
    float v1x = cx - ax, v1y = cy - ay;
    float v2x = px - ax, v2y = py - ay;

    float d00 = v0x * v0x + v0y * v0y;
    float d01 = v0x * v1x + v0y * v1y;
    float d11 = v1x * v1x + v1y * v1y;
    float d20 = v2x * v0x + v2y * v0y;
    float d21 = v2x * v1x + v2y * v1y;

    float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-12f)
        return {-1.0f, -1.0f, -1.0f};  // degenerate

    float inv = 1.0f / denom;
    float v = (d11 * d20 - d01 * d21) * inv;
    float w = (d00 * d21 - d01 * d20) * inv;
    float u = 1.0f - v - w;

    return {u, v, w};
}

}  // anonymous namespace

// ===========================================================================
// rasterize_triangle
//
// For one UV triangle, iterate over all texels within its bounding box,
// compute barycentric coordinates, and blend camera colors.
// ===========================================================================

void TextureMapper::rasterize_triangle(const UVTriangle& tri,
                                       std::vector<float>& atlas_rgb,
                                       std::vector<float>& atlas_weight) const {
    int w = config_.atlas_width;
    int h = config_.atlas_height;

    // Bounding box in texel space, clamped to atlas.
    int min_x = std::max(0, static_cast<int>(std::floor(std::min({tri.u0, tri.u1, tri.u2}))));
    int max_x = std::min(w - 1, static_cast<int>(std::ceil(std::max({tri.u0, tri.u1, tri.u2}))));
    int min_y = std::max(0, static_cast<int>(std::floor(std::min({tri.v0, tri.v1, tri.v2}))));
    int max_y = std::min(h - 1, static_cast<int>(std::ceil(std::max({tri.v0, tri.v1, tri.v2}))));

    for (int ty = min_y; ty <= max_y; ++ty) {
        for (int tx = min_x; tx <= max_x; ++tx) {
            // Texel center.
            float px = static_cast<float>(tx) + 0.5f;
            float py = static_cast<float>(ty) + 0.5f;

            // Barycentric coordinates in UV space.
            Bary b = barycentric(px, py,
                                 tri.u0, tri.v0,
                                 tri.u1, tri.v1,
                                 tri.u2, tri.v2);

            // Point is inside the triangle if all bary coords are >= 0.
            // Small negative tolerance to catch edge texels.
            constexpr float kBaryEps = -1e-3f;
            if (b.u < kBaryEps || b.v < kBaryEps || b.w < kBaryEps)
                continue;

            // Interpolate 3D position and normal.
            Vec3 world_pt = tri.p0 * b.u + tri.p1 * b.v + tri.p2 * b.w;
            Vec3 normal   = (tri.n0 * b.u + tri.n1 * b.v + tri.n2 * b.w).normalized();

            // Accumulate blended color from all cameras.
            float total_r = 0.0f, total_g = 0.0f, total_b = 0.0f;
            float total_w = 0.0f;
            float best_weight = 0.0f;
            float best_r = 0.0f, best_g = 0.0f, best_b = 0.0f;

            for (const auto& cam : cameras_) {
                // Project world point to camera pixel.
                float cam_px, cam_py;
                if (!project_point(cam, world_pt, cam_px, cam_py))
                    continue;

                // Compute view direction and weighting.
                Vec3 cam_pos = camera_world_position(cam);
                Vec3 to_cam  = cam_pos - world_pt;
                float dist   = to_cam.length();
                if (dist < 1e-8f) continue;
                Vec3 cam_dir = to_cam * (1.0f / dist);

                float ndotc = dot(normal, cam_dir);
                if (ndotc < config_.min_normal_dot)
                    continue;

                // Visibility test: is this surface point visible from the camera?
                if (!visibility_.is_visible(world_pt, cam_pos, tri.face_index))
                    continue;

                // Sample camera image.
                float sr, sg, sb;
                sample_image(cam, cam_px, cam_py, sr, sg, sb);

                // Apply per-camera color correction.
                apply_color_matrix(cam, sr, sg, sb);

                // Compute camera weight.
                float weight;
                switch (config_.blend_mode) {
                    case BlendMode::WEIGHTED_ANGLE: {
                        // Penalize grazing angles with smooth ramp.
                        float grazing = (ndotc < 0.26f)
                            ? ndotc / 0.26f  // linear ramp 0..1
                            : 1.0f;

                        // Gentle distance falloff: 1 / (1 + d^2).
                        // Keeps nearby cameras preferred without harsh cutoffs.
                        float dist_falloff = 1.0f / (1.0f + dist * dist);

                        weight = ndotc * grazing * dist_falloff;
                        break;
                    }
                    case BlendMode::BEST_VIEW:
                        weight = ndotc;
                        break;
                    case BlendMode::UNIFORM:
                        weight = 1.0f;
                        break;
                }

                if (config_.blend_mode == BlendMode::BEST_VIEW) {
                    if (weight > best_weight) {
                        best_weight = weight;
                        best_r = sr;
                        best_g = sg;
                        best_b = sb;
                    }
                } else {
                    total_r += sr * weight;
                    total_g += sg * weight;
                    total_b += sb * weight;
                    total_w += weight;
                }
            }

            float final_r, final_g, final_b;
            if (config_.blend_mode == BlendMode::BEST_VIEW) {
                if (best_weight <= 0.0f)
                    continue;
                final_r = best_r;
                final_g = best_g;
                final_b = best_b;
            } else {
                if (total_w <= 0.0f)
                    continue;
                final_r = total_r / total_w;
                final_g = total_g / total_w;
                final_b = total_b / total_w;
            }

            // Write to atlas (accumulate — later texels on shared edges may
            // overwrite, which is fine since they'll have the same value
            // for a well-constructed UV layout).
            int idx = (ty * w + tx) * 3;
            atlas_rgb[idx + 0] = final_r;
            atlas_rgb[idx + 1] = final_g;
            atlas_rgb[idx + 2] = final_b;
            atlas_weight[ty * w + tx] = 1.0f;  // mark as filled
        }
    }
}

// ===========================================================================
// dilate_atlas
//
// Fills gutter pixels by iteratively expanding filled regions into empty
// neighbors.  Each iteration pushes chart edges out by 1 pixel.
// ===========================================================================

void TextureMapper::dilate_atlas(std::vector<float>& atlas_rgb,
                                 std::vector<float>& atlas_weight) const {
    int w = config_.atlas_width;
    int h = config_.atlas_height;
    int padding = config_.seam_padding_px;

    // Temporary buffers for ping-pong dilation.
    std::vector<float> tmp_rgb(atlas_rgb.size());
    std::vector<float> tmp_weight(atlas_weight.size());

    // 4-connected neighbor offsets.
    static const int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static const int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};
    static constexpr int kNumNeighbors = 8;

    for (int iter = 0; iter < padding; ++iter) {
        std::copy(atlas_rgb.begin(), atlas_rgb.end(), tmp_rgb.begin());
        std::copy(atlas_weight.begin(), atlas_weight.end(), tmp_weight.begin());

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = y * w + x;
                if (atlas_weight[idx] > 0.0f)
                    continue;  // already filled

                // Average the filled neighbors.
                float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
                int count = 0;

                for (int n = 0; n < kNumNeighbors; ++n) {
                    int nx = x + dx[n];
                    int ny = y + dy[n];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h)
                        continue;
                    int nidx = ny * w + nx;
                    if (atlas_weight[nidx] <= 0.0f)
                        continue;
                    sum_r += atlas_rgb[nidx * 3 + 0];
                    sum_g += atlas_rgb[nidx * 3 + 1];
                    sum_b += atlas_rgb[nidx * 3 + 2];
                    ++count;
                }

                if (count > 0) {
                    float inv = 1.0f / static_cast<float>(count);
                    tmp_rgb[idx * 3 + 0] = sum_r * inv;
                    tmp_rgb[idx * 3 + 1] = sum_g * inv;
                    tmp_rgb[idx * 3 + 2] = sum_b * inv;
                    tmp_weight[idx] = 1.0f;
                }
            }
        }

        std::swap(atlas_rgb, tmp_rgb);
        std::swap(atlas_weight, tmp_weight);
    }
}

// ===========================================================================
// generate_atlas
// ===========================================================================

std::vector<uint8_t> TextureMapper::generate_atlas() {
    int w = config_.atlas_width;
    int h = config_.atlas_height;
    size_t num_texels = static_cast<size_t>(w) * static_cast<size_t>(h);

    // Floating-point accumulation buffers (linear RGB).
    std::vector<float> atlas_rgb(num_texels * 3, 0.0f);
    std::vector<float> atlas_weight(num_texels, 0.0f);

    // Rasterize each UV triangle.
    for (const auto& tri : uv_triangles_) {
        rasterize_triangle(tri, atlas_rgb, atlas_weight);
    }

    // Dilate chart edges to fill gutter pixels.
    if (config_.seam_padding_px > 0) {
        dilate_atlas(atlas_rgb, atlas_weight);
    }

    // Convert linear RGB float to sRGB uint8.
    std::vector<uint8_t> output(num_texels * 3);
    for (size_t i = 0; i < num_texels * 3; ++i) {
        output[i] = linear_to_srgb(atlas_rgb[i]);
    }

    return output;
}

} // namespace heimdall::texture
