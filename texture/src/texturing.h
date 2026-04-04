#pragma once

// ============================================================================
// heimdall::texture — Multi-view texture atlas generation.
//
// Projects camera images onto a UV-unwrapped mesh to produce a texture atlas
// suitable for glTF / KTX2 encoding.
//
// Pipeline:
//   1. configure()      — set atlas resolution and blending parameters
//   2. set_mesh()       — provide the UV-unwrapped TriMesh
//   3. set_camera()     — for each camera: image + CameraCalibration
//   4. generate_atlas() — rasterize the atlas and return RGB bytes
//
// CPU-only, offline quality path.  No CUDA, no external deps beyond stdlib.
// ============================================================================

#include "visibility.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace heimdall::texture {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

enum class BlendMode {
    WEIGHTED_ANGLE,   // Weight by cos(normal, cam_dir) * distance falloff
    BEST_VIEW,        // Pick the single camera with the highest weight
    UNIFORM           // Equal weight to all visible cameras
};

struct TextureConfig {
    int       atlas_width      = 2048;
    int       atlas_height     = 2048;
    BlendMode blend_mode       = BlendMode::WEIGHTED_ANGLE;
    int       seam_padding_px  = 4;      // gutter pixels around UV chart edges
    float     gamma            = 2.2f;   // sRGB output gamma (2.2 = standard)

    // Minimum dot(normal, cam_dir) to consider a camera for a texel.
    // Cameras at grazing angles (< ~15 degrees) are typically excluded.
    float     min_normal_dot   = 0.1f;
};

// ---------------------------------------------------------------------------
// Camera image + calibration (one per camera view)
// ---------------------------------------------------------------------------

struct CameraView {
    int camera_index = -1;

    // Intrinsics
    int   image_width  = 0;
    int   image_height = 0;
    float fx = 0.0f, fy = 0.0f, cx = 0.0f, cy = 0.0f;

    // Extrinsics: world-to-camera transform (OpenCV convention).
    // rotation: 3x3 row-major, translation: 3-element vector.
    float rotation[9]    = {};
    float translation[3] = {};

    // Image pixels: linear RGB, row-major, top-left origin.
    // Length must be image_width * image_height * 3.
    std::vector<float> image_linear_rgb;

    // Optional per-camera 3x3 color correction matrix (row-major).
    // Applied to the linear RGB sample before blending.
    std::optional<std::array<float, 9>> color_matrix;
};

// ---------------------------------------------------------------------------
// Internal: UV-space triangle for atlas rasterization.
// ---------------------------------------------------------------------------

struct UVTriangle {
    // UV coordinates (in texel space: [0, atlas_width] x [0, atlas_height])
    float u0, v0, u1, v1, u2, v2;
    // 3D positions of the three vertices
    Vec3 p0, p1, p2;
    // 3D normals of the three vertices (for interpolation)
    Vec3 n0, n1, n2;
    // Original mesh face index
    uint32_t face_index;
};

// ---------------------------------------------------------------------------
// TextureMapper
// ---------------------------------------------------------------------------

class TextureMapper {
public:
    TextureMapper() = default;

    // Step 1: Set atlas resolution and blending parameters.
    void configure(const TextureConfig& config);

    // Step 2: Provide the UV-unwrapped mesh.
    //
    // positions:  xyz interleaved, 3 floats per vertex
    // normals:    xyz interleaved, 3 floats per vertex (same count as positions)
    // texcoords:  uv interleaved,  2 floats per vertex
    // indices:    3 uint32_t per triangle
    //
    // If normals are empty, flat face normals will be computed automatically.
    void set_mesh(const float* positions, size_t num_vertices,
                  const float* normals,
                  const float* texcoords,
                  const uint32_t* indices, size_t num_triangles);

    // Step 3: Add a camera view (image + calibration).
    // Call once per camera.  Images must be in linear RGB float.
    void set_camera(const CameraView& view);

    // Step 4: Generate the texture atlas.
    //
    // Returns an atlas_width * atlas_height * 3 byte buffer (sRGB, uint8).
    // Row-major, top-left origin, RGB channel order.
    std::vector<uint8_t> generate_atlas();

    // Accessors.
    const TextureConfig& config() const { return config_; }
    int atlas_width()  const { return config_.atlas_width; }
    int atlas_height() const { return config_.atlas_height; }

private:
    TextureConfig config_;

    // Mesh data (stored for rasterization + projection).
    std::vector<float>    positions_;    // interleaved xyz
    std::vector<float>    normals_;      // interleaved xyz
    std::vector<float>    texcoords_;    // interleaved uv
    std::vector<uint32_t> indices_;
    size_t                num_vertices_  = 0;
    size_t                num_triangles_ = 0;

    // Camera views.
    std::vector<CameraView> cameras_;

    // Pre-built UV triangles for atlas rasterization.
    std::vector<UVTriangle> uv_triangles_;

    // BVH for occlusion testing.
    VisibilityTester visibility_;

    // -- Internal helpers --

    // Build UV-space triangles from mesh data.
    void build_uv_triangles();

    // Compute face normal for triangle `face_idx`.
    Vec3 face_normal(uint32_t face_idx) const;

    // Compute camera position in world space from extrinsics.
    static Vec3 camera_world_position(const CameraView& cam);

    // Project a 3D world point to 2D pixel coordinates in a camera.
    // Returns false if the point is behind the camera or outside the image.
    static bool project_point(const CameraView& cam,
                              const Vec3& world_pt,
                              float& px, float& py);

    // Sample a camera image (bilinear interpolation) at (px, py).
    // Returns linear RGB.
    static void sample_image(const CameraView& cam,
                             float px, float py,
                             float& r, float& g, float& b);

    // Apply optional per-camera color correction matrix.
    static void apply_color_matrix(const CameraView& cam,
                                   float& r, float& g, float& b);

    // Linear RGB [0,1] float to sRGB uint8.
    static uint8_t linear_to_srgb(float linear);

    // Rasterize a single UV triangle into the atlas, blending camera colors.
    void rasterize_triangle(const UVTriangle& tri,
                            std::vector<float>& atlas_rgb,
                            std::vector<float>& atlas_weight) const;

    // Dilate/pad the atlas to fill gutter pixels around UV chart edges.
    void dilate_atlas(std::vector<float>& atlas_rgb,
                      std::vector<float>& atlas_weight) const;
};

} // namespace heimdall::texture
