#include "fusion.h"

#include <cstring>
#include <stdexcept>

namespace heimdall::fusion {

// ── Helper: build a 4x4 row-major matrix from 3x3 rotation + 3x1 translation ──

static void build_4x4(const float rot[9], const float trans[3], float out[16]) {
    // Row 0
    out[0]  = rot[0]; out[1]  = rot[1]; out[2]  = rot[2]; out[3]  = trans[0];
    // Row 1
    out[4]  = rot[3]; out[5]  = rot[4]; out[6]  = rot[5]; out[7]  = trans[1];
    // Row 2
    out[8]  = rot[6]; out[9]  = rot[7]; out[10] = rot[8]; out[11] = trans[2];
    // Row 3
    out[12] = 0.0f;   out[13] = 0.0f;   out[14] = 0.0f;   out[15] = 1.0f;
}

// Invert a rigid-body 4x4 transform (R|t) -> (R^T | -R^T * t).
static void invert_rigid_4x4(const float in[16], float out[16]) {
    // Transpose the 3x3 rotation part
    out[0]  = in[0]; out[1]  = in[4]; out[2]  = in[8];
    out[4]  = in[1]; out[5]  = in[5]; out[6]  = in[9];
    out[8]  = in[2]; out[9]  = in[6]; out[10] = in[10];

    // Translation: -R^T * t
    float tx = in[3], ty = in[7], tz = in[11];
    out[3]  = -(out[0]*tx + out[1]*ty + out[2]*tz);
    out[7]  = -(out[4]*tx + out[5]*ty + out[6]*tz);
    out[11] = -(out[8]*tx + out[9]*ty + out[10]*tz);

    // Bottom row
    out[12] = 0.0f; out[13] = 0.0f; out[14] = 0.0f; out[15] = 1.0f;
}

// ── FusionEngine ────────────────────────────────────────────────────────

FusionEngine::FusionEngine(const FusionConfig& config)
    : config_(config)
{
    configure(config);
}

FusionEngine::~FusionEngine() = default;

void FusionEngine::configure(const FusionConfig& config) {
    config_ = config;

    if (config_.method == FusionMethod::TSDF) {
        VolumeParams vp = make_volume_params(config_);
        tsdf_volume_ = std::make_unique<TsdfVolume>(vp);
        pc_fusion_.reset();
    } else {
        PointCloudFusionConfig pc_config;
        pc_config.depth_min = config_.min_depth;
        pc_config.depth_max = config_.max_depth;
        pc_config.min_consistent_views = config_.min_views;
        pc_config.consistency_threshold = config_.consistency_threshold;
        pc_config.weight_by_angle = config_.weight_by_angle;
        pc_fusion_ = std::make_unique<PointCloudFusion>(pc_config);
        tsdf_volume_.reset();
    }

    integration_count_ = 0;
    accumulated_views_.clear();
}

void FusionEngine::integrate_depth(const DepthInput& input, cudaStream_t stream) {
    FusionCameraParams cam = make_camera_params(input.calibration);

    if (config_.method == FusionMethod::TSDF) {
        tsdf_volume_->integrate(
            input.depth_gpu,
            input.color_gpu,
            input.alpha_gpu,
            cam,
            stream
        );
    } else {
        // For point cloud fusion, accumulate views and fuse on extract.
        DepthViewGpu view;
        view.camera_index = input.camera_index;
        view.width = input.calibration.width;
        view.height = input.calibration.height;
        view.depth_gpu = input.depth_gpu;
        view.color_gpu = input.color_gpu;
        view.alpha_gpu = input.alpha_gpu;
        view.camera = cam;
        accumulated_views_.push_back(view);
    }

    integration_count_++;
}

FusedPointCloud FusionEngine::extract_point_cloud(cudaStream_t stream) {
    FusedPointCloud result;

    if (config_.method == FusionMethod::TSDF) {
        ExtractedPoints pts = tsdf_volume_->extract_points(stream);
        result.num_points = pts.num_points;
        result.positions = std::move(pts.positions);
        result.normals = std::move(pts.normals);
        result.colors = std::move(pts.colors);
    } else {
        if (accumulated_views_.empty()) return result;
        ExtractedPoints pts = pc_fusion_->fuse(accumulated_views_, stream);
        result.num_points = pts.num_points;
        result.positions = std::move(pts.positions);
        result.normals = std::move(pts.normals);
        result.colors = std::move(pts.colors);
    }

    return result;
}

FusedMesh FusionEngine::extract_mesh(cudaStream_t stream) {
    if (config_.method != FusionMethod::TSDF) {
        throw std::runtime_error(
            "FusionEngine::extract_mesh() requires TSDF fusion method. "
            "Point cloud fusion does not produce a mesh directly — "
            "use extract_point_cloud() and pipe to Poisson reconstruction."
        );
    }

    ExtractedMesh raw = tsdf_volume_->extract_mesh(stream);

    FusedMesh result;
    int num_verts = static_cast<int>(raw.vertices.size());
    int num_tris = static_cast<int>(raw.indices.size()) / 3;

    result.num_vertices = num_verts;
    result.num_triangles = num_tris;
    result.indices = std::move(raw.indices);

    // Unpack MeshVertex SOA into flat arrays for downstream consumption
    result.vertices.resize(num_verts * 3);
    result.normals.resize(num_verts * 3);
    result.colors.resize(num_verts * 3);

    for (int i = 0; i < num_verts; i++) {
        const MeshVertex& mv = raw.vertices[i];
        result.vertices[i * 3 + 0] = mv.x;
        result.vertices[i * 3 + 1] = mv.y;
        result.vertices[i * 3 + 2] = mv.z;
        result.normals[i * 3 + 0] = mv.nx;
        result.normals[i * 3 + 1] = mv.ny;
        result.normals[i * 3 + 2] = mv.nz;
        result.colors[i * 3 + 0] = mv.r;
        result.colors[i * 3 + 1] = mv.g;
        result.colors[i * 3 + 2] = mv.b;
    }

    return result;
}

void FusionEngine::reset(cudaStream_t stream) {
    if (config_.method == FusionMethod::TSDF && tsdf_volume_) {
        tsdf_volume_->reset(stream);
    }

    accumulated_views_.clear();
    integration_count_ = 0;
}

FusionCameraParams FusionEngine::make_camera_params(const CameraCalibration& calib) {
    FusionCameraParams cam;
    cam.width = calib.width;
    cam.height = calib.height;
    cam.fx = calib.fx;
    cam.fy = calib.fy;
    cam.cx = calib.cx;
    cam.cy = calib.cy;

    // The calibration stores rotation + translation as world→camera (OpenCV convention).
    // Build the 4x4 world_to_cam matrix.
    build_4x4(calib.rotation, calib.translation, cam.world_to_cam);

    // Invert to get cam_to_world.
    invert_rigid_4x4(cam.world_to_cam, cam.cam_to_world);

    return cam;
}

VolumeParams FusionEngine::make_volume_params(const FusionConfig& config) {
    VolumeParams vp;
    vp.origin_x = config.volume_min_x;
    vp.origin_y = config.volume_min_y;
    vp.origin_z = config.volume_min_z;
    vp.voxel_size = config.voxel_size;

    float extent_x = config.volume_max_x - config.volume_min_x;
    float extent_y = config.volume_max_y - config.volume_min_y;
    float extent_z = config.volume_max_z - config.volume_min_z;

    vp.dim_x = static_cast<int>(extent_x / config.voxel_size + 0.5f);
    vp.dim_y = static_cast<int>(extent_y / config.voxel_size + 0.5f);
    vp.dim_z = static_cast<int>(extent_z / config.voxel_size + 0.5f);

    // Clamp to at least 1
    vp.dim_x = (vp.dim_x > 0) ? vp.dim_x : 1;
    vp.dim_y = (vp.dim_y > 0) ? vp.dim_y : 1;
    vp.dim_z = (vp.dim_z > 0) ? vp.dim_z : 1;

    vp.total_voxels = static_cast<int64_t>(vp.dim_x) * vp.dim_y * vp.dim_z;

    vp.truncation_distance = config.tsdf_truncation;
    vp.depth_min = config.min_depth;
    vp.depth_max = config.max_depth;
    vp.weight_by_angle = config.weight_by_angle;

    return vp;
}

} // namespace heimdall::fusion
