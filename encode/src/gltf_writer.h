#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace heimdall::encode {

struct MeshFrame {
    int64_t frame_id;
    int64_t timestamp_ns;

    std::vector<float> positions;      // xyz interleaved, 3 floats per vertex
    std::vector<float> normals;        // xyz interleaved, 3 floats per vertex
    std::vector<float> texcoords;      // uv interleaved, 2 floats per vertex (optional)
    std::vector<uint32_t> indices;     // triangle indices, 3 per face
    std::vector<uint8_t> vertex_colors; // rgb, 3 bytes per vertex (optional)

    // Texture atlas (optional — omitted in real-time VDTM mode)
    std::vector<uint8_t> texture_ktx2; // KTX2-encoded texture atlas
    int texture_width;
    int texture_height;
};

struct GltfWriterConfig {
    bool use_meshopt_compression = true;
    bool use_draco_fallback = false;
    bool include_texture = true;
    bool binary_output = true;           // .glb vs .gltf
    int meshopt_position_bits = 14;
    int meshopt_texcoord_bits = 12;
    int meshopt_normal_bits = 8;
};

// Write a single mesh frame to a .glb file.
// Uses tinygltf internally.
bool write_gltf_frame(
    const std::string& output_path,
    const MeshFrame& frame,
    const GltfWriterConfig& config
);

// Write a streaming manifest for a sequence of frames.
struct SegmentInfo {
    int64_t start_frame;
    int64_t end_frame;
    double duration_s;
    std::vector<std::string> frame_paths;  // relative paths to .glb files
    size_t total_bytes;
};

bool write_stream_manifest(
    const std::string& output_path,
    const std::vector<SegmentInfo>& segments,
    double fps,
    const std::string& version = "1.0"
);

} // namespace heimdall::encode
