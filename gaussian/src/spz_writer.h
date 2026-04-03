#pragma once

#include "pipeline.h"
#include <cstdint>
#include <string>
#include <vector>

namespace heimdall::gaussian {

// SPZ format writer for streaming Gaussian splats.
// SPZ (Niantic) uses column-based layout + gzip for ~10x compression vs PLY.
//
// For streaming, we extend with delta compression:
//   - Keyframes: full SPZ (all Gaussian attributes)
//   - Delta frames: only changed attributes for tracked Gaussians
//     (position delta, rotation delta, opacity — SH stays from keyframe)
//
// This gives ~20-50x compression vs raw PLY for inter-frame deltas.

struct SpzChunk {
    int64_t start_frame;
    int64_t end_frame;
    bool is_keyframe;
    std::vector<uint8_t> compressed_data;   // gzip-compressed column-based SPZ
    size_t uncompressed_size;
};

struct SpzWriterConfig {
    int keyframe_interval = 30;             // full SPZ every N frames
    bool enable_delta = true;               // delta compression between keyframes
    int gzip_level = 6;                     // 1=fast, 9=small
    int sh_degree = 3;                      // spherical harmonics degree (0-3)
    bool quantize_positions = true;         // 16-bit quantized positions
    bool quantize_rotations = true;         // 8-bit quantized quaternions
};

// Encode a single Gaussian frame to SPZ format.
SpzChunk encode_frame(
    const GaussianFrame& frame,
    const GaussianFrame* prev_keyframe,     // null for keyframes
    const SpzWriterConfig& config
);

// Encode a sequence of frames into streaming chunks.
std::vector<SpzChunk> encode_sequence(
    const std::vector<GaussianFrame>& frames,
    const SpzWriterConfig& config
);

// Write a single SPZ chunk to disk.
bool write_spz_file(const std::string& path, const SpzChunk& chunk);

// Write SPZ data into a glTF buffer for KHR_gaussian_splatting_compression_spz.
// Returns the buffer bytes suitable for embedding in a .glb file.
std::vector<uint8_t> spz_to_gltf_buffer(const SpzChunk& chunk);

} // namespace heimdall::gaussian
