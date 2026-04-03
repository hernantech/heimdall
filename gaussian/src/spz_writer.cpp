#include "spz_writer.h"
#include <zlib.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace heimdall::gaussian {

// ---------------------------------------------------------------------------
// SPZ binary format constants
// ---------------------------------------------------------------------------

// 16-byte header:
//   magic   4 bytes  "SPZ\0"
//   version 4 bytes  uint32 LE
//   count   4 bytes  uint32 LE (number of Gaussians)
//   sh_deg  2 bytes  uint16 LE (SH degree 0-3)
//   flags   2 bytes  uint16 LE (bit 0 = is_delta)
static constexpr uint32_t kSpzMagic   = 0x005a5053; // "SPZ\0" in little-endian
static constexpr uint32_t kSpzVersion = 1;

static constexpr uint16_t kFlagDelta  = 0x0001;

// Number of SH coefficients per channel for a given degree.
// degree 0 -> 1, 1 -> 4, 2 -> 9, 3 -> 16
static int sh_coeffs_per_channel(int degree) {
    switch (degree) {
        case 0: return 1;
        case 1: return 4;
        case 2: return 9;
        case 3: return 16;
        default: return 16;
    }
}

// ---------------------------------------------------------------------------
// Helpers: little-endian serialization
// ---------------------------------------------------------------------------

static void write_u16_le(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

static void write_u32_le(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

static void write_f32_le(std::vector<uint8_t>& buf, float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    write_u32_le(buf, bits);
}

// ---------------------------------------------------------------------------
// Quantization helpers
// ---------------------------------------------------------------------------

// Quantize a float to 16-bit unsigned, mapping [min_val, max_val] -> [0, 65535].
static uint16_t quantize_16(float v, float min_val, float max_val) {
    if (max_val <= min_val) return 0;
    float t = (v - min_val) / (max_val - min_val);
    t = std::clamp(t, 0.0f, 1.0f);
    return static_cast<uint16_t>(std::round(t * 65535.0f));
}

// Quantize a quaternion component from [-1, 1] to uint8 [0, 255].
static uint8_t quantize_quat_8(float v) {
    float t = (v + 1.0f) * 0.5f;
    t = std::clamp(t, 0.0f, 1.0f);
    return static_cast<uint8_t>(std::round(t * 255.0f));
}

// Quantize opacity from [0, 1] to uint8 [0, 255].
static uint8_t quantize_opacity_8(float v) {
    float t = std::clamp(v, 0.0f, 1.0f);
    return static_cast<uint8_t>(std::round(t * 255.0f));
}

// Quantize SH coefficient to uint8. SH values are typically in a small range;
// we use a fixed mapping from [-SH_RANGE, +SH_RANGE] -> [0, 255].
static constexpr float kShRange = 5.0f;

static uint8_t quantize_sh_8(float v) {
    float t = (v + kShRange) / (2.0f * kShRange);
    t = std::clamp(t, 0.0f, 1.0f);
    return static_cast<uint8_t>(std::round(t * 255.0f));
}

// ---------------------------------------------------------------------------
// Bounding box for position quantization
// ---------------------------------------------------------------------------

struct BBox {
    float min[3];
    float max[3];
};

static BBox compute_bbox(const std::vector<Gaussian>& gaussians, int count) {
    BBox bb;
    bb.min[0] = bb.min[1] = bb.min[2] = std::numeric_limits<float>::max();
    bb.max[0] = bb.max[1] = bb.max[2] = std::numeric_limits<float>::lowest();
    for (int i = 0; i < count; i++) {
        for (int a = 0; a < 3; a++) {
            bb.min[a] = std::min(bb.min[a], gaussians[i].position[a]);
            bb.max[a] = std::max(bb.max[a], gaussians[i].position[a]);
        }
    }
    return bb;
}

// ---------------------------------------------------------------------------
// Column-based packing: build the uncompressed SPZ payload
// ---------------------------------------------------------------------------

// Builds the raw (uncompressed) byte buffer for a full keyframe.
// Layout:
//   header (16 bytes)
//   if quantize_positions: bbox (6 floats = 24 bytes), then 3*N uint16
//   else: 3*N float32
//   3*N float32 for scales
//   if quantize_rotations: 4*N uint8
//   else: 4*N float32
//   N uint8 for opacity (always quantized to 8-bit)
//   sh_coeffs_per_channel * 3 * N values (uint8 each)
static std::vector<uint8_t> pack_keyframe(
    const GaussianFrame& frame,
    const SpzWriterConfig& config
) {
    const int n = frame.num_gaussians;
    const int sh_per_ch = sh_coeffs_per_channel(config.sh_degree);
    const int total_sh = sh_per_ch * 3;

    // Estimate buffer size to reduce reallocations.
    size_t estimate = 16; // header
    if (config.quantize_positions) {
        estimate += 24 + static_cast<size_t>(n) * 3 * 2; // bbox + uint16 positions
    } else {
        estimate += static_cast<size_t>(n) * 3 * 4;
    }
    estimate += static_cast<size_t>(n) * 3 * 4; // scales (float32)
    if (config.quantize_rotations) {
        estimate += static_cast<size_t>(n) * 4;
    } else {
        estimate += static_cast<size_t>(n) * 4 * 4;
    }
    estimate += static_cast<size_t>(n);         // opacity
    estimate += static_cast<size_t>(n) * total_sh; // SH

    std::vector<uint8_t> buf;
    buf.reserve(estimate);

    // --- Header ---
    write_u32_le(buf, kSpzMagic);
    write_u32_le(buf, kSpzVersion);
    write_u32_le(buf, static_cast<uint32_t>(n));
    write_u16_le(buf, static_cast<uint16_t>(config.sh_degree));
    write_u16_le(buf, 0); // flags: not a delta

    // --- Positions (column-based: all x, then all y, then all z) ---
    if (config.quantize_positions) {
        BBox bb = compute_bbox(frame.gaussians, n);
        // Write bounding box so decoder can dequantize
        for (int a = 0; a < 3; a++) write_f32_le(buf, bb.min[a]);
        for (int a = 0; a < 3; a++) write_f32_le(buf, bb.max[a]);

        for (int a = 0; a < 3; a++) {
            for (int i = 0; i < n; i++) {
                uint16_t q = quantize_16(frame.gaussians[i].position[a], bb.min[a], bb.max[a]);
                write_u16_le(buf, q);
            }
        }
    } else {
        for (int a = 0; a < 3; a++) {
            for (int i = 0; i < n; i++) {
                write_f32_le(buf, frame.gaussians[i].position[a]);
            }
        }
    }

    // --- Scales (column-based: all sx, then all sy, then all sz) ---
    // Scales are stored as float32 (log-space values don't quantize well)
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < n; i++) {
            write_f32_le(buf, frame.gaussians[i].scale[a]);
        }
    }

    // --- Rotations (column-based: all w, then all x, then all y, then all z) ---
    if (config.quantize_rotations) {
        for (int q = 0; q < 4; q++) {
            for (int i = 0; i < n; i++) {
                buf.push_back(quantize_quat_8(frame.gaussians[i].rotation[q]));
            }
        }
    } else {
        for (int q = 0; q < 4; q++) {
            for (int i = 0; i < n; i++) {
                write_f32_le(buf, frame.gaussians[i].rotation[q]);
            }
        }
    }

    // --- Opacity (column-based: N values, always uint8) ---
    for (int i = 0; i < n; i++) {
        buf.push_back(quantize_opacity_8(frame.gaussians[i].opacity));
    }

    // --- SH coefficients (column-based: for each coeff index, all N values) ---
    // The Gaussian struct stores 48 floats (degree 3 = 16 coeffs * 3 channels).
    // We write only the coefficients needed for the configured SH degree.
    // Channel layout: coeff 0 ch R, coeff 0 ch G, coeff 0 ch B, coeff 1 ch R, ...
    // In the Gaussian struct, sh[0..15] = R, sh[16..31] = G, sh[32..47] = B
    // Column layout: for each (coeff_idx, channel), write all N Gaussians
    for (int ch = 0; ch < 3; ch++) {
        for (int c = 0; c < sh_per_ch; c++) {
            int sh_idx = ch * 16 + c; // index into Gaussian::sh[48]
            for (int i = 0; i < n; i++) {
                buf.push_back(quantize_sh_8(frame.gaussians[i].sh[sh_idx]));
            }
        }
    }

    return buf;
}

// Builds the raw byte buffer for a delta frame.
// Delta frames encode only position deltas, rotation deltas, and opacity.
// SH coefficients and scales are inherited from the keyframe.
// Layout:
//   header (16 bytes, flags has kFlagDelta set)
//   if quantize_positions: delta_bbox (24 bytes), 3*N uint16 position deltas
//   else: 3*N float32 position deltas
//   if quantize_rotations: 4*N uint8 rotation deltas
//   else: 4*N float32 rotation deltas
//   N uint8 opacity
static std::vector<uint8_t> pack_delta(
    const GaussianFrame& frame,
    const GaussianFrame& keyframe,
    const SpzWriterConfig& config
) {
    // Delta frames require matching Gaussian counts (tracked Gaussians).
    // If counts differ, the caller should have sent a keyframe instead.
    const int n = frame.num_gaussians;
    assert(n == keyframe.num_gaussians);

    // Compute position deltas
    std::vector<float> pos_delta(static_cast<size_t>(n) * 3);
    for (int i = 0; i < n; i++) {
        for (int a = 0; a < 3; a++) {
            pos_delta[i * 3 + a] = frame.gaussians[i].position[a] - keyframe.gaussians[i].position[a];
        }
    }

    // Compute rotation deltas (simple difference; decoder adds back)
    std::vector<float> rot_delta(static_cast<size_t>(n) * 4);
    for (int i = 0; i < n; i++) {
        for (int q = 0; q < 4; q++) {
            rot_delta[i * 4 + q] = frame.gaussians[i].rotation[q] - keyframe.gaussians[i].rotation[q];
        }
    }

    // Estimate buffer size
    size_t estimate = 16;
    if (config.quantize_positions) {
        estimate += 24 + static_cast<size_t>(n) * 3 * 2;
    } else {
        estimate += static_cast<size_t>(n) * 3 * 4;
    }
    if (config.quantize_rotations) {
        estimate += static_cast<size_t>(n) * 4;
    } else {
        estimate += static_cast<size_t>(n) * 4 * 4;
    }
    estimate += static_cast<size_t>(n); // opacity

    std::vector<uint8_t> buf;
    buf.reserve(estimate);

    // --- Header ---
    write_u32_le(buf, kSpzMagic);
    write_u32_le(buf, kSpzVersion);
    write_u32_le(buf, static_cast<uint32_t>(n));
    write_u16_le(buf, static_cast<uint16_t>(config.sh_degree));
    write_u16_le(buf, kFlagDelta);

    // --- Position deltas (column-based) ---
    if (config.quantize_positions) {
        // Compute bounding box of deltas
        float dmin[3], dmax[3];
        for (int a = 0; a < 3; a++) {
            dmin[a] = std::numeric_limits<float>::max();
            dmax[a] = std::numeric_limits<float>::lowest();
        }
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < 3; a++) {
                float d = pos_delta[i * 3 + a];
                dmin[a] = std::min(dmin[a], d);
                dmax[a] = std::max(dmax[a], d);
            }
        }
        for (int a = 0; a < 3; a++) write_f32_le(buf, dmin[a]);
        for (int a = 0; a < 3; a++) write_f32_le(buf, dmax[a]);

        for (int a = 0; a < 3; a++) {
            for (int i = 0; i < n; i++) {
                uint16_t q = quantize_16(pos_delta[i * 3 + a], dmin[a], dmax[a]);
                write_u16_le(buf, q);
            }
        }
    } else {
        for (int a = 0; a < 3; a++) {
            for (int i = 0; i < n; i++) {
                write_f32_le(buf, pos_delta[i * 3 + a]);
            }
        }
    }

    // --- Rotation deltas (column-based) ---
    if (config.quantize_rotations) {
        // Rotation deltas are small, map [-1,1] -> [0,255]
        for (int q = 0; q < 4; q++) {
            for (int i = 0; i < n; i++) {
                buf.push_back(quantize_quat_8(rot_delta[i * 4 + q]));
            }
        }
    } else {
        for (int q = 0; q < 4; q++) {
            for (int i = 0; i < n; i++) {
                write_f32_le(buf, rot_delta[i * 4 + q]);
            }
        }
    }

    // --- Opacity (absolute, not delta -- viewer needs the final value) ---
    for (int i = 0; i < n; i++) {
        buf.push_back(quantize_opacity_8(frame.gaussians[i].opacity));
    }

    return buf;
}

// ---------------------------------------------------------------------------
// gzip compression via zlib
// ---------------------------------------------------------------------------

static std::vector<uint8_t> gzip_compress(const std::vector<uint8_t>& input, int level) {
    if (input.empty()) return {};

    // deflateInit2 with windowBits = 15 + 16 produces gzip format
    z_stream zs{};
    int ret = deflateInit2(&zs, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
    if (ret != Z_OK) {
        throw std::runtime_error("spz_writer: deflateInit2 failed (" + std::to_string(ret) + ")");
    }

    // Upper bound on compressed size
    size_t bound = deflateBound(&zs, static_cast<uLong>(input.size()));
    std::vector<uint8_t> output(bound);

    zs.next_in  = const_cast<Bytef*>(input.data());
    zs.avail_in = static_cast<uInt>(input.size());
    zs.next_out = output.data();
    zs.avail_out = static_cast<uInt>(output.size());

    ret = deflate(&zs, Z_FINISH);
    if (ret != Z_STREAM_END) {
        deflateEnd(&zs);
        throw std::runtime_error("spz_writer: deflate failed (" + std::to_string(ret) + ")");
    }

    output.resize(zs.total_out);
    deflateEnd(&zs);
    return output;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

SpzChunk encode_frame(
    const GaussianFrame& frame,
    const GaussianFrame* prev_keyframe,
    const SpzWriterConfig& config
) {
    SpzChunk chunk;
    chunk.start_frame = frame.frame_id;
    chunk.end_frame   = frame.frame_id;

    // Determine if this should be a keyframe:
    //   - explicitly marked as keyframe
    //   - no previous keyframe to delta against
    //   - Gaussian count changed (tracker added/removed splats)
    bool force_keyframe = frame.is_keyframe
        || prev_keyframe == nullptr
        || !config.enable_delta
        || frame.num_gaussians != prev_keyframe->num_gaussians;

    chunk.is_keyframe = force_keyframe;

    // Handle empty frame
    if (frame.num_gaussians == 0 || frame.gaussians.empty()) {
        // Write a minimal header-only SPZ (zero Gaussians)
        std::vector<uint8_t> raw;
        write_u32_le(raw, kSpzMagic);
        write_u32_le(raw, kSpzVersion);
        write_u32_le(raw, 0); // count
        write_u16_le(raw, static_cast<uint16_t>(config.sh_degree));
        write_u16_le(raw, force_keyframe ? 0 : kFlagDelta);

        chunk.uncompressed_size = raw.size();
        chunk.compressed_data = gzip_compress(raw, config.gzip_level);
        return chunk;
    }

    std::vector<uint8_t> raw;
    if (force_keyframe) {
        raw = pack_keyframe(frame, config);
    } else {
        raw = pack_delta(frame, *prev_keyframe, config);
    }

    chunk.uncompressed_size = raw.size();
    chunk.compressed_data = gzip_compress(raw, config.gzip_level);
    return chunk;
}

std::vector<SpzChunk> encode_sequence(
    const std::vector<GaussianFrame>& frames,
    const SpzWriterConfig& config
) {
    if (frames.empty()) return {};

    std::vector<SpzChunk> chunks;
    chunks.reserve(frames.size());

    const GaussianFrame* current_keyframe = nullptr;
    int frames_since_keyframe = 0;

    for (size_t i = 0; i < frames.size(); i++) {
        const auto& frame = frames[i];

        // Force keyframe at configured interval or on first frame
        bool need_keyframe = (i == 0)
            || frames_since_keyframe >= config.keyframe_interval
            || frame.is_keyframe;

        const GaussianFrame* ref = need_keyframe ? nullptr : current_keyframe;
        SpzChunk chunk = encode_frame(frame, ref, config);

        // If encode_frame decided to make it a keyframe (e.g., count mismatch),
        // update our tracking accordingly.
        if (chunk.is_keyframe) {
            current_keyframe = &frame;
            frames_since_keyframe = 1;
        } else {
            frames_since_keyframe++;
        }

        chunks.push_back(std::move(chunk));
    }

    return chunks;
}

bool write_spz_file(const std::string& path, const SpzChunk& chunk) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;

    ofs.write(reinterpret_cast<const char*>(chunk.compressed_data.data()),
              static_cast<std::streamsize>(chunk.compressed_data.size()));

    return ofs.good();
}

std::vector<uint8_t> spz_to_gltf_buffer(const SpzChunk& chunk) {
    // Wrap SPZ data for embedding in a glTF binary (.glb) buffer.
    //
    // KHR_gaussian_splatting_compression_spz extension format:
    //   - 4 bytes: magic "GSPZ" (identifies this buffer as compressed splats)
    //   - 4 bytes: uint32 LE version (1)
    //   - 4 bytes: uint32 LE flags (bit 0 = is_delta)
    //   - 4 bytes: uint32 LE compressed_size
    //   - 4 bytes: uint32 LE uncompressed_size
    //   - N bytes: compressed SPZ payload
    //   - Padding to 4-byte alignment (glTF requirement)

    static constexpr uint32_t kGspzMagic = 0x5a505347; // "GSPZ" in LE

    const size_t payload_size = chunk.compressed_data.size();
    const size_t header_size = 20; // 5 * uint32
    const size_t total_unpadded = header_size + payload_size;
    const size_t padding = (4 - (total_unpadded % 4)) % 4;
    const size_t total = total_unpadded + padding;

    std::vector<uint8_t> buf;
    buf.reserve(total);

    write_u32_le(buf, kGspzMagic);
    write_u32_le(buf, 1); // version
    write_u32_le(buf, chunk.is_keyframe ? 0u : 1u); // flags
    write_u32_le(buf, static_cast<uint32_t>(payload_size));
    write_u32_le(buf, static_cast<uint32_t>(chunk.uncompressed_size));

    buf.insert(buf.end(), chunk.compressed_data.begin(), chunk.compressed_data.end());

    // Pad to 4-byte alignment with zeros
    for (size_t p = 0; p < padding; p++) {
        buf.push_back(0);
    }

    return buf;
}

} // namespace heimdall::gaussian
