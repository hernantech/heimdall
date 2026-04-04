#pragma once

#include "../../gaussian/src/pipeline.h"
#include <cstdint>
#include <vector>

namespace heimdall::worker {

// Quantized Gaussian transport format.
//
// Raw Gaussian: 62 bytes (3×f32 pos + 3×f32 scale + 4×f32 rot + f32 opacity + 48×f32 SH)
// Quantized DC-only: 20 bytes (3×u16 pos + 3×u16 scale + 4×i8 rot + u8 opacity + 3×u8 SH DC)
// Quantized full SH: 68 bytes (20 + 48×u8 SH)
//
// ~3x compression for DC-only, ~1x for full SH.
// Additional gzip/zstd on the wire for further reduction.

static constexpr uint32_t QUANTIZER_MAGIC = 0x48454947; // "HEIG"
static constexpr uint8_t QUANTIZER_VERSION = 1;

enum QuantizeFlags : uint8_t {
    QFLAG_DC_ONLY = 0x01,   // only SH band-0 (3 coeffs)
    QFLAG_FULL_SH = 0x00,   // all 48 SH coefficients
};

#pragma pack(push, 1)

struct QuantizedHeader {
    uint32_t magic;          // QUANTIZER_MAGIC
    uint8_t version;         // QUANTIZER_VERSION
    uint8_t flags;           // QuantizeFlags
    uint16_t reserved;
    uint32_t num_gaussians;
    // Bounding box for position dequantization
    float bbox_min[3];
    float bbox_max[3];
};
static_assert(sizeof(QuantizedHeader) == 40, "QuantizedHeader must be 40 bytes");

struct QuantizedGaussianDC {
    uint16_t pos[3];         // position quantized to uint16 within bbox
    uint16_t scale[3];       // log-scale quantized to uint16
    int8_t rotation[4];      // quaternion wxyz mapped to [-127, 127]
    uint8_t opacity;         // [0, 255] maps to [0.0, 1.0]
    uint8_t sh_dc[3];        // SH band-0 RGB, maps [-5, 5] to [0, 255]
};
static_assert(sizeof(QuantizedGaussianDC) == 20, "QuantizedGaussianDC must be 20 bytes");

struct QuantizedGaussianFull {
    uint16_t pos[3];
    uint16_t scale[3];
    int8_t rotation[4];
    uint8_t opacity;
    uint8_t sh[48];          // all SH coefficients, maps [-5, 5] to [0, 255]
    uint8_t padding[3];      // align to 4 bytes
};
static_assert(sizeof(QuantizedGaussianFull) == 68, "QuantizedGaussianFull must be 68 bytes");

#pragma pack(pop)

struct QuantizerConfig {
    bool dc_only = true;             // only transmit SH band-0 (saves 45 bytes/Gaussian)
    float sh_range = 5.0f;           // SH values clamped to [-sh_range, sh_range]
    float scale_log_min = -10.0f;    // log(scale) range for quantization
    float scale_log_max = 5.0f;
};

// Quantize a vector of Gaussians into a compact binary blob.
// Returns: header + packed Gaussian data.
std::vector<uint8_t> quantize_gaussians(
    const std::vector<gaussian::Gaussian>& gaussians,
    const QuantizerConfig& config = {}
);

// Dequantize a binary blob back into Gaussians.
// Returns empty vector on invalid data.
std::vector<gaussian::Gaussian> dequantize_gaussians(
    const uint8_t* data,
    size_t size
);

// Read just the header (for routing/stats without full dequantize).
bool read_header(const uint8_t* data, size_t size, QuantizedHeader& out);

} // namespace heimdall::worker
