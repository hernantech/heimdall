#include "quantizer.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace heimdall::worker {

static uint16_t float_to_u16(float val, float min_val, float max_val) {
    float range = max_val - min_val;
    if (range < 1e-8f) return 32768;
    float normalized = (val - min_val) / range;
    normalized = std::clamp(normalized, 0.0f, 1.0f);
    return static_cast<uint16_t>(normalized * 65535.0f + 0.5f);
}

static float u16_to_float(uint16_t val, float min_val, float max_val) {
    float normalized = static_cast<float>(val) / 65535.0f;
    return min_val + normalized * (max_val - min_val);
}

static int8_t float_to_i8(float val) {
    float clamped = std::clamp(val, -1.0f, 1.0f);
    return static_cast<int8_t>(clamped * 127.0f);
}

static float i8_to_float(int8_t val) {
    return static_cast<float>(val) / 127.0f;
}

static uint8_t float_to_u8(float val, float min_val, float max_val) {
    float range = max_val - min_val;
    if (range < 1e-8f) return 128;
    float normalized = (val - min_val) / range;
    normalized = std::clamp(normalized, 0.0f, 1.0f);
    return static_cast<uint8_t>(normalized * 255.0f + 0.5f);
}

static float u8_to_float(uint8_t val, float min_val, float max_val) {
    float normalized = static_cast<float>(val) / 255.0f;
    return min_val + normalized * (max_val - min_val);
}

std::vector<uint8_t> quantize_gaussians(
    const std::vector<gaussian::Gaussian>& gaussians,
    const QuantizerConfig& config
) {
    if (gaussians.empty()) {
        // Header-only for empty frames
        std::vector<uint8_t> result(sizeof(QuantizedHeader), 0);
        auto* hdr = reinterpret_cast<QuantizedHeader*>(result.data());
        hdr->magic = QUANTIZER_MAGIC;
        hdr->version = QUANTIZER_VERSION;
        hdr->flags = config.dc_only ? QFLAG_DC_ONLY : QFLAG_FULL_SH;
        hdr->num_gaussians = 0;
        return result;
    }

    // Compute bounding box
    float bbox_min[3] = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    float bbox_max[3] = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    };

    for (const auto& g : gaussians) {
        for (int i = 0; i < 3; i++) {
            bbox_min[i] = std::min(bbox_min[i], g.position[i]);
            bbox_max[i] = std::max(bbox_max[i], g.position[i]);
        }
    }

    // Pad bbox slightly to avoid edge quantization issues
    for (int i = 0; i < 3; i++) {
        float pad = (bbox_max[i] - bbox_min[i]) * 0.001f + 1e-6f;
        bbox_min[i] -= pad;
        bbox_max[i] += pad;
    }

    // Allocate output
    size_t per_gaussian = config.dc_only ? sizeof(QuantizedGaussianDC) : sizeof(QuantizedGaussianFull);
    size_t total = sizeof(QuantizedHeader) + per_gaussian * gaussians.size();
    std::vector<uint8_t> result(total);

    // Write header
    auto* hdr = reinterpret_cast<QuantizedHeader*>(result.data());
    hdr->magic = QUANTIZER_MAGIC;
    hdr->version = QUANTIZER_VERSION;
    hdr->flags = config.dc_only ? QFLAG_DC_ONLY : QFLAG_FULL_SH;
    hdr->reserved = 0;
    hdr->num_gaussians = static_cast<uint32_t>(gaussians.size());
    std::memcpy(hdr->bbox_min, bbox_min, 12);
    std::memcpy(hdr->bbox_max, bbox_max, 12);

    // Write Gaussians
    uint8_t* write_ptr = result.data() + sizeof(QuantizedHeader);

    for (const auto& g : gaussians) {
        if (config.dc_only) {
            auto* qg = reinterpret_cast<QuantizedGaussianDC*>(write_ptr);
            for (int i = 0; i < 3; i++)
                qg->pos[i] = float_to_u16(g.position[i], bbox_min[i], bbox_max[i]);
            for (int i = 0; i < 3; i++)
                qg->scale[i] = float_to_u16(
                    std::log(std::max(g.scale[i], 1e-8f)),
                    config.scale_log_min, config.scale_log_max
                );
            for (int i = 0; i < 4; i++)
                qg->rotation[i] = float_to_i8(g.rotation[i]);
            qg->opacity = static_cast<uint8_t>(std::clamp(g.opacity, 0.0f, 1.0f) * 255.0f + 0.5f);
            for (int i = 0; i < 3; i++)
                qg->sh_dc[i] = float_to_u8(g.sh[i], -config.sh_range, config.sh_range);

            write_ptr += sizeof(QuantizedGaussianDC);
        } else {
            auto* qg = reinterpret_cast<QuantizedGaussianFull*>(write_ptr);
            for (int i = 0; i < 3; i++)
                qg->pos[i] = float_to_u16(g.position[i], bbox_min[i], bbox_max[i]);
            for (int i = 0; i < 3; i++)
                qg->scale[i] = float_to_u16(
                    std::log(std::max(g.scale[i], 1e-8f)),
                    config.scale_log_min, config.scale_log_max
                );
            for (int i = 0; i < 4; i++)
                qg->rotation[i] = float_to_i8(g.rotation[i]);
            qg->opacity = static_cast<uint8_t>(std::clamp(g.opacity, 0.0f, 1.0f) * 255.0f + 0.5f);
            for (int i = 0; i < 48; i++)
                qg->sh[i] = float_to_u8(g.sh[i], -config.sh_range, config.sh_range);
            qg->padding[0] = qg->padding[1] = qg->padding[2] = 0;

            write_ptr += sizeof(QuantizedGaussianFull);
        }
    }

    return result;
}

std::vector<gaussian::Gaussian> dequantize_gaussians(
    const uint8_t* data,
    size_t size
) {
    QuantizedHeader hdr;
    if (!read_header(data, size, hdr)) return {};
    if (hdr.num_gaussians == 0) return {};

    bool dc_only = (hdr.flags & QFLAG_DC_ONLY) != 0;
    size_t per_gs = dc_only ? sizeof(QuantizedGaussianDC) : sizeof(QuantizedGaussianFull);
    size_t expected = sizeof(QuantizedHeader) + per_gs * hdr.num_gaussians;
    if (size < expected) return {};

    // Default scale log range and SH range (must match encoder)
    float scale_log_min = -10.0f;
    float scale_log_max = 5.0f;
    float sh_range = 5.0f;

    std::vector<gaussian::Gaussian> result(hdr.num_gaussians);
    const uint8_t* read_ptr = data + sizeof(QuantizedHeader);

    for (uint32_t idx = 0; idx < hdr.num_gaussians; idx++) {
        auto& g = result[idx];

        if (dc_only) {
            const auto* qg = reinterpret_cast<const QuantizedGaussianDC*>(read_ptr);
            for (int i = 0; i < 3; i++)
                g.position[i] = u16_to_float(qg->pos[i], hdr.bbox_min[i], hdr.bbox_max[i]);
            for (int i = 0; i < 3; i++)
                g.scale[i] = std::exp(u16_to_float(qg->scale[i], scale_log_min, scale_log_max));
            // Dequantize and renormalize quaternion
            float qnorm = 0.0f;
            for (int i = 0; i < 4; i++) {
                g.rotation[i] = i8_to_float(qg->rotation[i]);
                qnorm += g.rotation[i] * g.rotation[i];
            }
            qnorm = std::sqrt(qnorm);
            if (qnorm > 1e-6f) {
                for (int i = 0; i < 4; i++) g.rotation[i] /= qnorm;
            }
            g.opacity = static_cast<float>(qg->opacity) / 255.0f;
            // SH: DC term only, rest zero
            std::memset(g.sh, 0, sizeof(g.sh));
            for (int i = 0; i < 3; i++)
                g.sh[i] = u8_to_float(qg->sh_dc[i], -sh_range, sh_range);

            read_ptr += sizeof(QuantizedGaussianDC);
        } else {
            const auto* qg = reinterpret_cast<const QuantizedGaussianFull*>(read_ptr);
            for (int i = 0; i < 3; i++)
                g.position[i] = u16_to_float(qg->pos[i], hdr.bbox_min[i], hdr.bbox_max[i]);
            for (int i = 0; i < 3; i++)
                g.scale[i] = std::exp(u16_to_float(qg->scale[i], scale_log_min, scale_log_max));
            float qnorm = 0.0f;
            for (int i = 0; i < 4; i++) {
                g.rotation[i] = i8_to_float(qg->rotation[i]);
                qnorm += g.rotation[i] * g.rotation[i];
            }
            qnorm = std::sqrt(qnorm);
            if (qnorm > 1e-6f) {
                for (int i = 0; i < 4; i++) g.rotation[i] /= qnorm;
            }
            g.opacity = static_cast<float>(qg->opacity) / 255.0f;
            for (int i = 0; i < 48; i++)
                g.sh[i] = u8_to_float(qg->sh[i], -sh_range, sh_range);

            read_ptr += sizeof(QuantizedGaussianFull);
        }
    }

    return result;
}

bool read_header(const uint8_t* data, size_t size, QuantizedHeader& out) {
    if (!data || size < sizeof(QuantizedHeader)) return false;
    std::memcpy(&out, data, sizeof(QuantizedHeader));
    return out.magic == QUANTIZER_MAGIC && out.version == QUANTIZER_VERSION;
}

} // namespace heimdall::worker
