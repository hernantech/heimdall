#include "gltf_gs_writer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <iomanip>

namespace heimdall::gaussian {

// ---------------------------------------------------------------------------
// Helpers — minimal JSON serialisation (no external deps)
// ---------------------------------------------------------------------------

namespace {

// Escape a string for JSON embedding.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// Format a floating-point number avoiding unnecessary trailing zeros.
std::string fmt_float(double v) {
    if (std::isinf(v) || std::isnan(v)) return "0";
    std::ostringstream ss;
    ss << std::setprecision(9) << v;
    return ss.str();
}

// Compute the number of SH coefficients stored per Gaussian given the
// configured SH degree.  Degree d has (d+1)^2 coefficients per colour
// channel; we store 3 channels interleaved.
int sh_coeff_count(int degree) {
    int coeffs_per_channel = (degree + 1) * (degree + 1);
    return coeffs_per_channel * 3;
}

// Pad |size| up to the next multiple of |alignment|.
size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Compute axis-aligned bounding box min/max for the position attribute.
void compute_position_bounds(const std::vector<Gaussian>& gs,
                             float out_min[3], float out_max[3]) {
    if (gs.empty()) {
        out_min[0] = out_min[1] = out_min[2] = 0.0f;
        out_max[0] = out_max[1] = out_max[2] = 0.0f;
        return;
    }
    out_min[0] = out_min[1] = out_min[2] =  std::numeric_limits<float>::max();
    out_max[0] = out_max[1] = out_max[2] = -std::numeric_limits<float>::max();
    for (const auto& g : gs) {
        for (int i = 0; i < 3; ++i) {
            out_min[i] = std::min(out_min[i], g.position[i]);
            out_max[i] = std::max(out_max[i], g.position[i]);
        }
    }
}

// GLB constants.
constexpr uint32_t GLB_MAGIC   = 0x46546C67; // "glTF"
constexpr uint32_t GLB_VERSION = 2;
constexpr uint32_t CHUNK_TYPE_JSON = 0x4E4F534A; // "JSON"
constexpr uint32_t CHUNK_TYPE_BIN  = 0x004E4942; // "BIN\0"

// Append a little-endian uint32 to a byte vector.
void push_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>((v >>  0) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >>  8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

GltfGaussianWriter::GltfGaussianWriter(const GltfGaussianWriterConfig& config)
    : config_(config) {}

// ---------------------------------------------------------------------------
// Raw (uncompressed) attribute binary buffer
// ---------------------------------------------------------------------------

std::vector<uint8_t> GltfGaussianWriter::build_raw_bin(
        const GaussianFrame& frame) {
    const int n = frame.num_gaussians;
    if (n <= 0) return {};

    const int sh_count = sh_coeff_count(config_.sh_degree);

    // Layout (tightly packed, float32):
    //   positions  : n * 3 floats
    //   rotations  : n * 4 floats
    //   scales     : n * 3 floats
    //   opacities  : n * 1 float
    //   sh_coeffs  : n * sh_count floats
    const size_t pos_bytes  = n * 3 * sizeof(float);
    const size_t rot_bytes  = n * 4 * sizeof(float);
    const size_t sc_bytes   = n * 3 * sizeof(float);
    const size_t op_bytes   = n * 1 * sizeof(float);
    const size_t sh_bytes   = n * sh_count * sizeof(float);
    const size_t total      = pos_bytes + rot_bytes + sc_bytes + op_bytes + sh_bytes;

    std::vector<uint8_t> bin(total);
    size_t offset = 0;

    // Positions
    for (int i = 0; i < n; ++i) {
        std::memcpy(bin.data() + offset, frame.gaussians[i].position, 3 * sizeof(float));
        offset += 3 * sizeof(float);
    }

    // Rotations (wxyz quaternion as stored in Gaussian struct)
    for (int i = 0; i < n; ++i) {
        std::memcpy(bin.data() + offset, frame.gaussians[i].rotation, 4 * sizeof(float));
        offset += 4 * sizeof(float);
    }

    // Scales
    for (int i = 0; i < n; ++i) {
        std::memcpy(bin.data() + offset, frame.gaussians[i].scale, 3 * sizeof(float));
        offset += 3 * sizeof(float);
    }

    // Opacities
    for (int i = 0; i < n; ++i) {
        std::memcpy(bin.data() + offset, &frame.gaussians[i].opacity, sizeof(float));
        offset += sizeof(float);
    }

    // SH coefficients (clamp to configured degree)
    for (int i = 0; i < n; ++i) {
        std::memcpy(bin.data() + offset, frame.gaussians[i].sh,
                    sh_count * sizeof(float));
        offset += sh_count * sizeof(float);
    }

    assert(offset == total);
    return bin;
}

// ---------------------------------------------------------------------------
// Raw (uncompressed) glTF JSON
// ---------------------------------------------------------------------------

std::string GltfGaussianWriter::build_raw_json(
        const GaussianFrame& frame, size_t bin_length) {
    const int n = frame.num_gaussians;
    const int sh_count = sh_coeff_count(config_.sh_degree);

    // Byte offsets of each attribute in the buffer.
    const size_t pos_offset = 0;
    const size_t pos_bytes  = n * 3 * sizeof(float);

    const size_t rot_offset = pos_offset + pos_bytes;
    const size_t rot_bytes  = n * 4 * sizeof(float);

    const size_t sc_offset  = rot_offset + rot_bytes;
    const size_t sc_bytes   = n * 3 * sizeof(float);

    const size_t op_offset  = sc_offset + sc_bytes;
    const size_t op_bytes   = n * 1 * sizeof(float);

    const size_t sh_offset  = op_offset + op_bytes;
    const size_t sh_bytes   = n * sh_count * sizeof(float);

    // Position bounds (required by glTF for POSITION accessor).
    float pos_min[3], pos_max[3];
    compute_position_bounds(frame.gaussians, pos_min, pos_max);

    // Accessor indices:
    //   0 = position, 1 = rotation, 2 = scale, 3 = opacity, 4 = sh
    // BufferView indices mirror accessor indices.

    std::ostringstream j;
    j << std::setprecision(9);

    j << "{";

    // --- asset ---
    j << "\"asset\":{\"version\":\"2.0\",\"generator\":\"heimdall\"},";

    // --- extensionsUsed ---
    j << "\"extensionsUsed\":[\"KHR_gaussian_splatting\"],";

    // --- buffers ---
    j << "\"buffers\":[{\"byteLength\":" << bin_length << "}],";

    // --- bufferViews ---
    j << "\"bufferViews\":[";
    j << "{\"buffer\":0,\"byteOffset\":" << pos_offset << ",\"byteLength\":" << pos_bytes << "},";
    j << "{\"buffer\":0,\"byteOffset\":" << rot_offset << ",\"byteLength\":" << rot_bytes << "},";
    j << "{\"buffer\":0,\"byteOffset\":" << sc_offset  << ",\"byteLength\":" << sc_bytes  << "},";
    j << "{\"buffer\":0,\"byteOffset\":" << op_offset  << ",\"byteLength\":" << op_bytes  << "},";
    j << "{\"buffer\":0,\"byteOffset\":" << sh_offset  << ",\"byteLength\":" << sh_bytes  << "}";
    j << "],";

    // --- accessors ---
    // componentType 5126 = FLOAT
    j << "\"accessors\":[";

    // 0: POSITION (VEC3, with min/max)
    j << "{\"bufferView\":0,\"byteOffset\":0,\"componentType\":5126,"
         "\"count\":" << n << ",\"type\":\"VEC3\","
         "\"min\":[" << fmt_float(pos_min[0]) << "," << fmt_float(pos_min[1]) << "," << fmt_float(pos_min[2]) << "],"
         "\"max\":[" << fmt_float(pos_max[0]) << "," << fmt_float(pos_max[1]) << "," << fmt_float(pos_max[2]) << "]},";

    // 1: ROTATION (VEC4 quaternion)
    j << "{\"bufferView\":1,\"byteOffset\":0,\"componentType\":5126,"
         "\"count\":" << n << ",\"type\":\"VEC4\"},";

    // 2: SCALE (VEC3)
    j << "{\"bufferView\":2,\"byteOffset\":0,\"componentType\":5126,"
         "\"count\":" << n << ",\"type\":\"VEC3\"},";

    // 3: OPACITY (SCALAR)
    j << "{\"bufferView\":3,\"byteOffset\":0,\"componentType\":5126,"
         "\"count\":" << n << ",\"type\":\"SCALAR\"},";

    // 4: SH coefficients — stored as VEC3 array, one per (coeff, channel) group
    // Per the KHR_gaussian_splatting spec, SH data is a single accessor of
    // float scalars containing (degree+1)^2 * 3 values per Gaussian.
    j << "{\"bufferView\":4,\"byteOffset\":0,\"componentType\":5126,"
         "\"count\":" << (n * sh_count) << ",\"type\":\"SCALAR\"}";

    j << "],";

    // --- meshes ---
    j << "\"meshes\":[{\"primitives\":[{"
         "\"mode\":0,"
         "\"attributes\":{"
            "\"POSITION\":0,"
            "\"ROTATION\":1,"
            "\"SCALE\":2,"
            "\"OPACITY\":3,"
            "\"SH\":4"
         "},"
         "\"extensions\":{"
            "\"KHR_gaussian_splatting\":{"
                "\"shDegree\":" << config_.sh_degree
         << "}"
         "}"
         "}]}],";

    // --- nodes / scenes ---
    j << "\"nodes\":[{\"mesh\":0}],";
    j << "\"scenes\":[{\"nodes\":[0]}],";
    j << "\"scene\":0";

    j << "}";

    return j.str();
}

// ---------------------------------------------------------------------------
// SPZ-compressed binary buffer
// ---------------------------------------------------------------------------

std::vector<uint8_t> GltfGaussianWriter::build_spz_bin(const SpzChunk& spz) {
    return spz_to_gltf_buffer(spz);
}

// ---------------------------------------------------------------------------
// SPZ-compressed glTF JSON
// ---------------------------------------------------------------------------

std::string GltfGaussianWriter::build_spz_json(
        const GaussianFrame& frame, const SpzChunk& spz,
        size_t bin_length) {
    const int n = frame.num_gaussians;

    // Position bounds (still useful for spatial culling even with SPZ).
    float pos_min[3], pos_max[3];
    compute_position_bounds(frame.gaussians, pos_min, pos_max);

    std::ostringstream j;
    j << std::setprecision(9);

    j << "{";

    // --- asset ---
    j << "\"asset\":{\"version\":\"2.0\",\"generator\":\"heimdall\"},";

    // --- extensionsUsed ---
    j << "\"extensionsUsed\":["
         "\"KHR_gaussian_splatting\","
         "\"KHR_gaussian_splatting_compression_spz\""
         "],";

    // --- buffers ---
    j << "\"buffers\":[{\"byteLength\":" << bin_length << "}],";

    // --- bufferViews ---
    //   0: SPZ compressed blob (the entire BIN chunk)
    j << "\"bufferViews\":[";
    j << "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":" << bin_length << "}";
    j << "],";

    // --- accessors ---
    // For SPZ-compressed data the client decompresses at load time.
    // We still provide a POSITION accessor for bounding-box hints but
    // mark its count; the actual vertex data comes from SPZ decompression.
    // Some renderers use the accessor count to pre-allocate, so we keep it.
    j << "\"accessors\":[";
    j << "{\"bufferView\":0,\"byteOffset\":0,\"componentType\":5126,"
         "\"count\":" << n << ",\"type\":\"VEC3\","
         "\"min\":[" << fmt_float(pos_min[0]) << "," << fmt_float(pos_min[1]) << "," << fmt_float(pos_min[2]) << "],"
         "\"max\":[" << fmt_float(pos_max[0]) << "," << fmt_float(pos_max[1]) << "," << fmt_float(pos_max[2]) << "]}";
    j << "],";

    // --- meshes ---
    j << "\"meshes\":[{\"primitives\":[{"
         "\"mode\":0,"
         "\"attributes\":{"
            "\"POSITION\":0"
         "},"
         "\"extensions\":{"
            "\"KHR_gaussian_splatting\":{"
                "\"shDegree\":" << config_.sh_degree << ","
                "\"extensions\":{"
                    "\"KHR_gaussian_splatting_compression_spz\":{"
                        "\"source\":" << 0 << ","
                        "\"gaussianCount\":" << n << ","
                        "\"isKeyframe\":" << (spz.is_keyframe ? "true" : "false")
                    << "}"
                "}"
            << "}"
         << "}"
         "}]}],";

    // --- nodes / scenes ---
    j << "\"nodes\":[{\"mesh\":0}],";
    j << "\"scenes\":[{\"nodes\":[0]}],";
    j << "\"scene\":0";

    j << "}";

    return j.str();
}

// ---------------------------------------------------------------------------
// GLB assembly
// ---------------------------------------------------------------------------

std::vector<uint8_t> GltfGaussianWriter::assemble_glb(
        const std::string& json, const std::vector<uint8_t>& bin) {
    // JSON chunk: pad with spaces (0x20) to 4-byte alignment.
    const size_t json_raw_len    = json.size();
    const size_t json_padded_len = align_up(json_raw_len, 4);

    // BIN chunk: pad with null bytes (0x00) to 4-byte alignment.
    const size_t bin_raw_len    = bin.size();
    const size_t bin_padded_len = align_up(bin_raw_len, 4);

    // Total file size: 12 (header) + 8 (json chunk header) + json_padded
    //                  + 8 (bin chunk header) + bin_padded
    // If bin is empty, omit the BIN chunk entirely.
    const bool has_bin = !bin.empty();
    const size_t total_length = 12
        + 8 + json_padded_len
        + (has_bin ? (8 + bin_padded_len) : 0);

    std::vector<uint8_t> glb;
    glb.reserve(total_length);

    // --- 12-byte header ---
    push_u32(glb, GLB_MAGIC);
    push_u32(glb, GLB_VERSION);
    push_u32(glb, static_cast<uint32_t>(total_length));

    // --- JSON chunk ---
    push_u32(glb, static_cast<uint32_t>(json_padded_len));
    push_u32(glb, CHUNK_TYPE_JSON);
    glb.insert(glb.end(), json.begin(), json.end());
    // Pad with spaces.
    for (size_t i = json_raw_len; i < json_padded_len; ++i)
        glb.push_back(0x20);

    // --- BIN chunk (optional) ---
    if (has_bin) {
        push_u32(glb, static_cast<uint32_t>(bin_padded_len));
        push_u32(glb, CHUNK_TYPE_BIN);
        glb.insert(glb.end(), bin.begin(), bin.end());
        // Pad with null bytes.
        for (size_t i = bin_raw_len; i < bin_padded_len; ++i)
            glb.push_back(0x00);
    }

    assert(glb.size() == total_length);
    return glb;
}

// ---------------------------------------------------------------------------
// write_frame
// ---------------------------------------------------------------------------

bool GltfGaussianWriter::write_frame(
        const std::string& output_path,
        const GaussianFrame& frame,
        const SpzChunk* spz) {

    std::string json_str;
    std::vector<uint8_t> bin;

    const bool use_spz = (spz != nullptr) && !spz->compressed_data.empty()
                         && config_.prefer_spz_compression;

    // Edge case: zero Gaussians.
    if (frame.num_gaussians <= 0 ||
        frame.gaussians.empty()) {

        // Write a minimal valid GLB with an empty mesh.
        std::ostringstream j;
        j << "{";
        j << "\"asset\":{\"version\":\"2.0\",\"generator\":\"heimdall\"},";
        j << "\"extensionsUsed\":[\"KHR_gaussian_splatting\"],";
        j << "\"meshes\":[{\"primitives\":[{"
              "\"mode\":0,"
              "\"attributes\":{},"
              "\"extensions\":{"
                "\"KHR_gaussian_splatting\":{"
                    "\"shDegree\":" << config_.sh_degree
            << "}"
              "}"
              "}]}],";
        j << "\"nodes\":[{\"mesh\":0}],";
        j << "\"scenes\":[{\"nodes\":[0]}],";
        j << "\"scene\":0";
        j << "}";

        json_str = j.str();
        // No BIN chunk for empty frames.
    } else if (use_spz) {
        bin = build_spz_bin(*spz);
        json_str = build_spz_json(frame, *spz, bin.size());
    } else {
        bin = build_raw_bin(frame);
        json_str = build_raw_json(frame, bin.size());
    }

    std::vector<uint8_t> glb = assemble_glb(json_str, bin);

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(glb.data()),
              static_cast<std::streamsize>(glb.size()));
    return out.good();
}

// ---------------------------------------------------------------------------
// write_manifest
// ---------------------------------------------------------------------------

bool GltfGaussianWriter::write_manifest(
        const std::string& output_path,
        const std::vector<GaussianSegmentInfo>& segments,
        double fps,
        const std::string& version) {

    // Compute totals.
    int64_t total_frames = 0;
    size_t  total_bytes  = 0;
    int64_t avg_gaussians = 0;
    int     avg_count = 0;

    for (const auto& seg : segments) {
        for (const auto& f : seg.frames) {
            ++total_frames;
            total_bytes += f.file_bytes;
            avg_gaussians += f.num_gaussians;
            ++avg_count;
        }
    }

    const double duration_s = (fps > 0) ? (total_frames / fps) : 0.0;
    const int64_t average_gaussians = (avg_count > 0) ? (avg_gaussians / avg_count) : 0;
    const size_t average_frame_bytes = (avg_count > 0) ? (total_bytes / avg_count) : 0;

    std::ostringstream j;
    j << std::setprecision(9);

    j << "{\n";
    j << "    \"$schema\": \"https://heimdall.dev/schemas/manifest/v1\",\n";
    j << "    \"version\": \"" << json_escape(version) << "\",\n";
    j << "    \"type\": \"volumetric-sequence\",\n";
    j << "    \"fps\": " << fmt_float(fps) << ",\n";
    j << "    \"total_frames\": " << total_frames << ",\n";
    j << "    \"duration_s\": " << fmt_float(duration_s) << ",\n";
    j << "    \"representation\": \"gaussian\",\n";

    // Gaussian-specific geometry metadata.
    j << "    \"geometry\": {\n";
    j << "        \"compression\": \"spz\",\n";
    j << "        \"container\": \"glb\",\n";
    j << "        \"extensions\": [\"KHR_gaussian_splatting\", \"KHR_gaussian_splatting_compression_spz\"],\n";
    j << "        \"sh_degree\": " << config_.sh_degree << ",\n";
    j << "        \"average_gaussians\": " << average_gaussians << ",\n";
    j << "        \"average_frame_bytes\": " << average_frame_bytes << "\n";
    j << "    },\n";

    // Segments.
    j << "    \"segments\": [\n";
    for (size_t si = 0; si < segments.size(); ++si) {
        const auto& seg = segments[si];
        j << "        {\n";
        j << "            \"index\": " << si << ",\n";
        j << "            \"start_frame\": " << seg.start_frame << ",\n";
        j << "            \"end_frame\": " << seg.end_frame << ",\n";
        j << "            \"duration_s\": " << fmt_float(seg.duration_s) << ",\n";
        j << "            \"base_url\": \"" << json_escape(seg.base_url) << "\",\n";
        j << "            \"frames\": [\n";
        for (size_t fi = 0; fi < seg.frames.size(); ++fi) {
            const auto& f = seg.frames[fi];
            j << "                {\"frame\": " << f.frame_id
              << ", \"file\": \"" << json_escape(f.file_path)
              << "\", \"bytes\": " << f.file_bytes
              << ", \"gaussians\": " << f.num_gaussians
              << ", \"keyframe\": " << (f.is_keyframe ? "true" : "false")
              << ", \"spz\": " << (f.spz_compressed ? "true" : "false")
              << "}";
            if (fi + 1 < seg.frames.size()) j << ",";
            j << "\n";
        }
        j << "            ],\n";
        j << "            \"total_bytes\": " << seg.total_bytes << "\n";
        j << "        }";
        if (si + 1 < segments.size()) j << ",";
        j << "\n";
    }
    j << "    ]\n";

    j << "}\n";

    std::ofstream out(output_path);
    if (!out.is_open()) return false;
    out << j.str();
    return out.good();
}

} // namespace heimdall::gaussian
