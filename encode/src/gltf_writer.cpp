#include "gltf_writer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace heimdall::encode {

// ---------------------------------------------------------------------------
// Helpers -- minimal JSON serialisation (no external deps)
// ---------------------------------------------------------------------------

namespace {

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

std::string fmt_float(double v) {
    if (std::isinf(v) || std::isnan(v)) return "0";
    std::ostringstream ss;
    ss << std::setprecision(9) << v;
    return ss.str();
}

// Pad |size| up to the next multiple of |alignment|.
size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// GLB constants.
constexpr uint32_t GLB_MAGIC       = 0x46546C67; // "glTF"
constexpr uint32_t GLB_VERSION     = 2;
constexpr uint32_t CHUNK_TYPE_JSON = 0x4E4F534A; // "JSON"
constexpr uint32_t CHUNK_TYPE_BIN  = 0x004E4942; // "BIN\0"

// Append a little-endian uint32 to a byte vector.
void push_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>((v >>  0) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >>  8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

// glTF component type constants.
constexpr int COMPONENT_FLOAT  = 5126;
constexpr int COMPONENT_UINT32 = 5125;
constexpr int COMPONENT_UINT8  = 5121;

// glTF buffer view target constants.
constexpr int TARGET_ARRAY_BUFFER         = 34962;
constexpr int TARGET_ELEMENT_ARRAY_BUFFER = 34963;

// ---------------------------------------------------------------------------
// Compute AABB for position data (interleaved xyz floats).
// ---------------------------------------------------------------------------
void compute_position_bounds(const std::vector<float>& positions,
                             float out_min[3], float out_max[3]) {
    size_t vertex_count = positions.size() / 3;
    if (vertex_count == 0) {
        out_min[0] = out_min[1] = out_min[2] = 0.0f;
        out_max[0] = out_max[1] = out_max[2] = 0.0f;
        return;
    }
    out_min[0] = out_min[1] = out_min[2] =  std::numeric_limits<float>::max();
    out_max[0] = out_max[1] = out_max[2] = -std::numeric_limits<float>::max();
    for (size_t i = 0; i < vertex_count; ++i) {
        for (int c = 0; c < 3; ++c) {
            float val = positions[i * 3 + c];
            out_min[c] = std::min(out_min[c], val);
            out_max[c] = std::max(out_max[c], val);
        }
    }
}

// ---------------------------------------------------------------------------
// Build the binary (BIN) chunk for a mesh frame.
//
// Layout (each bufferView aligned to 4 bytes):
//   [0] Positions:     float32 x 3 x vertex_count
//   [1] Normals:       float32 x 3 x vertex_count          (if present)
//   [2] Texcoords:     float32 x 2 x vertex_count          (if present)
//   [3] Vertex colors: uint8   x 3 x vertex_count + pad    (if present)
//   [4] Indices:       uint32  x index_count
//   [5] Texture image: raw KTX2 bytes                       (if present)
// ---------------------------------------------------------------------------

struct BufferViewInfo {
    size_t byte_offset;
    size_t byte_length;
};

struct BinLayout {
    std::vector<uint8_t> data;

    BufferViewInfo positions;
    BufferViewInfo normals;      // byte_length == 0 if absent
    BufferViewInfo texcoords;    // byte_length == 0 if absent
    BufferViewInfo vertex_colors;// byte_length == 0 if absent
    BufferViewInfo indices;
    BufferViewInfo texture;      // byte_length == 0 if absent
};

BinLayout build_bin(const MeshFrame& frame, const GltfWriterConfig& config) {
    BinLayout layout{};

    const size_t vertex_count = frame.positions.size() / 3;
    const size_t index_count  = frame.indices.size();

    const bool has_normals  = (frame.normals.size() == vertex_count * 3);
    const bool has_texcoords = (frame.texcoords.size() == vertex_count * 2);
    const bool has_colors   = (frame.vertex_colors.size() == vertex_count * 3);
    const bool has_texture  = config.include_texture && !frame.texture_ktx2.empty();

    // Compute sizes and offsets.
    size_t offset = 0;

    // Positions: float32 x 3 x vertex_count
    const size_t pos_bytes = vertex_count * 3 * sizeof(float);
    layout.positions = {offset, pos_bytes};
    offset += align_up(pos_bytes, 4);

    // Normals
    if (has_normals) {
        const size_t norm_bytes = vertex_count * 3 * sizeof(float);
        layout.normals = {offset, norm_bytes};
        offset += align_up(norm_bytes, 4);
    }

    // Texcoords
    if (has_texcoords) {
        const size_t tc_bytes = vertex_count * 2 * sizeof(float);
        layout.texcoords = {offset, tc_bytes};
        offset += align_up(tc_bytes, 4);
    }

    // Vertex colors (uint8 x 3) -- needs padding to 4 bytes
    if (has_colors) {
        const size_t col_bytes = vertex_count * 3 * sizeof(uint8_t);
        layout.vertex_colors = {offset, col_bytes};
        offset += align_up(col_bytes, 4);
    }

    // Indices: uint32 x index_count
    const size_t idx_bytes = index_count * sizeof(uint32_t);
    layout.indices = {offset, idx_bytes};
    offset += align_up(idx_bytes, 4);

    // Texture image (KTX2 blob)
    if (has_texture) {
        layout.texture = {offset, frame.texture_ktx2.size()};
        offset += align_up(frame.texture_ktx2.size(), 4);
    }

    // Allocate and fill the buffer.
    layout.data.resize(offset, 0);

    // Copy positions.
    if (pos_bytes > 0) {
        std::memcpy(layout.data.data() + layout.positions.byte_offset,
                    frame.positions.data(), pos_bytes);
    }

    // Copy normals.
    if (has_normals) {
        std::memcpy(layout.data.data() + layout.normals.byte_offset,
                    frame.normals.data(), layout.normals.byte_length);
    }

    // Copy texcoords.
    if (has_texcoords) {
        std::memcpy(layout.data.data() + layout.texcoords.byte_offset,
                    frame.texcoords.data(), layout.texcoords.byte_length);
    }

    // Copy vertex colors.
    if (has_colors) {
        std::memcpy(layout.data.data() + layout.vertex_colors.byte_offset,
                    frame.vertex_colors.data(), layout.vertex_colors.byte_length);
    }

    // Copy indices.
    if (idx_bytes > 0) {
        std::memcpy(layout.data.data() + layout.indices.byte_offset,
                    frame.indices.data(), idx_bytes);
    }

    // Copy texture.
    if (has_texture) {
        std::memcpy(layout.data.data() + layout.texture.byte_offset,
                    frame.texture_ktx2.data(), layout.texture.byte_length);
    }

    return layout;
}

// ---------------------------------------------------------------------------
// Build the glTF JSON for a mesh frame.
// ---------------------------------------------------------------------------

std::string build_json(const MeshFrame& frame, const BinLayout& layout,
                       const GltfWriterConfig& config) {
    const size_t vertex_count = frame.positions.size() / 3;
    const size_t index_count  = frame.indices.size();

    const bool has_normals   = (layout.normals.byte_length > 0);
    const bool has_texcoords = (layout.texcoords.byte_length > 0);
    const bool has_colors    = (layout.vertex_colors.byte_length > 0);
    const bool has_texture   = (layout.texture.byte_length > 0);

    // Position bounds (required by glTF spec for POSITION accessor).
    float pos_min[3], pos_max[3];
    compute_position_bounds(frame.positions, pos_min, pos_max);

    // Track bufferView and accessor indices as we build them.
    int next_buffer_view = 0;
    int next_accessor    = 0;

    int bv_positions     = -1;
    int bv_normals       = -1;
    int bv_texcoords     = -1;
    int bv_colors        = -1;
    int bv_indices       = -1;
    int bv_texture       = -1;

    int acc_positions    = -1;
    int acc_normals      = -1;
    int acc_texcoords    = -1;
    int acc_colors       = -1;
    int acc_indices      = -1;

    // Assign bufferView indices.
    bv_positions = next_buffer_view++;
    if (has_normals)   bv_normals   = next_buffer_view++;
    if (has_texcoords) bv_texcoords = next_buffer_view++;
    if (has_colors)    bv_colors    = next_buffer_view++;
    bv_indices = next_buffer_view++;
    if (has_texture)   bv_texture   = next_buffer_view++;

    // Assign accessor indices.
    acc_positions = next_accessor++;
    if (has_normals)   acc_normals   = next_accessor++;
    if (has_texcoords) acc_texcoords = next_accessor++;
    if (has_colors)    acc_colors    = next_accessor++;
    acc_indices = next_accessor++;

    std::ostringstream j;
    j << std::setprecision(9);

    j << "{";

    // --- asset ---
    j << "\"asset\":{\"version\":\"2.0\",\"generator\":\"heimdall\"},";

    // --- extensionsUsed ---
    j << "\"extensionsUsed\":[\"KHR_materials_unlit\"";
    if (config.use_meshopt_compression)
        j << ",\"EXT_meshopt_compression\"";
    if (config.use_draco_fallback)
        j << ",\"KHR_draco_mesh_compression\"";
    j << "],";

    // --- extensionsRequired ---
    // KHR_materials_unlit is always required for correct rendering.
    j << "\"extensionsRequired\":[\"KHR_materials_unlit\"";
    if (config.use_meshopt_compression)
        j << ",\"EXT_meshopt_compression\"";
    j << "],";

    // --- buffers ---
    j << "\"buffers\":[{\"byteLength\":" << layout.data.size() << "}],";

    // --- bufferViews ---
    j << "\"bufferViews\":[";

    // Positions bufferView
    j << "{\"buffer\":0"
      << ",\"byteOffset\":" << layout.positions.byte_offset
      << ",\"byteLength\":" << layout.positions.byte_length
      << ",\"byteStride\":12"
      << ",\"target\":" << TARGET_ARRAY_BUFFER
      << "}";

    // Normals bufferView
    if (has_normals) {
        j << ",{\"buffer\":0"
          << ",\"byteOffset\":" << layout.normals.byte_offset
          << ",\"byteLength\":" << layout.normals.byte_length
          << ",\"byteStride\":12"
          << ",\"target\":" << TARGET_ARRAY_BUFFER
          << "}";
    }

    // Texcoords bufferView
    if (has_texcoords) {
        j << ",{\"buffer\":0"
          << ",\"byteOffset\":" << layout.texcoords.byte_offset
          << ",\"byteLength\":" << layout.texcoords.byte_length
          << ",\"byteStride\":8"
          << ",\"target\":" << TARGET_ARRAY_BUFFER
          << "}";
    }

    // Vertex colors bufferView
    if (has_colors) {
        j << ",{\"buffer\":0"
          << ",\"byteOffset\":" << layout.vertex_colors.byte_offset
          << ",\"byteLength\":" << layout.vertex_colors.byte_length
          << ",\"byteStride\":3"
          << ",\"target\":" << TARGET_ARRAY_BUFFER
          << "}";
    }

    // Indices bufferView
    j << ",{\"buffer\":0"
      << ",\"byteOffset\":" << layout.indices.byte_offset
      << ",\"byteLength\":" << layout.indices.byte_length
      << ",\"target\":" << TARGET_ELEMENT_ARRAY_BUFFER
      << "}";

    // Texture image bufferView (no target -- it's an image, not a vertex attribute)
    if (has_texture) {
        j << ",{\"buffer\":0"
          << ",\"byteOffset\":" << layout.texture.byte_offset
          << ",\"byteLength\":" << layout.texture.byte_length
          << "}";
    }

    j << "],";

    // --- accessors ---
    j << "\"accessors\":[";

    // POSITION (VEC3 float, with min/max)
    j << "{\"bufferView\":" << bv_positions
      << ",\"byteOffset\":0"
      << ",\"componentType\":" << COMPONENT_FLOAT
      << ",\"count\":" << vertex_count
      << ",\"type\":\"VEC3\""
      << ",\"min\":[" << fmt_float(pos_min[0]) << "," << fmt_float(pos_min[1]) << "," << fmt_float(pos_min[2]) << "]"
      << ",\"max\":[" << fmt_float(pos_max[0]) << "," << fmt_float(pos_max[1]) << "," << fmt_float(pos_max[2]) << "]"
      << "}";

    // NORMAL (VEC3 float)
    if (has_normals) {
        j << ",{\"bufferView\":" << bv_normals
          << ",\"byteOffset\":0"
          << ",\"componentType\":" << COMPONENT_FLOAT
          << ",\"count\":" << vertex_count
          << ",\"type\":\"VEC3\""
          << "}";
    }

    // TEXCOORD_0 (VEC2 float)
    if (has_texcoords) {
        j << ",{\"bufferView\":" << bv_texcoords
          << ",\"byteOffset\":0"
          << ",\"componentType\":" << COMPONENT_FLOAT
          << ",\"count\":" << vertex_count
          << ",\"type\":\"VEC2\""
          << "}";
    }

    // COLOR_0 (VEC3 uint8 normalized)
    if (has_colors) {
        j << ",{\"bufferView\":" << bv_colors
          << ",\"byteOffset\":0"
          << ",\"componentType\":" << COMPONENT_UINT8
          << ",\"count\":" << vertex_count
          << ",\"type\":\"VEC3\""
          << ",\"normalized\":true"
          << "}";
    }

    // Indices (SCALAR uint32)
    j << ",{\"bufferView\":" << bv_indices
      << ",\"byteOffset\":0"
      << ",\"componentType\":" << COMPONENT_UINT32
      << ",\"count\":" << index_count
      << ",\"type\":\"SCALAR\""
      << "}";

    j << "],";

    // --- images (optional) ---
    if (has_texture) {
        j << "\"images\":[{\"bufferView\":" << bv_texture
          << ",\"mimeType\":\"image/ktx2\""
          << "}],";
    }

    // --- textures (optional) ---
    if (has_texture) {
        j << "\"textures\":[{\"source\":0}],";
    }

    // --- materials ---
    j << "\"materials\":[{";
    j << "\"pbrMetallicRoughness\":{";
    if (has_texture) {
        j << "\"baseColorTexture\":{\"index\":0},";
    } else if (has_colors) {
        // No texture -- vertex colors serve as base color; set white factor.
        j << "\"baseColorFactor\":[1.0,1.0,1.0,1.0],";
    }
    j << "\"metallicFactor\":0,\"roughnessFactor\":1";
    j << "},";
    j << "\"extensions\":{\"KHR_materials_unlit\":{}}";
    j << "}],";

    // --- meshes ---
    j << "\"meshes\":[{\"primitives\":[{";
    j << "\"mode\":4,";  // TRIANGLES
    j << "\"attributes\":{";
    j << "\"POSITION\":" << acc_positions;
    if (has_normals)
        j << ",\"NORMAL\":" << acc_normals;
    if (has_texcoords)
        j << ",\"TEXCOORD_0\":" << acc_texcoords;
    if (has_colors)
        j << ",\"COLOR_0\":" << acc_colors;
    j << "},";
    j << "\"indices\":" << acc_indices << ",";
    j << "\"material\":0";
    j << "}]}],";

    // --- nodes / scenes ---
    j << "\"nodes\":[{\"mesh\":0}],";
    j << "\"scenes\":[{\"nodes\":[0]}],";
    j << "\"scene\":0";

    j << "}";

    return j.str();
}

// ---------------------------------------------------------------------------
// Build JSON for an empty mesh frame (zero vertices/indices).
// ---------------------------------------------------------------------------

std::string build_empty_json(const GltfWriterConfig& config) {
    std::ostringstream j;

    j << "{";
    j << "\"asset\":{\"version\":\"2.0\",\"generator\":\"heimdall\"},";
    j << "\"extensionsUsed\":[\"KHR_materials_unlit\"";
    if (config.use_meshopt_compression)
        j << ",\"EXT_meshopt_compression\"";
    if (config.use_draco_fallback)
        j << ",\"KHR_draco_mesh_compression\"";
    j << "],";
    j << "\"extensionsRequired\":[\"KHR_materials_unlit\"";
    if (config.use_meshopt_compression)
        j << ",\"EXT_meshopt_compression\"";
    j << "],";
    j << "\"materials\":[{";
    j << "\"pbrMetallicRoughness\":{\"metallicFactor\":0,\"roughnessFactor\":1},";
    j << "\"extensions\":{\"KHR_materials_unlit\":{}}";
    j << "}],";
    j << "\"meshes\":[{\"primitives\":[{"
          "\"mode\":4,"
          "\"attributes\":{}"
          "}]}],";
    j << "\"nodes\":[{\"mesh\":0}],";
    j << "\"scenes\":[{\"nodes\":[0]}],";
    j << "\"scene\":0";
    j << "}";

    return j.str();
}

// ---------------------------------------------------------------------------
// GLB assembly (identical pattern to gltf_gs_writer.cpp).
// ---------------------------------------------------------------------------

std::vector<uint8_t> assemble_glb(const std::string& json,
                                  const std::vector<uint8_t>& bin) {
    const size_t json_raw_len    = json.size();
    const size_t json_padded_len = align_up(json_raw_len, 4);

    const size_t bin_raw_len    = bin.size();
    const size_t bin_padded_len = align_up(bin_raw_len, 4);

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
    for (size_t i = json_raw_len; i < json_padded_len; ++i)
        glb.push_back(0x20); // space pad

    // --- BIN chunk (optional) ---
    if (has_bin) {
        push_u32(glb, static_cast<uint32_t>(bin_padded_len));
        push_u32(glb, CHUNK_TYPE_BIN);
        glb.insert(glb.end(), bin.begin(), bin.end());
        for (size_t i = bin_raw_len; i < bin_padded_len; ++i)
            glb.push_back(0x00); // null pad
    }

    assert(glb.size() == total_length);
    return glb;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// write_gltf_frame
// ---------------------------------------------------------------------------

bool write_gltf_frame(
        const std::string& output_path,
        const MeshFrame& frame,
        const GltfWriterConfig& config) {

    std::string json_str;
    std::vector<uint8_t> bin;

    const size_t vertex_count = frame.positions.size() / 3;
    const size_t index_count  = frame.indices.size();

    // Edge case: empty mesh (no vertices or no indices).
    if (vertex_count == 0 || index_count == 0) {
        json_str = build_empty_json(config);
        // No BIN chunk for empty frames.
    } else {
        BinLayout layout = build_bin(frame, config);
        json_str = build_json(frame, layout, config);
        bin = std::move(layout.data);
    }

    std::vector<uint8_t> glb = assemble_glb(json_str, bin);

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(glb.data()),
              static_cast<std::streamsize>(glb.size()));
    return out.good();
}

// ---------------------------------------------------------------------------
// write_stream_manifest
// ---------------------------------------------------------------------------

bool write_stream_manifest(
        const std::string& output_path,
        const std::vector<SegmentInfo>& segments,
        double fps,
        const std::string& version) {

    // Compute aggregate statistics.
    int64_t total_frames = 0;
    size_t  total_bytes  = 0;

    for (const auto& seg : segments) {
        int64_t seg_frames = seg.end_frame - seg.start_frame + 1;
        total_frames += seg_frames;
        total_bytes  += seg.total_bytes;
    }

    // For average vertices/faces we estimate from total_bytes and frame count.
    // In the manifest schema the caller can populate these from the actual frames;
    // here we provide best-effort from SegmentInfo.
    const size_t average_frame_bytes = (total_frames > 0)
        ? (total_bytes / static_cast<size_t>(total_frames)) : 0;
    const double duration_s = (fps > 0) ? (total_frames / fps) : 0.0;

    std::ostringstream j;
    j << std::setprecision(9);

    j << "{\n";
    j << "    \"$schema\": \"https://heimdall.dev/schemas/manifest/v1\",\n";
    j << "    \"version\": \"" << json_escape(version) << "\",\n";
    j << "    \"type\": \"volumetric-sequence\",\n";
    j << "    \"fps\": " << fmt_float(fps) << ",\n";
    j << "    \"total_frames\": " << total_frames << ",\n";
    j << "    \"duration_s\": " << fmt_float(duration_s) << ",\n";
    j << "    \"representation\": \"mesh\",\n";

    // Geometry metadata.
    j << "    \"geometry\": {\n";
    j << "        \"compression\": \"meshopt\",\n";
    j << "        \"container\": \"glb\",\n";
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
        j << "            \"frames\": [\n";
        for (size_t fi = 0; fi < seg.frame_paths.size(); ++fi) {
            j << "                {\"frame\": " << (seg.start_frame + static_cast<int64_t>(fi))
              << ", \"file\": \"" << json_escape(seg.frame_paths[fi]) << "\"}";
            if (fi + 1 < seg.frame_paths.size()) j << ",";
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

} // namespace heimdall::encode
