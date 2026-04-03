#pragma once

#include "pipeline.h"
#include "spz_writer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace heimdall::gaussian {

// Configuration for the Gaussian glTF/GLB writer.
struct GltfGaussianWriterConfig {
    bool prefer_spz_compression = true;   // use SPZ extension when SpzChunk provided
    int sh_degree = 3;                    // spherical harmonics degree (0-3)
};

// Metadata about a single written frame, used for manifest generation.
struct GaussianFrameMeta {
    int64_t frame_id;
    int64_t timestamp_ns;
    int num_gaussians;
    bool is_keyframe;
    bool spz_compressed;
    std::string file_path;   // relative path to the .glb file
    size_t file_bytes;       // size of the .glb on disk
};

// Segment of consecutive frames for the streaming manifest.
struct GaussianSegmentInfo {
    int64_t start_frame;
    int64_t end_frame;
    double duration_s;
    std::string base_url;    // relative directory for this segment
    std::vector<GaussianFrameMeta> frames;
    size_t total_bytes;
};

// Writes per-frame .glb files containing Gaussian splat data using the
// KHR_gaussian_splatting and KHR_gaussian_splatting_compression_spz
// extensions.  Dependency-free: constructs glTF JSON and GLB binary
// manually without tinygltf or any other external library.
class GltfGaussianWriter {
public:
    explicit GltfGaussianWriter(const GltfGaussianWriterConfig& config = {});
    ~GltfGaussianWriter() = default;

    // Write a single frame to a .glb file.
    //
    // If |spz| is non-null, the compressed SPZ blob is embedded via
    // KHR_gaussian_splatting_compression_spz and the raw attributes are
    // omitted.  Otherwise raw attributes (POSITION, ROTATION, SCALE,
    // OPACITY, SH coefficients) are written as typed accessors.
    //
    // Returns true on success.  Writes an empty-frame stub for frames
    // with zero Gaussians.
    bool write_frame(
        const std::string& output_path,
        const GaussianFrame& frame,
        const SpzChunk* spz = nullptr
    );

    // Write a streaming manifest JSON for sequenced playback.
    // Compatible with heimdall's manifest format but with
    // representation="gaussian".
    bool write_manifest(
        const std::string& output_path,
        const std::vector<GaussianSegmentInfo>& segments,
        double fps,
        const std::string& version = "1.0"
    );

private:
    GltfGaussianWriterConfig config_;

    // Build glTF JSON + binary buffer for raw (uncompressed) Gaussian data.
    std::string build_raw_json(const GaussianFrame& frame,
                               size_t bin_length);
    std::vector<uint8_t> build_raw_bin(const GaussianFrame& frame);

    // Build glTF JSON + binary buffer for SPZ-compressed Gaussian data.
    std::string build_spz_json(const GaussianFrame& frame,
                               const SpzChunk& spz,
                               size_t bin_length);
    std::vector<uint8_t> build_spz_bin(const SpzChunk& spz);

    // Assemble a complete GLB file from JSON and BIN payloads.
    static std::vector<uint8_t> assemble_glb(const std::string& json,
                                             const std::vector<uint8_t>& bin);
};

} // namespace heimdall::gaussian
