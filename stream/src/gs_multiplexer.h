#pragma once

#include "../../gaussian/src/spz_writer.h"
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace heimdall::stream {

// ---------------------------------------------------------------------------
// Wire protocol for the WebRTC data channel.
//
// Every packet on the data channel starts with an 18-byte header:
//   bytes  0- 3  magic        uint32 LE  0x48454D44 ("HEMD")
//   byte   4     packet_type  uint8      0=gaussian_keyframe
//                                        1=gaussian_delta
//                                        2=video_frame
//                                        3=manifest
//   byte   5     flags        uint8      bit 0: has_video
//                                        bit 1: is_final
//   bytes  6- 9  frame_id     uint32 LE
//   bytes 10-13  timestamp_ms uint32 LE
//   bytes 14-17  payload_size uint32 LE
//
// Payload immediately follows the header.
// ---------------------------------------------------------------------------

static constexpr uint32_t kPacketMagic = 0x48454D44;

enum class PacketType : uint8_t {
    GaussianKeyframe = 0,
    GaussianDelta    = 1,
    VideoFrame       = 2,
    Manifest         = 3,
};

enum PacketFlags : uint8_t {
    kFlagHasVideo = 1 << 0,
    kFlagIsFinal  = 1 << 1,
};

static constexpr size_t kPacketHeaderSize = 18;

struct PacketHeader {
    uint32_t magic        = kPacketMagic;
    PacketType type       = PacketType::GaussianKeyframe;
    uint8_t  flags        = 0;
    uint32_t frame_id     = 0;
    uint32_t timestamp_ms = 0;
    uint32_t payload_size = 0;
};

// Serialize / deserialize to the 18-byte wire format (little-endian).
std::vector<uint8_t> serialize_header(const PacketHeader& h);
bool deserialize_header(const uint8_t* data, size_t len, PacketHeader& out);

// A fully serialized packet ready for the transport layer.
struct MuxPacket {
    PacketHeader header;
    std::vector<uint8_t> payload;

    // Convenience: build the complete wire bytes (header + payload).
    std::vector<uint8_t> to_bytes() const;
};

// ---------------------------------------------------------------------------
// H.264 NAL unit wrapper for a single camera's video frame.
// ---------------------------------------------------------------------------
struct VideoFrameData {
    int camera_index;
    uint32_t frame_id;
    uint32_t timestamp_ms;
    std::vector<uint8_t> nal_data;  // one or more NAL units for this frame
};

// ---------------------------------------------------------------------------
// Quality reduction signal sent upstream when the mux is congested.
// ---------------------------------------------------------------------------
enum class QualityAction {
    None,
    DropDeltaFrames,     // keep keyframes only
    ReduceGaussianCount, // request fewer Gaussians from inference
    SkipVideoTracks,     // stop sending video entirely
};

using QualityCallback = std::function<void(QualityAction)>;

// ---------------------------------------------------------------------------
// Stats reported by the multiplexer.
// ---------------------------------------------------------------------------
struct MuxStats {
    int64_t bytes_sent          = 0;
    int64_t gaussian_packets    = 0;
    int64_t video_packets       = 0;
    int64_t dropped_frames      = 0;
    int     queue_depth         = 0;
    double  send_bitrate_kbps   = 0.0;  // estimated over last window
};

// ---------------------------------------------------------------------------
// Configuration for the multiplexer.
// ---------------------------------------------------------------------------
struct MuxConfig {
    // Bitrate budget (kbps) shared across geometry + video tracks.
    int target_bitrate_kbps = 8000;

    // Maximum number of camera video tracks to multiplex.
    // 0 = gaussian-only mode.
    int max_video_tracks = 0;

    // Ordered list of camera indices to include as video tracks.
    // If empty and max_video_tracks > 0, accept any camera up to the limit.
    std::vector<int> video_camera_indices;

    // Maximum milliseconds to hold video frames waiting for their geometry.
    int max_buffer_ms = 500;

    // Send-queue depth at which adaptive quality kicks in.
    int queue_high_water = 30;

    // Send-queue depth at which we start dropping delta frames.
    int queue_critical = 60;

    // Maximum payload size per packet (0 = no limit / no fragmentation).
    size_t max_payload_bytes = 0;
};

// ---------------------------------------------------------------------------
// GaussianStreamMultiplexer
//
// Accepts Gaussian SPZ chunks and optional H.264 camera feeds from the
// pipeline threads, orders them correctly, and produces MuxPackets for
// the network thread to hand off to GStreamer webrtcbin.
//
// Thread safety:
//   - push_gaussian_frame / push_video_frame: called from pipeline thread(s)
//   - get_next_packet / stats: called from the network / main thread
//   - configure / set_quality_callback: call before start, or externally
//     serialized by the caller
// ---------------------------------------------------------------------------
class GaussianStreamMultiplexer {
public:
    GaussianStreamMultiplexer();
    ~GaussianStreamMultiplexer();

    // ---- setup (call before streaming) -----------------------------------

    void configure(const MuxConfig& config);
    void set_quality_callback(QualityCallback cb);

    // Enqueue a manifest packet (sent once at session start, or on change).
    void push_manifest(const std::string& manifest_json);

    // ---- push from pipeline thread(s) ------------------------------------

    // Accept an SPZ-compressed Gaussian chunk for streaming.
    void push_gaussian_frame(const gaussian::SpzChunk& chunk,
                             uint32_t timestamp_ms);

    // Accept an H.264 NAL unit payload for a camera video track.
    void push_video_frame(const VideoFrameData& vf);

    // ---- pull from network thread ----------------------------------------

    // Returns the next packet that should be sent, or std::nullopt if the
    // queue is empty. Non-blocking.
    std::optional<MuxPacket> get_next_packet();

    // ---- stats / control -------------------------------------------------

    MuxStats stats() const;

    // Mark end-of-stream. Pushes a final packet with kFlagIsFinal.
    void finish(uint32_t last_frame_id, uint32_t timestamp_ms);

private:
    // Internal: enqueue a packet, applying adaptive quality rules.
    void enqueue(MuxPacket pkt);

    // Internal: check if video frames that were held can now be released.
    void release_ready_video();

    // Internal: apply adaptive quality pressure.
    // Returns a quality action to signal (or None). Caller fires the
    // callback OUTSIDE the lock to avoid re-entrant deadlock.
    QualityAction check_quality_pressure();

    // Internal: expire held video sets older than max_buffer_ms.
    void expire_held_video();

    // Internal: check if a given camera index is accepted.
    bool is_video_track_accepted(int camera_index) const;

    MuxConfig config_;
    QualityCallback quality_cb_;

    // Protects send_queue_, held_video_, and bookkeeping state.
    mutable std::mutex mu_;

    // Ready-to-send packets, ordered for the transport.
    std::deque<MuxPacket> send_queue_;

    // Video frames held until their corresponding geometry has been enqueued.
    // Keyed by frame_id.
    struct HeldVideoSet {
        uint32_t frame_id;
        uint32_t timestamp_ms;
        std::vector<MuxPacket> packets;
    };
    std::deque<HeldVideoSet> held_video_;

    // Highest geometry frame_id that has been enqueued into send_queue_.
    uint32_t last_geometry_frame_id_ = 0;

    // Bookkeeping for stats.
    std::atomic<int64_t> bytes_sent_{0};
    std::atomic<int64_t> gaussian_packets_{0};
    std::atomic<int64_t> video_packets_{0};
    std::atomic<int64_t> dropped_frames_{0};

    // Sliding window for bitrate estimation.
    struct SendRecord {
        int64_t bytes;
        uint32_t timestamp_ms;
    };
    std::deque<SendRecord> send_log_;
    static constexpr int kBitrateWindowMs = 2000;

    // Quality state: most recently signaled action, to avoid repeats.
    QualityAction last_quality_action_ = QualityAction::None;
};

} // namespace heimdall::stream
