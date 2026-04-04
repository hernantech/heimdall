#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace heimdall::transport {

// SRT (Secure Reliable Transport) based ingest for streaming
// encoded camera feeds from capture machine to RunPod over Tailscale.
//
// Why SRT over WebRTC for ingest:
//   - Fixed-latency, high-bandwidth (WebRTC's adaptive bitrate destroys MVS detail)
//   - Designed for VPN/WAN (exactly Tailscale's use case)
//   - FEC and ARQ for reliable delivery without jitter buffers
//   - No browser dependency on the ingest side
//
// Capture machine (sender):
//   20× NVENC H.265 → SRT multiplex → Tailscale → RunPod
//
// RunPod (receiver):
//   SRT demux → 20× NVDEC decode → CUDA shared memory → pipeline

struct SrtConfig {
    std::string bind_address = "0.0.0.0";
    int port = 9000;
    int latency_ms = 200;           // SRT latency (retransmission window)
    int max_bandwidth_mbps = 100;   // hard cap
    int payload_size = 1316;        // SRT default payload (fits in MTU)
    bool encryption = false;
    std::string passphrase;         // AES-128/256 if encryption=true
    int num_cameras = 20;
};

// Per-camera encoded stream metadata.
struct CameraStreamInfo {
    int camera_index;
    int serial_number;
    int width;
    int height;
    int fps;
    int bitrate_kbps;
    int64_t frame_id;               // latest received frame
    int64_t timestamp_ns;
};

// Encoded frame received from SRT.
struct EncodedFrame {
    int camera_index;
    int64_t frame_id;
    int64_t timestamp_ns;
    std::vector<uint8_t> nal_units; // H.265 NAL units
    bool is_keyframe;
};

using FrameCallback = std::function<void(const EncodedFrame&)>;

// --- Sender (capture machine side) ---

class SrtSender {
public:
    explicit SrtSender(const SrtConfig& config);
    ~SrtSender();

    // Connect to a remote SRT listener (RunPod).
    bool connect(const std::string& remote_host, int remote_port);

    // Send an encoded frame for a specific camera.
    // Non-blocking — queues for transmission.
    bool send_frame(const EncodedFrame& frame);

    // Send all cameras' frames for a single timestep as a batch.
    // Multiplexes with camera_index headers.
    bool send_frame_batch(const std::vector<EncodedFrame>& frames);

    void disconnect();

    // Stats
    double send_bitrate_mbps() const;
    int64_t total_frames_sent() const;
    int retransmit_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --- Receiver (RunPod side) ---

class SrtReceiver {
public:
    explicit SrtReceiver(const SrtConfig& config);
    ~SrtReceiver();

    // Start listening for incoming SRT connections.
    bool listen();

    // Register callback for received frames.
    // Called from the receiver thread — must be thread-safe.
    void set_frame_callback(FrameCallback cb);

    // Start/stop the receiver thread.
    void start();
    void stop();

    // Get latest stream info per camera.
    std::vector<CameraStreamInfo> get_stream_info() const;

    // Stats
    double receive_bitrate_mbps() const;
    int64_t total_frames_received() const;
    int packet_loss_count() const;
    double jitter_ms() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --- Multiplexing ---

// Header prepended to each camera's NAL unit payload in the SRT stream.
// Allows demuxing multiple cameras from a single SRT connection.
struct MuxHeader {
    static constexpr uint32_t MAGIC = 0x48454D43; // "HEMC" (Heimdall Camera)
    uint32_t magic;
    uint16_t camera_index;
    uint16_t flags;             // bit 0: is_keyframe
    uint32_t frame_id;
    uint32_t timestamp_ms;
    uint32_t payload_size;
};
static_assert(sizeof(MuxHeader) == 20, "MuxHeader must be 20 bytes");

// Serialize/deserialize MuxHeader to/from bytes.
std::vector<uint8_t> serialize_mux_header(const MuxHeader& header);
bool deserialize_mux_header(const uint8_t* data, size_t len, MuxHeader& out);

} // namespace heimdall::transport
