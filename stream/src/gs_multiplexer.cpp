#include "gs_multiplexer.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace heimdall::stream {

// ---------------------------------------------------------------------------
// Wire format helpers
// ---------------------------------------------------------------------------

namespace {

void write_u32_le(uint8_t* dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>(v);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
}

uint32_t read_u32_le(const uint8_t* src) {
    return static_cast<uint32_t>(src[0])
         | (static_cast<uint32_t>(src[1]) << 8)
         | (static_cast<uint32_t>(src[2]) << 16)
         | (static_cast<uint32_t>(src[3]) << 24);
}

} // anonymous namespace

std::vector<uint8_t> serialize_header(const PacketHeader& h) {
    std::vector<uint8_t> buf(kPacketHeaderSize);
    write_u32_le(&buf[0], h.magic);
    buf[4] = static_cast<uint8_t>(h.type);
    buf[5] = h.flags;
    write_u32_le(&buf[6], h.frame_id);
    write_u32_le(&buf[10], h.timestamp_ms);
    write_u32_le(&buf[14], h.payload_size);
    return buf;
}

bool deserialize_header(const uint8_t* data, size_t len, PacketHeader& out) {
    if (len < kPacketHeaderSize) return false;

    out.magic        = read_u32_le(&data[0]);
    if (out.magic != kPacketMagic) return false;

    out.type         = static_cast<PacketType>(data[4]);
    out.flags        = data[5];
    out.frame_id     = read_u32_le(&data[6]);
    out.timestamp_ms = read_u32_le(&data[10]);
    out.payload_size = read_u32_le(&data[14]);
    return true;
}

std::vector<uint8_t> MuxPacket::to_bytes() const {
    auto hdr = serialize_header(header);
    hdr.insert(hdr.end(), payload.begin(), payload.end());
    return hdr;
}

// ---------------------------------------------------------------------------
// GaussianStreamMultiplexer
// ---------------------------------------------------------------------------

GaussianStreamMultiplexer::GaussianStreamMultiplexer() = default;
GaussianStreamMultiplexer::~GaussianStreamMultiplexer() = default;

void GaussianStreamMultiplexer::configure(const MuxConfig& config) {
    config_ = config;
}

void GaussianStreamMultiplexer::set_quality_callback(QualityCallback cb) {
    quality_cb_ = std::move(cb);
}

// ---------------------------------------------------------------------------
// push_manifest
// ---------------------------------------------------------------------------

void GaussianStreamMultiplexer::push_manifest(const std::string& manifest_json) {
    MuxPacket pkt;
    pkt.header.type         = PacketType::Manifest;
    pkt.header.flags        = 0;
    pkt.header.frame_id     = 0;
    pkt.header.timestamp_ms = 0;
    pkt.payload.assign(manifest_json.begin(), manifest_json.end());
    pkt.header.payload_size = static_cast<uint32_t>(pkt.payload.size());

    enqueue(std::move(pkt));
}

// ---------------------------------------------------------------------------
// push_gaussian_frame
// ---------------------------------------------------------------------------

void GaussianStreamMultiplexer::push_gaussian_frame(
        const gaussian::SpzChunk& chunk,
        uint32_t timestamp_ms) {

    // Empty chunks are silently ignored.
    if (chunk.compressed_data.empty()) return;

    MuxPacket pkt;
    pkt.header.type = chunk.is_keyframe
                        ? PacketType::GaussianKeyframe
                        : PacketType::GaussianDelta;
    pkt.header.flags = (config_.max_video_tracks > 0)
                        ? kFlagHasVideo : static_cast<uint8_t>(0);
    pkt.header.frame_id     = static_cast<uint32_t>(chunk.start_frame);
    pkt.header.timestamp_ms = timestamp_ms;
    pkt.payload             = chunk.compressed_data;
    pkt.header.payload_size = static_cast<uint32_t>(pkt.payload.size());

    QualityAction signal = QualityAction::None;

    {
        std::lock_guard<std::mutex> lock(mu_);

        // Adaptive quality: under critical pressure drop delta frames.
        if (!chunk.is_keyframe
            && static_cast<int>(send_queue_.size()) >= config_.queue_critical) {
            dropped_frames_.fetch_add(1, std::memory_order_relaxed);
            signal = check_quality_pressure();
        } else {
            send_queue_.push_back(std::move(pkt));
            gaussian_packets_.fetch_add(1, std::memory_order_relaxed);

            // Track the latest geometry frame_id so held video can be released.
            uint32_t fid = static_cast<uint32_t>(chunk.start_frame);
            if (fid > last_geometry_frame_id_) {
                last_geometry_frame_id_ = fid;
            }

            release_ready_video();
            signal = check_quality_pressure();
        }
    }

    // Fire quality callback outside the lock to avoid re-entrant deadlock.
    if (signal != QualityAction::None && quality_cb_) {
        quality_cb_(signal);
    }
}

// ---------------------------------------------------------------------------
// push_video_frame
// ---------------------------------------------------------------------------

void GaussianStreamMultiplexer::push_video_frame(const VideoFrameData& vf) {
    // If video is disabled, silently drop.
    if (config_.max_video_tracks == 0) return;

    // Check if this camera is accepted.
    if (!is_video_track_accepted(vf.camera_index)) return;

    // Empty NAL data is silently ignored.
    if (vf.nal_data.empty()) return;

    // Build the payload: [camera_index LE u32][NAL data...]
    // This lets the receiver demultiplex video tracks.
    std::vector<uint8_t> prefixed(4 + vf.nal_data.size());
    uint8_t ci[4];
    write_u32_le(ci, static_cast<uint32_t>(vf.camera_index));
    std::memcpy(prefixed.data(), ci, 4);
    std::memcpy(prefixed.data() + 4, vf.nal_data.data(), vf.nal_data.size());

    MuxPacket pkt;
    pkt.header.type         = PacketType::VideoFrame;
    pkt.header.flags        = kFlagHasVideo;
    pkt.header.frame_id     = vf.frame_id;
    pkt.header.timestamp_ms = vf.timestamp_ms;
    pkt.payload             = std::move(prefixed);
    pkt.header.payload_size = static_cast<uint32_t>(pkt.payload.size());

    QualityAction signal = QualityAction::None;

    {
        std::lock_guard<std::mutex> lock(mu_);

        // If geometry for this frame has already been enqueued, send directly.
        if (vf.frame_id <= last_geometry_frame_id_) {
            send_queue_.push_back(std::move(pkt));
            video_packets_.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Hold until the geometry arrives (or max_buffer_ms expires).
            auto it = std::find_if(held_video_.begin(), held_video_.end(),
                [&](const HeldVideoSet& h) { return h.frame_id == vf.frame_id; });
            if (it != held_video_.end()) {
                it->packets.push_back(std::move(pkt));
            } else {
                HeldVideoSet hvs;
                hvs.frame_id     = vf.frame_id;
                hvs.timestamp_ms = vf.timestamp_ms;
                hvs.packets.push_back(std::move(pkt));
                held_video_.push_back(std::move(hvs));
            }

            // Expire held video that has waited too long.
            expire_held_video();
        }

        signal = check_quality_pressure();
    }

    if (signal != QualityAction::None && quality_cb_) {
        quality_cb_(signal);
    }
}

// ---------------------------------------------------------------------------
// get_next_packet (called from network thread)
// ---------------------------------------------------------------------------

std::optional<MuxPacket> GaussianStreamMultiplexer::get_next_packet() {
    std::lock_guard<std::mutex> lock(mu_);

    if (send_queue_.empty()) return std::nullopt;

    MuxPacket pkt = std::move(send_queue_.front());
    send_queue_.pop_front();

    // Update byte accounting.
    int64_t pkt_bytes = static_cast<int64_t>(kPacketHeaderSize + pkt.payload.size());
    bytes_sent_.fetch_add(pkt_bytes, std::memory_order_relaxed);

    // Record for bitrate estimation.
    send_log_.push_back({pkt_bytes, pkt.header.timestamp_ms});

    // Prune old entries outside the bitrate window.
    uint32_t newest = pkt.header.timestamp_ms;
    while (!send_log_.empty()
           && newest - send_log_.front().timestamp_ms
                  > static_cast<uint32_t>(kBitrateWindowMs)) {
        send_log_.pop_front();
    }

    return pkt;
}

// ---------------------------------------------------------------------------
// finish
// ---------------------------------------------------------------------------

void GaussianStreamMultiplexer::finish(uint32_t last_frame_id, uint32_t timestamp_ms) {
    std::lock_guard<std::mutex> lock(mu_);

    // Flush any remaining held video.
    for (auto& hvs : held_video_) {
        for (auto& p : hvs.packets) {
            send_queue_.push_back(std::move(p));
            video_packets_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    held_video_.clear();

    // Push a zero-payload final packet.
    MuxPacket fin;
    fin.header.type         = PacketType::GaussianKeyframe;
    fin.header.flags        = kFlagIsFinal;
    fin.header.frame_id     = last_frame_id;
    fin.header.timestamp_ms = timestamp_ms;
    fin.header.payload_size = 0;
    send_queue_.push_back(std::move(fin));
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------

MuxStats GaussianStreamMultiplexer::stats() const {
    MuxStats s;
    s.bytes_sent       = bytes_sent_.load(std::memory_order_relaxed);
    s.gaussian_packets = gaussian_packets_.load(std::memory_order_relaxed);
    s.video_packets    = video_packets_.load(std::memory_order_relaxed);
    s.dropped_frames   = dropped_frames_.load(std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(mu_);
        s.queue_depth = static_cast<int>(send_queue_.size());

        // Estimate bitrate over the sliding window.
        if (send_log_.size() >= 2) {
            uint32_t newest = send_log_.back().timestamp_ms;
            int64_t window_bytes = 0;
            uint32_t oldest = newest;
            for (auto it = send_log_.rbegin(); it != send_log_.rend(); ++it) {
                if (newest - it->timestamp_ms
                        > static_cast<uint32_t>(kBitrateWindowMs)) {
                    break;
                }
                window_bytes += it->bytes;
                oldest = it->timestamp_ms;
            }
            uint32_t span = newest - oldest;
            if (span > 0) {
                s.send_bitrate_kbps =
                    static_cast<double>(window_bytes) * 8.0
                    / static_cast<double>(span);
            }
        }
    }

    return s;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

void GaussianStreamMultiplexer::enqueue(MuxPacket pkt) {
    std::lock_guard<std::mutex> lock(mu_);
    send_queue_.push_back(std::move(pkt));
}

void GaussianStreamMultiplexer::release_ready_video() {
    // Called with mu_ held.
    auto it = held_video_.begin();
    while (it != held_video_.end()) {
        if (it->frame_id <= last_geometry_frame_id_) {
            for (auto& p : it->packets) {
                send_queue_.push_back(std::move(p));
                video_packets_.fetch_add(1, std::memory_order_relaxed);
            }
            it = held_video_.erase(it);
        } else {
            ++it;
        }
    }
}

void GaussianStreamMultiplexer::expire_held_video() {
    // Called with mu_ held.
    if (held_video_.empty()) return;

    // Find a reference timestamp from the most recent geometry packet.
    uint32_t ref_ts = 0;
    for (auto rit = send_queue_.rbegin(); rit != send_queue_.rend(); ++rit) {
        if (rit->header.type == PacketType::GaussianKeyframe ||
            rit->header.type == PacketType::GaussianDelta) {
            ref_ts = rit->header.timestamp_ms;
            break;
        }
    }
    // Fallback: use the newest held video timestamp.
    if (ref_ts == 0 && !held_video_.empty()) {
        ref_ts = held_video_.back().timestamp_ms;
    }
    if (ref_ts == 0) return;

    auto it = held_video_.begin();
    while (it != held_video_.end()) {
        if (ref_ts > it->timestamp_ms &&
            (ref_ts - it->timestamp_ms)
                > static_cast<uint32_t>(config_.max_buffer_ms)) {
            dropped_frames_.fetch_add(
                static_cast<int64_t>(it->packets.size()),
                std::memory_order_relaxed);
            it = held_video_.erase(it);
        } else {
            ++it;
        }
    }
}

QualityAction GaussianStreamMultiplexer::check_quality_pressure() {
    // Called with mu_ held.
    // Returns the action to signal to the caller. The caller is responsible
    // for invoking quality_cb_ OUTSIDE the lock.
    int depth = static_cast<int>(send_queue_.size());

    QualityAction action = QualityAction::None;

    if (depth >= config_.queue_critical) {
        if (config_.max_video_tracks > 0) {
            action = QualityAction::SkipVideoTracks;
        } else {
            action = QualityAction::ReduceGaussianCount;
        }
    } else if (depth >= config_.queue_high_water) {
        action = QualityAction::DropDeltaFrames;
    }

    // Only signal transitions (avoid repeating the same action).
    if (action != QualityAction::None && action != last_quality_action_) {
        last_quality_action_ = action;
        return action;
    }

    // Reset quality state when queue drains below high-water, so that
    // the next pressure spike will trigger a fresh callback.
    if (depth < config_.queue_high_water
        && last_quality_action_ != QualityAction::None) {
        last_quality_action_ = QualityAction::None;
    }

    return QualityAction::None;
}

bool GaussianStreamMultiplexer::is_video_track_accepted(int camera_index) const {
    if (config_.max_video_tracks <= 0) return false;

    // If an explicit allow-list is configured, use it.
    if (!config_.video_camera_indices.empty()) {
        return std::find(config_.video_camera_indices.begin(),
                         config_.video_camera_indices.end(),
                         camera_index)
               != config_.video_camera_indices.end();
    }

    // Otherwise accept any camera index in [0, max_video_tracks).
    return camera_index >= 0 && camera_index < config_.max_video_tracks;
}

} // namespace heimdall::stream
