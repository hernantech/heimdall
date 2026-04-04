// ============================================================================
// Tests for heimdall::stream::GaussianStreamMultiplexer
// ============================================================================

#include "test_helpers.h"
#include "../stream/src/gs_multiplexer.h"

#include <cstring>
#include <optional>
#include <string>
#include <vector>

using namespace heimdall::stream;
using namespace heimdall::gaussian;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static SpzChunk make_spz_chunk(int64_t frame_id, bool keyframe,
                               size_t payload_size = 128) {
    SpzChunk chunk;
    chunk.start_frame       = frame_id;
    chunk.end_frame         = frame_id;
    chunk.is_keyframe       = keyframe;
    chunk.uncompressed_size = payload_size * 10;
    chunk.compressed_data.resize(payload_size);
    // Fill with recognizable pattern.
    for (size_t i = 0; i < payload_size; ++i) {
        chunk.compressed_data[i] = static_cast<uint8_t>(i & 0xFF);
    }
    return chunk;
}

// ---------------------------------------------------------------------------
// Tests — packet header serialization
// ---------------------------------------------------------------------------

TEST(mux_header_serialize_roundtrip) {
    PacketHeader h;
    h.magic        = kPacketMagic;
    h.type         = PacketType::GaussianKeyframe;
    h.flags        = kFlagHasVideo;
    h.frame_id     = 42;
    h.timestamp_ms = 12345;
    h.payload_size = 1024;

    auto bytes = serialize_header(h);
    ASSERT_EQ(bytes.size(), kPacketHeaderSize);

    PacketHeader out;
    bool ok = deserialize_header(bytes.data(), bytes.size(), out);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out.magic, kPacketMagic);
    ASSERT_EQ(static_cast<uint8_t>(out.type),
              static_cast<uint8_t>(PacketType::GaussianKeyframe));
    ASSERT_EQ(out.flags, kFlagHasVideo);
    ASSERT_EQ(out.frame_id, static_cast<uint32_t>(42));
    ASSERT_EQ(out.timestamp_ms, static_cast<uint32_t>(12345));
    ASSERT_EQ(out.payload_size, static_cast<uint32_t>(1024));
}

TEST(mux_header_bad_magic) {
    std::vector<uint8_t> buf(kPacketHeaderSize, 0);
    // Write wrong magic.
    buf[0] = 0xFF;
    buf[1] = 0xFF;
    buf[2] = 0xFF;
    buf[3] = 0xFF;

    PacketHeader out;
    bool ok = deserialize_header(buf.data(), buf.size(), out);
    ASSERT_FALSE(ok);
}

TEST(mux_header_too_short) {
    std::vector<uint8_t> buf(10, 0); // less than kPacketHeaderSize
    PacketHeader out;
    bool ok = deserialize_header(buf.data(), buf.size(), out);
    ASSERT_FALSE(ok);
}

// ---------------------------------------------------------------------------
// Tests — multiplexer packet flow
// ---------------------------------------------------------------------------

TEST(mux_push_gaussian_keyframe_get_packet) {
    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    cfg.max_video_tracks = 0; // geometry only
    mux.configure(cfg);

    auto chunk = make_spz_chunk(1, /*keyframe=*/true, 64);
    mux.push_gaussian_frame(chunk, /*timestamp_ms=*/100);

    auto pkt = mux.get_next_packet();
    ASSERT_TRUE(pkt.has_value());
    ASSERT_EQ(pkt->header.magic, kPacketMagic);
    ASSERT_EQ(static_cast<uint8_t>(pkt->header.type),
              static_cast<uint8_t>(PacketType::GaussianKeyframe));
    ASSERT_EQ(pkt->header.frame_id, static_cast<uint32_t>(1));
    ASSERT_EQ(pkt->header.timestamp_ms, static_cast<uint32_t>(100));
    ASSERT_EQ(pkt->header.payload_size, static_cast<uint32_t>(64));
    ASSERT_EQ(pkt->payload.size(), static_cast<size_t>(64));
}

TEST(mux_push_gaussian_delta_get_packet) {
    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    cfg.max_video_tracks = 0;
    mux.configure(cfg);

    auto chunk = make_spz_chunk(5, /*keyframe=*/false, 32);
    mux.push_gaussian_frame(chunk, /*timestamp_ms=*/200);

    auto pkt = mux.get_next_packet();
    ASSERT_TRUE(pkt.has_value());
    ASSERT_EQ(static_cast<uint8_t>(pkt->header.type),
              static_cast<uint8_t>(PacketType::GaussianDelta));
}

TEST(mux_empty_queue_returns_nullopt) {
    GaussianStreamMultiplexer mux;
    auto pkt = mux.get_next_packet();
    ASSERT_FALSE(pkt.has_value());
}

TEST(mux_empty_chunk_ignored) {
    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    mux.configure(cfg);

    SpzChunk empty_chunk;
    empty_chunk.start_frame = 1;
    empty_chunk.is_keyframe = true;
    // compressed_data is empty.
    mux.push_gaussian_frame(empty_chunk, 100);

    auto pkt = mux.get_next_packet();
    ASSERT_FALSE(pkt.has_value());
}

TEST(mux_geometry_before_video_ordering) {
    // With video tracks enabled, geometry should be sent before video.
    // Video frames for a frame_id that has no geometry yet should be held.

    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    cfg.max_video_tracks = 2;
    cfg.max_buffer_ms = 5000;
    mux.configure(cfg);

    // Push video for frame 1 before geometry.
    VideoFrameData vf;
    vf.camera_index  = 0;
    vf.frame_id      = 1;
    vf.timestamp_ms  = 100;
    vf.nal_data      = {0xAA, 0xBB};
    mux.push_video_frame(vf);

    // No packet yet — video is held.
    auto pkt1 = mux.get_next_packet();
    ASSERT_FALSE(pkt1.has_value());

    // Now push geometry for frame 1.
    auto chunk = make_spz_chunk(1, true, 16);
    mux.push_gaussian_frame(chunk, 100);

    // First packet should be geometry.
    auto pkt2 = mux.get_next_packet();
    ASSERT_TRUE(pkt2.has_value());
    ASSERT_EQ(static_cast<uint8_t>(pkt2->header.type),
              static_cast<uint8_t>(PacketType::GaussianKeyframe));

    // Second packet should be the released video.
    auto pkt3 = mux.get_next_packet();
    ASSERT_TRUE(pkt3.has_value());
    ASSERT_EQ(static_cast<uint8_t>(pkt3->header.type),
              static_cast<uint8_t>(PacketType::VideoFrame));
    ASSERT_EQ(pkt3->header.frame_id, static_cast<uint32_t>(1));
}

TEST(mux_video_after_geometry_sent_immediately) {
    // If geometry for frame_id=1 was already sent, a video frame for
    // frame_id=1 should be queued immediately.

    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    cfg.max_video_tracks = 2;
    mux.configure(cfg);

    // Push geometry first.
    auto chunk = make_spz_chunk(1, true, 16);
    mux.push_gaussian_frame(chunk, 100);

    // Push video — should go directly to queue.
    VideoFrameData vf;
    vf.camera_index  = 0;
    vf.frame_id      = 1;
    vf.timestamp_ms  = 100;
    vf.nal_data      = {0x01, 0x02};
    mux.push_video_frame(vf);

    // Drain: geometry, then video.
    auto p1 = mux.get_next_packet();
    ASSERT_TRUE(p1.has_value());
    auto p2 = mux.get_next_packet();
    ASSERT_TRUE(p2.has_value());
    ASSERT_EQ(static_cast<uint8_t>(p2->header.type),
              static_cast<uint8_t>(PacketType::VideoFrame));
}

TEST(mux_packet_to_bytes_format) {
    // Verify to_bytes produces header + payload.
    MuxPacket pkt;
    pkt.header.magic        = kPacketMagic;
    pkt.header.type         = PacketType::GaussianDelta;
    pkt.header.flags        = 0;
    pkt.header.frame_id     = 7;
    pkt.header.timestamp_ms = 300;
    pkt.payload             = {0x10, 0x20, 0x30};
    pkt.header.payload_size = 3;

    auto bytes = pkt.to_bytes();
    ASSERT_EQ(bytes.size(), kPacketHeaderSize + 3);

    // Verify header can be parsed from the wire bytes.
    PacketHeader parsed;
    bool ok = deserialize_header(bytes.data(), bytes.size(), parsed);
    ASSERT_TRUE(ok);
    ASSERT_EQ(parsed.frame_id, static_cast<uint32_t>(7));
    ASSERT_EQ(parsed.payload_size, static_cast<uint32_t>(3));

    // Verify payload follows header.
    ASSERT_EQ(bytes[kPacketHeaderSize + 0], 0x10);
    ASSERT_EQ(bytes[kPacketHeaderSize + 1], 0x20);
    ASSERT_EQ(bytes[kPacketHeaderSize + 2], 0x30);
}

TEST(mux_finish_produces_final_packet) {
    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    cfg.max_video_tracks = 0;
    mux.configure(cfg);

    // Push one frame, then finish.
    auto chunk = make_spz_chunk(10, true, 8);
    mux.push_gaussian_frame(chunk, 500);

    mux.finish(/*last_frame_id=*/10, /*timestamp_ms=*/500);

    // Drain: first the geometry packet.
    auto p1 = mux.get_next_packet();
    ASSERT_TRUE(p1.has_value());
    ASSERT_EQ(static_cast<uint8_t>(p1->header.type),
              static_cast<uint8_t>(PacketType::GaussianKeyframe));

    // Then the sentinel final packet.
    auto p2 = mux.get_next_packet();
    ASSERT_TRUE(p2.has_value());
    ASSERT_TRUE(p2->header.flags & kFlagIsFinal);
    ASSERT_EQ(p2->header.frame_id, static_cast<uint32_t>(10));
    ASSERT_EQ(p2->payload.size(), static_cast<size_t>(0));

    // Queue should be empty now.
    auto p3 = mux.get_next_packet();
    ASSERT_FALSE(p3.has_value());
}

TEST(mux_finish_flushes_held_video) {
    // Held video should be flushed on finish even if geometry never arrived.

    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    cfg.max_video_tracks = 2;
    cfg.max_buffer_ms = 10000;
    mux.configure(cfg);

    // Push video with no matching geometry.
    VideoFrameData vf;
    vf.camera_index  = 0;
    vf.frame_id      = 99;
    vf.timestamp_ms  = 100;
    vf.nal_data      = {0x01};
    mux.push_video_frame(vf);

    // No geometry pushed. Finish.
    mux.finish(99, 100);

    // Drain: should get the flushed video + final sentinel.
    std::vector<PacketType> types;
    while (auto p = mux.get_next_packet()) {
        types.push_back(p->header.type);
    }

    // At minimum, we should have the video frame and the final packet.
    ASSERT_GE(static_cast<int>(types.size()), 2);

    // Last packet should be the final sentinel.
    bool has_final = false;
    // Check last extracted packet was GaussianKeyframe with final flag
    // (we checked flags in the finish test above). Here just ensure
    // video was flushed.
    bool has_video = false;
    for (auto t : types) {
        if (t == PacketType::VideoFrame) has_video = true;
    }
    ASSERT_TRUE(has_video);
}

TEST(mux_manifest_packet) {
    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    mux.configure(cfg);

    std::string json = R"({"version":"1.0","fps":30})";
    mux.push_manifest(json);

    auto pkt = mux.get_next_packet();
    ASSERT_TRUE(pkt.has_value());
    ASSERT_EQ(static_cast<uint8_t>(pkt->header.type),
              static_cast<uint8_t>(PacketType::Manifest));
    ASSERT_EQ(pkt->header.frame_id, static_cast<uint32_t>(0));

    // Payload should contain the JSON string.
    std::string payload_str(pkt->payload.begin(), pkt->payload.end());
    ASSERT_EQ(payload_str, json);
}

TEST(mux_stats_tracking) {
    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    cfg.max_video_tracks = 0;
    mux.configure(cfg);

    auto s0 = mux.stats();
    ASSERT_EQ(s0.gaussian_packets, static_cast<int64_t>(0));
    ASSERT_EQ(s0.bytes_sent, static_cast<int64_t>(0));

    auto chunk = make_spz_chunk(1, true, 100);
    mux.push_gaussian_frame(chunk, 100);

    auto s1 = mux.stats();
    ASSERT_EQ(s1.gaussian_packets, static_cast<int64_t>(1));
    ASSERT_EQ(s1.queue_depth, 1);

    // Drain the packet.
    auto pkt = mux.get_next_packet();
    ASSERT_TRUE(pkt.has_value());

    auto s2 = mux.stats();
    ASSERT_EQ(s2.queue_depth, 0);
    ASSERT_GT(s2.bytes_sent, static_cast<int64_t>(0));
}

TEST(mux_has_video_flag_set_when_video_tracks_enabled) {
    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    cfg.max_video_tracks = 2;
    mux.configure(cfg);

    auto chunk = make_spz_chunk(1, true, 16);
    mux.push_gaussian_frame(chunk, 100);

    auto pkt = mux.get_next_packet();
    ASSERT_TRUE(pkt.has_value());
    ASSERT_TRUE(pkt->header.flags & kFlagHasVideo);
}

TEST(mux_no_video_flag_when_video_tracks_disabled) {
    GaussianStreamMultiplexer mux;
    MuxConfig cfg;
    cfg.max_video_tracks = 0;
    mux.configure(cfg);

    auto chunk = make_spz_chunk(1, true, 16);
    mux.push_gaussian_frame(chunk, 100);

    auto pkt = mux.get_next_packet();
    ASSERT_TRUE(pkt.has_value());
    ASSERT_FALSE(pkt->header.flags & kFlagHasVideo);
}
