#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace heimdall::transport {

// Routes camera streams to distributed workers.
//
// The capture machine runs a single CameraRouter that decides
// which cameras go to which RunPod worker, based on stereo pair
// assignments and worker availability.
//
// Strategies:
//   GEOGRAPHIC: cameras 0-4 → worker 0, 5-9 → worker 1, etc.
//   INTERLEAVED_PAIRS: stereo pairs spread across workers for coverage
//   REDUNDANT: all workers get all cameras, each runs different pairs
//
// The router also handles dynamic re-assignment when workers go
// up/down (via health checks).

enum class RoutingStrategy {
    GEOGRAPHIC,         // spatial split, minimal bandwidth
    INTERLEAVED_PAIRS,  // best coverage per worker
    REDUNDANT           // all cameras to all workers, pairs differ
};

struct WorkerEndpoint {
    std::string worker_id;
    std::string host;       // Tailscale IP or hostname
    int port;
    bool is_healthy;
    double last_heartbeat_ms;
};

struct StereoPair {
    int camera_a;
    int camera_b;
    float baseline_m;       // distance between cameras
    float angular_spread;   // angle between camera directions (radians)
};

struct CameraAssignment {
    std::string worker_id;
    std::vector<int> camera_indices;        // which cameras this worker receives
    std::vector<StereoPair> stereo_pairs;   // which pairs this worker processes
};

struct RouterConfig {
    RoutingStrategy strategy = RoutingStrategy::INTERLEAVED_PAIRS;
    int num_cameras = 20;
    int pairs_per_worker = 3;       // stereo pairs per worker
    float min_baseline_m = 0.3f;    // minimum stereo baseline
    float max_baseline_m = 2.0f;    // maximum stereo baseline
    int health_check_interval_ms = 1000;
    int worker_timeout_ms = 5000;   // mark unhealthy after no heartbeat
};

class CameraRouter {
public:
    explicit CameraRouter(const RouterConfig& config);

    // Register a worker endpoint.
    void add_worker(const WorkerEndpoint& endpoint);
    void remove_worker(const std::string& worker_id);

    // Update worker health (called from heartbeat thread).
    void heartbeat(const std::string& worker_id);

    // Compute optimal camera-to-worker assignments.
    // Call this whenever workers change or periodically to rebalance.
    std::vector<CameraAssignment> compute_assignments(
        const std::vector<StereoPair>& all_pairs
    ) const;

    // Generate all valid stereo pairs from camera positions.
    // Filters by baseline range and returns sorted by angular spread.
    static std::vector<StereoPair> generate_stereo_pairs(
        const float (*camera_positions)[3],
        int num_cameras,
        float min_baseline_m,
        float max_baseline_m
    );

    // Get current healthy worker count.
    int healthy_worker_count() const;

    // Get all current assignments (cached from last compute).
    const std::vector<CameraAssignment>& current_assignments() const;

private:
    RouterConfig config_;
    std::map<std::string, WorkerEndpoint> workers_;
    std::vector<CameraAssignment> cached_assignments_;
};

} // namespace heimdall::transport
