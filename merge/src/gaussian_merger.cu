#include "gaussian_merger.h"
#include "../../gaussian/src/spatial_hash.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cassert>
#include <vector>

namespace heimdall::merge {

using gaussian::Gaussian;
using gaussian::SpatialHashGrid;
using gaussian::kHashTableSize;
using gaussian::kCellEmpty;
using gaussian::hash_cell;
using gaussian::position_to_cell;
using gaussian::hash_gaussians_kernel;
using gaussian::build_cell_table_kernel;
using gaussian::clear_cell_table_kernel;

static constexpr int kBlockSize = 256;

// Kernel: Spatial dedup + weighted merge.
// Probes 3x3x3 neighborhood. Gaussians from DIFFERENT workers within
// dedup_distance are merged: lower-opacity removed, survivor accumulates.
// Tie-breaking: higher opacity wins, then lowest index.
__global__ void dedup_merge_kernel(
    Gaussian* __restrict__ gs, const int* __restrict__ wids, int n,
    const SpatialHashGrid grid, float dedup_dist,
    int* __restrict__ rm, float* __restrict__ mw, int wmode
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* pi = gs[i].position;
    int wi = wids[i]; float oi = gs[i].opacity;
    int cx, cy, cz;
    position_to_cell(pi, grid.cell_size, cx, cy, cz);
    float tsq = dedup_dist * dedup_dist;
    int best = i; float bop = oi;

    for (int dz = -1; dz <= 1; dz++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            uint32_t h = hash_cell(cx+dx, cy+dy, cz+dz);
            int s = grid.cell_start[h];
            if (s == kCellEmpty) continue;
            int e = grid.cell_end[h];
            for (int k = s; k < e; k++) {
                int j = grid.sorted_indices[k];
                if (j == i || wids[j] == wi) continue;
                float px = gs[j].position[0]-pi[0];
                float py = gs[j].position[1]-pi[1];
                float pz = gs[j].position[2]-pi[2];
                if (px*px+py*py+pz*pz < tsq) {
                    float oj = gs[j].opacity;
                    if (oj > bop || (oj == bop && j < best))
                        { best = j; bop = oj; }
                }
            }
        }

    float w_i = (wmode == 1) ? oi : 1.0f;
    if (best != i) {
        rm[i] = 1;
        atomicAdd(&mw[best], w_i);
        for (int c = 0; c < 3; c++) {
            atomicAdd(&gs[best].position[c], w_i * pi[c]);
            atomicAdd(&gs[best].scale[c], w_i * gs[i].scale[c]);
            atomicAdd(&gs[best].sh[c], w_i * gs[i].sh[c]);
        }
        atomicAdd(&gs[best].opacity, w_i * oi);
    } else {
        rm[i] = 0; mw[i] = w_i;
        for (int c = 0; c < 3; c++) {
            gs[i].position[c] *= w_i;
            gs[i].scale[c] *= w_i;
            gs[i].sh[c] *= w_i;
        }
        gs[i].opacity *= w_i;
    }
}

// Normalize survivors by accumulated weight.
__global__ void normalize_survivors_kernel(
    Gaussian* __restrict__ gs, const int* __restrict__ rm,
    const float* __restrict__ mw, int n, float min_op
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || rm[i]) return;
    float w = mw[i]; if (w <= 0.0f) return;
    float inv = 1.0f / w;
    for (int c=0;c<3;c++) { gs[i].position[c]*=inv; gs[i].scale[c]*=inv; gs[i].sh[c]*=inv; }
    gs[i].opacity *= inv;
    if (gs[i].opacity < min_op) gs[i].opacity = 0.0f;
}

// Build keep flags for stream compaction.
__global__ void build_keep_flags_kernel(
    const int* __restrict__ rm, const Gaussian* __restrict__ gs,
    int* __restrict__ kf, int n, float min_op
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    kf[i] = (!rm[i] && gs[i].opacity >= min_op) ? 1 : 0;
}

// Scatter surviving Gaussians into compacted output.
__global__ void scatter_compact_kernel(
    const Gaussian* __restrict__ in, Gaussian* __restrict__ out,
    const int* __restrict__ kf, const int* __restrict__ si, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (kf[i]) out[si[i]] = in[i];
}

// Host implementation
static void ensure_cap(void** p, int* cap, int need, size_t es) {
    if (need <= *cap && *p) return;
    if (*p) cudaFree(*p);
    *cap = std::max(need, *cap * 2);
    cudaMalloc(p, (size_t)*cap * es);
}

GaussianMerger::GaussianMerger(const GaussianMergerConfig& cfg)
    : config_(cfg) { cudaStreamCreate(&stream_); }

GaussianMerger::~GaussianMerger() {
    reset();
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

void GaussianMerger::reset() {
    auto sf = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    sf(reinterpret_cast<void*&>(d_input_));
    sf(reinterpret_cast<void*&>(d_worker_ids_));
    sf(reinterpret_cast<void*&>(d_remove_flags_));
    sf(reinterpret_cast<void*&>(d_merge_weights_));
    sf(reinterpret_cast<void*&>(d_output_));
    sf(reinterpret_cast<void*&>(d_compact_indices_));
    sf(reinterpret_cast<void*&>(d_compact_count_));
    hash_storage_.free();
    input_capacity_ = output_capacity_ = input_count_ = output_count_ = 0;
}

void GaussianMerger::upload_partials(const AggregatedFrame& agg) {
    int total = 0;
    for (const auto& p : agg.partials) total += p.gaussians.num_gaussians;
    input_count_ = total; if (!total) return;
    ensure_cap((void**)&d_input_, &input_capacity_, total, sizeof(Gaussian));
    ensure_cap((void**)&d_worker_ids_, &input_capacity_, total, sizeof(int));
    ensure_cap((void**)&d_remove_flags_, &input_capacity_, total, sizeof(int));
    ensure_cap((void**)&d_merge_weights_, &input_capacity_, total, sizeof(float));
    ensure_cap((void**)&d_compact_indices_, &input_capacity_, total, sizeof(int));
    if (!d_compact_count_) cudaMalloc(&d_compact_count_, sizeof(int));
    int off = 0;
    for (const auto& p : agg.partials) {
        int n = p.gaussians.num_gaussians; if (!n) continue;
        cudaMemcpyAsync(d_input_+off, p.gaussians.gaussians.data(),
                        n*sizeof(Gaussian), cudaMemcpyHostToDevice, stream_);
        std::vector<int> w(n, p.worker_id);
        cudaMemcpyAsync(d_worker_ids_+off, w.data(), n*sizeof(int), cudaMemcpyHostToDevice, stream_);
        off += n;
    }
    cudaMemsetAsync(d_remove_flags_, 0, total*sizeof(int), stream_);
    cudaMemsetAsync(d_merge_weights_, 0, total*sizeof(float), stream_);
}

void GaussianMerger::build_hash_grid() {
    if (!input_count_) return;
    hash_storage_.allocate(input_count_);
    int gs = (input_count_+kBlockSize-1)/kBlockSize;
    int tgs = (kHashTableSize+kBlockSize-1)/kBlockSize;
    clear_cell_table_kernel<<<tgs,kBlockSize,0,stream_>>>(hash_storage_.cell_start, hash_storage_.cell_end, kHashTableSize);
    hash_gaussians_kernel<<<gs,kBlockSize,0,stream_>>>(d_input_, hash_storage_.cell_hashes, hash_storage_.sorted_indices, input_count_, config_.dedup_distance);
    thrust::device_ptr<uint32_t> keys(hash_storage_.cell_hashes);
    thrust::device_ptr<int> vals(hash_storage_.sorted_indices);
    thrust::sort_by_key(keys, keys+input_count_, vals);
    build_cell_table_kernel<<<gs,kBlockSize,0,stream_>>>(hash_storage_.cell_hashes, hash_storage_.cell_start, hash_storage_.cell_end, input_count_);
}

void GaussianMerger::dedup_and_merge() {
    if (!input_count_) return;
    SpatialHashGrid grid = hash_storage_.make_grid(config_.dedup_distance, input_count_);
    int gs = (input_count_+kBlockSize-1)/kBlockSize;
    dedup_merge_kernel<<<gs,kBlockSize,0,stream_>>>(d_input_, d_worker_ids_, input_count_, grid, config_.dedup_distance, d_remove_flags_, d_merge_weights_, config_.weight_mode);
    normalize_survivors_kernel<<<gs,kBlockSize,0,stream_>>>(d_input_, d_remove_flags_, d_merge_weights_, input_count_, config_.min_opacity);
}

void GaussianMerger::compact_output() {
    if (!input_count_) { output_count_=0; return; }
    int gs = (input_count_+kBlockSize-1)/kBlockSize;
    int* d_kf = d_compact_indices_; int* d_si = d_remove_flags_;
    build_keep_flags_kernel<<<gs,kBlockSize,0,stream_>>>(d_remove_flags_, d_input_, d_kf, input_count_, config_.min_opacity);
    cudaStreamSynchronize(stream_);
    thrust::device_ptr<int> kp(d_kf); thrust::device_ptr<int> sp(d_si);
    thrust::exclusive_scan(kp, kp+input_count_, sp);
    int ls=0, lk=0;
    cudaMemcpy(&ls, d_si+input_count_-1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lk, d_kf+input_count_-1, sizeof(int), cudaMemcpyDeviceToHost);
    output_count_ = std::min(ls+lk, config_.max_output_gaussians);
    ensure_cap((void**)&d_output_, &output_capacity_, output_count_, sizeof(Gaussian));
    scatter_compact_kernel<<<gs,kBlockSize,0,stream_>>>(d_input_, d_output_, d_kf, d_si, input_count_);
    cudaStreamSynchronize(stream_);
}

MergedFrame GaussianMerger::merge(const AggregatedFrame& agg) {
    upload_partials(agg); build_hash_grid(); dedup_and_merge(); compact_output();
    stat_last_input_=input_count_; stat_last_output_=output_count_;
    stat_last_dupes_=input_count_-output_count_;
    MergedFrame r; r.frame_id=agg.frame_id; r.timestamp_ns=agg.timestamp_ns;
    r.num_gaussians=output_count_; r.d_gaussians=d_output_;
    r.total_input_gaussians=input_count_; r.duplicates_removed=stat_last_dupes_;
    return r;
}

int GaussianMerger::last_input_count() const { return stat_last_input_; }
int GaussianMerger::last_output_count() const { return stat_last_output_; }
int GaussianMerger::last_duplicates_removed() const { return stat_last_dupes_; }

} // namespace heimdall::merge
