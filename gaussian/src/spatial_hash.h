#pragma once

// GPU spatial hash grid for O(1) nearest-neighbor queries on Gaussian positions.
//
// The grid divides world space into uniform cells whose side length equals the
// match distance threshold. Because a match can only exist within that radius,
// we only need to check the 27 neighboring cells (3x3x3) around the query point.
//
// Usage pattern (all device-side, per frame):
//   1. Build: hash each persistent Gaussian into a cell, store (cell_hash, index).
//   2. Sort by cell_hash (thrust::sort_by_key).
//   3. Build cell_start/cell_end tables via a scatter kernel.
//   4. Query: for each new Gaussian, hash its position, probe 27 neighbor cells,
//      compute distance only against the Gaussians in those cells.
//
// Memory: O(N) for entries + O(TABLE_SIZE) for the cell table.

#include "pipeline.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace heimdall::gaussian {

// Fixed hash table size — must be power of two for fast modulo.
// 1M slots is generous for 500K Gaussians; load factor ~ 0.5.
static constexpr int kHashTableSize = 1 << 20; // 1,048,576

// Sentinel value for empty cell table entries.
static constexpr int kCellEmpty = -1;

// Device-side hash grid structure passed to kernels by value.
struct SpatialHashGrid {
    int* cell_start;       // [kHashTableSize] — first index into sorted_indices for each cell
    int* cell_end;         // [kHashTableSize] — one-past-last index for each cell
    uint32_t* cell_hashes; // [num_entries] — hash per Gaussian (sorted)
    int* sorted_indices;   // [num_entries] — original Gaussian index (sorted alongside hashes)
    float cell_size;       // = match_distance_threshold
    int num_entries;
};

// --- Device helper functions ---

// Spatial hash: map a grid-space integer coordinate to a table slot.
// Uses a simple multiplicative hash (FNV-style primes).
__device__ __forceinline__
uint32_t hash_cell(int cx, int cy, int cz) {
    // Large primes, same trick as many GPU hash implementations.
    uint32_t h = static_cast<uint32_t>(cx) * 73856093u
               ^ static_cast<uint32_t>(cy) * 19349663u
               ^ static_cast<uint32_t>(cz) * 83492791u;
    return h & (kHashTableSize - 1); // power-of-two modulo
}

// Convert a world-space position to integer grid coordinates.
__device__ __forceinline__
void position_to_cell(const float pos[3], float cell_size, int& cx, int& cy, int& cz) {
    // Use floor to handle negative coordinates correctly.
    cx = static_cast<int>(floorf(pos[0] / cell_size));
    cy = static_cast<int>(floorf(pos[1] / cell_size));
    cz = static_cast<int>(floorf(pos[2] / cell_size));
}

// --- Build kernels ---

// Step 1: Compute cell hash for each persistent Gaussian.
__global__ void hash_gaussians_kernel(
    const Gaussian* __restrict__ gaussians,
    uint32_t* __restrict__ cell_hashes,
    int* __restrict__ indices,
    int num_gaussians,
    float cell_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_gaussians) return;

    int cx, cy, cz;
    position_to_cell(gaussians[i].position, cell_size, cx, cy, cz);
    cell_hashes[i] = hash_cell(cx, cy, cz);
    indices[i] = i;
}

// Step 2 (between kernels): thrust::sort_by_key on (cell_hashes, sorted_indices).

// Step 3: Build cell_start / cell_end from the sorted hash array.
__global__ void build_cell_table_kernel(
    const uint32_t* __restrict__ sorted_hashes,
    int* __restrict__ cell_start,
    int* __restrict__ cell_end,
    int num_entries
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_entries) return;

    uint32_t h = sorted_hashes[i];

    // Start of a new cell run?
    if (i == 0 || sorted_hashes[i - 1] != h) {
        cell_start[h] = i;
    }
    // End of a cell run?
    if (i == num_entries - 1 || sorted_hashes[i + 1] != h) {
        cell_end[h] = i + 1;
    }
}

// Clear cell table to kCellEmpty before building.
__global__ void clear_cell_table_kernel(
    int* __restrict__ cell_start,
    int* __restrict__ cell_end,
    int table_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_size) return;
    cell_start[i] = kCellEmpty;
    cell_end[i] = kCellEmpty;
}

// --- Query kernel ---

// Find nearest persistent Gaussian to each new Gaussian using spatial hash lookup.
// Probes the 3x3x3 neighborhood of cells around the query position.
__global__ void hash_match_kernel(
    const Gaussian* __restrict__ new_gs,
    int num_new,
    const Gaussian* __restrict__ old_gs,
    const SpatialHashGrid grid,
    int* __restrict__ match_indices,
    float match_threshold,
    float color_weight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_new) return;

    const float* pos = new_gs[i].position;
    int cx, cy, cz;
    position_to_cell(pos, grid.cell_size, cx, cy, cz);

    float best_dist = match_threshold;
    int best_idx = -1;

    // Probe 3x3x3 neighborhood.
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint32_t h = hash_cell(cx + dx, cy + dy, cz + dz);
                int start = grid.cell_start[h];
                if (start == kCellEmpty) continue;
                int end = grid.cell_end[h];

                for (int k = start; k < end; k++) {
                    int j = grid.sorted_indices[k];

                    // Quick spatial reject before full distance computation.
                    float px = old_gs[j].position[0] - pos[0];
                    float py = old_gs[j].position[1] - pos[1];
                    float pz = old_gs[j].position[2] - pos[2];
                    float pos_dist = sqrtf(px*px + py*py + pz*pz);
                    if (pos_dist >= best_dist) continue;

                    // Color distance (DC SH terms).
                    float dc0 = fabsf(old_gs[j].sh[0] - new_gs[i].sh[0]);
                    float dc1 = fabsf(old_gs[j].sh[1] - new_gs[i].sh[1]);
                    float dc2 = fabsf(old_gs[j].sh[2] - new_gs[i].sh[2]);
                    float color_dist = (dc0 + dc1 + dc2) / 3.0f;

                    float d = pos_dist + color_weight * color_dist;
                    if (d < best_dist) {
                        best_dist = d;
                        best_idx = j;
                    }
                }
            }
        }
    }

    match_indices[i] = best_idx;
}

// --- Host-side helpers for managing GPU memory ---

struct SpatialHashStorage {
    int* cell_start = nullptr;
    int* cell_end = nullptr;
    uint32_t* cell_hashes = nullptr;
    int* sorted_indices = nullptr;
    int capacity = 0; // max entries allocated

    void allocate(int max_entries) {
        if (max_entries <= capacity && cell_start != nullptr) return;

        free();
        capacity = max_entries;
        cudaMalloc(&cell_start, kHashTableSize * sizeof(int));
        cudaMalloc(&cell_end, kHashTableSize * sizeof(int));
        cudaMalloc(&cell_hashes, capacity * sizeof(uint32_t));
        cudaMalloc(&sorted_indices, capacity * sizeof(int));
    }

    void free() {
        if (cell_start) { cudaFree(cell_start); cell_start = nullptr; }
        if (cell_end) { cudaFree(cell_end); cell_end = nullptr; }
        if (cell_hashes) { cudaFree(cell_hashes); cell_hashes = nullptr; }
        if (sorted_indices) { cudaFree(sorted_indices); sorted_indices = nullptr; }
        capacity = 0;
    }

    SpatialHashGrid make_grid(float cell_size, int num_entries) const {
        return SpatialHashGrid{
            cell_start, cell_end, cell_hashes, sorted_indices, cell_size, num_entries
        };
    }

    ~SpatialHashStorage() { free(); }

    // Non-copyable, movable.
    SpatialHashStorage() = default;
    SpatialHashStorage(const SpatialHashStorage&) = delete;
    SpatialHashStorage& operator=(const SpatialHashStorage&) = delete;
    SpatialHashStorage(SpatialHashStorage&& o) noexcept
        : cell_start(o.cell_start), cell_end(o.cell_end),
          cell_hashes(o.cell_hashes), sorted_indices(o.sorted_indices),
          capacity(o.capacity)
    {
        o.cell_start = o.cell_end = nullptr;
        o.cell_hashes = nullptr;
        o.sorted_indices = nullptr;
        o.capacity = 0;
    }
    SpatialHashStorage& operator=(SpatialHashStorage&& o) noexcept {
        if (this != &o) {
            free();
            cell_start = o.cell_start; cell_end = o.cell_end;
            cell_hashes = o.cell_hashes; sorted_indices = o.sorted_indices;
            capacity = o.capacity;
            o.cell_start = o.cell_end = nullptr;
            o.cell_hashes = nullptr; o.sorted_indices = nullptr;
            o.capacity = 0;
        }
        return *this;
    }
};

} // namespace heimdall::gaussian
