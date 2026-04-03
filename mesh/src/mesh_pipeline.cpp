#include "mesh_pipeline.h"

#include <chrono>
#include <stdexcept>

namespace heimdall::mesh {

MeshPipeline::MeshPipeline(const MeshPipelineConfig& config)
    : config_(config) {}

MeshPipeline::Output MeshPipeline::process(const TriMesh& input) const {
    using Clock = std::chrono::steady_clock;

    Output out;
    MeshPipelineMetrics& m = out.metrics;
    auto pipeline_start = Clock::now();

    // =================================================================
    // 1. Validate input.
    // =================================================================

    if (input.empty()) {
        throw std::invalid_argument("MeshPipeline::process: input mesh is empty");
    }

    if (!input.is_valid()) {
        throw std::invalid_argument(
            "MeshPipeline::process: input mesh is invalid "
            "(index out of range or inconsistent attribute sizes)");
    }

    m.input_vertices = input.vertex_count();
    m.input_faces    = input.face_count();

    // =================================================================
    // 2. Pre-process: remove degenerates, compute normals.
    // =================================================================

    auto preprocess_start = Clock::now();

    // Work on a mutable copy for pre-processing.
    TriMesh working = input;

    if (config_.remove_degenerates) {
        m.degenerate_faces_removed = working.remove_degenerate_faces();
    }

    bool needs_normals =
        (config_.compute_normals_if_missing && !working.has_normals()) ||
        config_.force_recompute_normals;

    if (needs_normals) {
        working.recompute_normals();
    }

    auto preprocess_end = Clock::now();
    m.preprocess_ms = std::chrono::duration<double, std::milli>(
        preprocess_end - preprocess_start).count();

    // =================================================================
    // 3. Simplify.
    // =================================================================

    auto simplify_start = Clock::now();

    if (!config_.skip_simplification) {
        MeshSimplifier simplifier(config_.simplifier);
        auto simp_out = simplifier.simplify(working);
        working      = std::move(simp_out.mesh);
        m.simplifier = simp_out.metrics;
    } else {
        m.simplifier.input_faces     = working.face_count();
        m.simplifier.input_vertices  = working.vertex_count();
        m.simplifier.output_faces    = working.face_count();
        m.simplifier.output_vertices = working.vertex_count();
    }

    auto simplify_end = Clock::now();
    m.simplify_ms = std::chrono::duration<double, std::milli>(
        simplify_end - simplify_start).count();

    // =================================================================
    // 4. Recompute normals after simplification.
    //
    // Simplification can cause the original normals to be slightly off
    // because vertices have moved.  Recomputing from the simplified
    // geometry gives cleaner shading.
    // =================================================================

    if (!config_.skip_simplification && config_.recompute_normals_after_simplify) {
        working.recompute_normals();
    }

    // =================================================================
    // 5. UV unwrap.
    // =================================================================

    auto unwrap_start = Clock::now();

    if (!config_.skip_unwrap) {
        MeshUnwrapper unwrapper(config_.unwrapper);
        auto unwrap_out = unwrapper.unwrap(working);
        working        = std::move(unwrap_out.mesh);
        m.unwrapper    = unwrap_out.metrics;
    } else {
        m.unwrapper.input_vertices  = working.vertex_count();
        m.unwrapper.input_faces     = working.face_count();
        m.unwrapper.output_vertices = working.vertex_count();
        m.unwrapper.output_faces    = working.face_count();
    }

    auto unwrap_end = Clock::now();
    m.unwrap_ms = std::chrono::duration<double, std::milli>(
        unwrap_end - unwrap_start).count();

    // =================================================================
    // 6. Final output.
    // =================================================================

    out.mesh = std::move(working);
    m.output_vertices = out.mesh.vertex_count();
    m.output_faces    = out.mesh.face_count();

    auto pipeline_end = Clock::now();
    m.total_ms = std::chrono::duration<double, std::milli>(
        pipeline_end - pipeline_start).count();

    return out;
}

} // namespace heimdall::mesh
