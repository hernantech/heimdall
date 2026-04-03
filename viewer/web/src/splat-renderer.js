// splat-renderer.js -- WebGPU Gaussian splat renderer for heimdall
//
// Renders 3D Gaussian splats by:
//   1. Uploading per-Gaussian attributes to GPU storage buffers
//   2. Running a compute shader to sort Gaussians by depth (back-to-front)
//   3. Running a render pass that projects each Gaussian to a 2D screen-space
//      ellipse and alpha-blends them in sorted order
//
// This is a simplified but functional renderer suitable for the heimdall
// demo viewer. It covers the core algorithm: project 3D covariance to 2D,
// evaluate the Gaussian contribution per pixel, composite with alpha blending.
//
// Gaussian attributes (from pipeline.h):
//   position   float32 x 3      xyz world coords
//   scale      float32 x 3      per-axis scale
//   rotation   float32 x 4      quaternion (wxyz)
//   opacity    float32           alpha [0,1]
//   sh         float32 x 48     SH coefficients (degree 3, 16 coeffs x 3 RGB)

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------

/** Compute shader: evaluate SH band-0 color and project Gaussians to 2D. */
const PREPROCESS_WGSL = /* wgsl */`

struct Gaussian {
    pos: vec3f,
    scale: vec3f,
    rot: vec4f,     // quaternion w,x,y,z
    opacity: f32,
    sh0: vec3f,     // DC component of SH (band 0)
};

struct Splat2D {
    center: vec2f,      // screen-space center (pixels)
    axis_a: vec2f,      // major axis of the 2D ellipse
    axis_b: vec2f,      // minor axis of the 2D ellipse
    color: vec4f,       // RGBA (SH-evaluated color + opacity)
    depth: f32,         // view-space depth for sorting
    _pad: f32,
};

struct Uniforms {
    view: mat4x4f,
    proj: mat4x4f,
    viewport: vec2f,    // canvas width, height in pixels
    num_gaussians: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       gaussians : array<Gaussian>;
@group(0) @binding(1) var<storage, read_write>  splats    : array<Splat2D>;
@group(0) @binding(2) var<uniform>              uniforms  : Uniforms;

// Build a 3x3 rotation matrix from a unit quaternion (w,x,y,z).
fn quat_to_mat3(q: vec4f) -> mat3x3f {
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    return mat3x3f(
        vec3f(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y)),
        vec3f(2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
        vec3f(2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y)),
    );
}

@compute @workgroup_size(256)
fn preprocess(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= uniforms.num_gaussians) { return; }

    let g = gaussians[idx];

    // ---- view-space position ------------------------------------------------
    let world_pos = vec4f(g.pos, 1.0);
    let view_pos = uniforms.view * world_pos;
    let depth = view_pos.z;

    // Cull Gaussians behind the camera.
    if (depth < 0.1) {
        splats[idx].depth = 1e10;
        splats[idx].color = vec4f(0.0);
        return;
    }

    // ---- 3D covariance in world space ---------------------------------------
    let R = quat_to_mat3(g.rot);
    let S = mat3x3f(
        vec3f(g.scale.x, 0.0, 0.0),
        vec3f(0.0, g.scale.y, 0.0),
        vec3f(0.0, 0.0, g.scale.z),
    );
    let M = R * S;
    let Sigma = M * transpose(M);

    // ---- project covariance to 2D (Jacobian of perspective projection) ------
    let fx = uniforms.proj[0][0] * uniforms.viewport.x * 0.5;
    let fy = uniforms.proj[1][1] * uniforms.viewport.y * 0.5;

    let tz = view_pos.z;
    let tz2 = tz * tz;

    // Jacobian of the perspective projection at view_pos.
    let J = mat3x3f(
        vec3f(fx / tz, 0.0,             0.0),
        vec3f(0.0,     fy / tz,         0.0),
        vec3f(-fx * view_pos.x / tz2, -fy * view_pos.y / tz2, 0.0),
    );

    let view3 = mat3x3f(
        uniforms.view[0].xyz,
        uniforms.view[1].xyz,
        uniforms.view[2].xyz,
    );

    let cov3d_view = view3 * Sigma * transpose(view3);
    let cov2d_full = J * cov3d_view * transpose(J);

    // Extract the 2x2 sub-matrix of the projected covariance.
    let a = cov2d_full[0][0] + 0.3;  // small regularizer for stability
    let b = cov2d_full[0][1];
    let c = cov2d_full[1][1] + 0.3;

    // ---- eigendecomposition of 2x2 covariance for ellipse axes --------------
    let disc = sqrt(max((a - c) * (a - c) + 4.0 * b * b, 0.0));
    let lambda1 = 0.5 * (a + c + disc);
    let lambda2 = 0.5 * (a + c - disc);

    if (lambda1 <= 0.0 || lambda2 <= 0.0) {
        splats[idx].depth = 1e10;
        splats[idx].color = vec4f(0.0);
        return;
    }

    let r1 = sqrt(lambda1);
    let r2 = sqrt(lambda2);

    // Eigenvector for lambda1 to get orientation.
    var v1 = vec2f(b, lambda1 - a);
    let v1_len = length(v1);
    if (v1_len > 1e-6) {
        v1 = v1 / v1_len;
    } else {
        v1 = vec2f(1.0, 0.0);
    }
    let v2 = vec2f(-v1.y, v1.x);

    // Splat axes: direction * radius (at ~3-sigma for visible extent).
    let axis_a = v1 * r1 * 3.0;
    let axis_b = v2 * r2 * 3.0;

    // ---- NDC -> screen-space center -----------------------------------------
    let clip = uniforms.proj * view_pos;
    let ndc = clip.xy / clip.w;
    let screen = (ndc * 0.5 + 0.5) * uniforms.viewport;

    // ---- SH band-0 color (DC coefficient) -----------------------------------
    // The SH DC component maps to color via:  C = SH_C0 * sh0 + 0.5
    let SH_C0 = 0.28209479177387814;
    let color = clamp(g.sh0 * SH_C0 + vec3f(0.5), vec3f(0.0), vec3f(1.0));

    // ---- write output -------------------------------------------------------
    splats[idx].center = screen;
    splats[idx].axis_a = axis_a;
    splats[idx].axis_b = axis_b;
    splats[idx].color  = vec4f(color, g.opacity);
    splats[idx].depth  = depth;
}
`;

/** Compute shader: bitonic sort of splat indices by depth (back-to-front). */
const SORT_WGSL = /* wgsl */`

struct Splat2D {
    center: vec2f,
    axis_a: vec2f,
    axis_b: vec2f,
    color: vec4f,
    depth: f32,
    _pad: f32,
};

struct SortUniforms {
    num_elements: u32,
    block_size: u32,
    sub_block_size: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       splats    : array<Splat2D>;
@group(0) @binding(1) var<storage, read_write>  indices   : array<u32>;
@group(0) @binding(2) var<uniform>              sort_uni  : SortUniforms;

@compute @workgroup_size(256)
fn sort_step(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= sort_uni.num_elements) { return; }

    let block = sort_uni.block_size;
    let sub = sort_uni.sub_block_size;

    let grp = i / sub;
    let idx_in_grp = i % sub;
    let pair_base = grp * sub * 2 + idx_in_grp;
    let pair_other = pair_base + sub;

    if (pair_other >= sort_uni.num_elements) { return; }

    let ascending = ((pair_base / block) % 2) == 0;

    let idx_a = indices[pair_base];
    let idx_b = indices[pair_other];
    let depth_a = splats[idx_a].depth;
    let depth_b = splats[idx_b].depth;

    // Back-to-front: we want greater depth first.
    let should_swap = select(depth_a < depth_b, depth_a > depth_b, ascending);

    if (should_swap) {
        indices[pair_base]  = idx_b;
        indices[pair_other] = idx_a;
    }
}
`;

/** Vertex + fragment shaders: render sorted 2D splats as quads. */
const RENDER_WGSL = /* wgsl */`

struct Splat2D {
    center: vec2f,
    axis_a: vec2f,
    axis_b: vec2f,
    color: vec4f,
    depth: f32,
    _pad: f32,
};

struct RenderUniforms {
    viewport: vec2f,
    _pad: vec2f,
};

@group(0) @binding(0) var<storage, read> splats   : array<Splat2D>;
@group(0) @binding(1) var<storage, read> indices  : array<u32>;
@group(0) @binding(2) var<uniform>       render_u : RenderUniforms;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
    @location(1) offset: vec2f,   // local quad coords [-1,1]
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOutput {
    // Each instance is a quad (2 triangles, 6 vertices via triangle-list).
    // Quad corners in local space: (-1,-1), (1,-1), (1,1), (-1,-1), (1,1), (-1,1)
    var corners = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0,  1.0),
        vec2f(-1.0, -1.0), vec2f(1.0,  1.0), vec2f(-1.0,  1.0),
    );
    let corner = corners[vid];

    let splat_idx = indices[iid];
    let s = splats[splat_idx];

    // Position the quad in screen-space using the ellipse axes.
    let screen_pos = s.center + s.axis_a * corner.x + s.axis_b * corner.y;

    // Convert screen-space pixels to clip-space [-1, 1].
    let ndc = (screen_pos / render_u.viewport) * 2.0 - 1.0;

    var out: VertexOutput;
    out.pos = vec4f(ndc.x, -ndc.y, 0.0, 1.0);  // flip Y for WebGPU convention
    out.color = s.color;
    out.offset = corner;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // Evaluate 2D Gaussian: exp(-0.5 * dot(offset, offset))
    // The offset is in the eigenbasis of the 2D covariance; at the quad edge
    // (offset length = 1) we are at the 3-sigma boundary, so we need to
    // scale: the quad spans [-1,1] which maps to [-3sigma, 3sigma].
    let d = dot(in.offset, in.offset);
    let alpha = exp(-0.5 * d * 9.0) * in.color.a;  // 9.0 = 3^2 (3-sigma mapping)

    if (alpha < 1.0 / 255.0) { discard; }

    return vec4f(in.color.rgb * alpha, alpha);
}
`;

// ---------------------------------------------------------------------------
// SplatRenderer class
// ---------------------------------------------------------------------------

/**
 * SplatRenderer manages the WebGPU resources for rendering Gaussian splats.
 *
 * Usage:
 *   const renderer = new SplatRenderer();
 *   await renderer.init(canvas);
 *   // Each frame:
 *   renderer.updateGaussians(gaussianDataFloat32Array, count);
 *   renderer.render(viewMatrix, projMatrix);
 */
export class SplatRenderer {

    /** @type {GPUDevice | null} */
    #device = null;

    /** @type {GPUCanvasContext | null} */
    #context = null;

    /** @type {GPUTextureFormat} */
    #format = 'bgra8unorm';

    /** @type {HTMLCanvasElement | null} */
    #canvas = null;

    // Pipeline objects.
    #preprocessPipeline = null;
    #sortPipeline = null;
    #renderPipeline = null;

    // Buffers.
    #gaussianBuffer = null;
    #splatBuffer = null;
    #indexBuffer = null;
    #uniformBuffer = null;
    #sortUniformBuffer = null;
    #renderUniformBuffer = null;

    // Bind groups (recreated when buffers change).
    #preprocessBG = null;
    #sortBG = null;
    #renderBG = null;

    // Bind group layouts.
    #preprocessBGL = null;
    #sortBGL = null;
    #renderBGL = null;

    // Current Gaussian count.
    #numGaussians = 0;

    // Maximum Gaussians allocated (buffers are oversized and reused).
    #maxGaussians = 0;

    /** Whether the renderer has been successfully initialized. */
    get ready() { return this.#device !== null; }

    /** Current Gaussian count. */
    get gaussianCount() { return this.#numGaussians; }

    // -- initialization -------------------------------------------------------

    /**
     * Initialize the WebGPU device and pipelines.
     *
     * @param {HTMLCanvasElement} canvas
     * @returns {Promise<boolean>} true if WebGPU is available and init succeeded.
     */
    async init(canvas) {
        if (!navigator.gpu) return false;

        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) return false;

        this.#device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            },
        });
        this.#canvas = canvas;
        this.#context = canvas.getContext('webgpu');
        this.#format = navigator.gpu.getPreferredCanvasFormat();
        this.#context.configure({
            device: this.#device,
            format: this.#format,
            alphaMode: 'premultiplied',
        });

        this.#createPipelines();
        return true;
    }

    /**
     * Release all GPU resources.
     */
    destroy() {
        this.#gaussianBuffer?.destroy();
        this.#splatBuffer?.destroy();
        this.#indexBuffer?.destroy();
        this.#uniformBuffer?.destroy();
        this.#sortUniformBuffer?.destroy();
        this.#renderUniformBuffer?.destroy();
        this.#device?.destroy();
        this.#device = null;
    }

    // -- per-frame data upload ------------------------------------------------

    /**
     * Upload new Gaussian data for rendering.
     *
     * The input is a flat Float32Array laid out per-Gaussian as:
     *   [px, py, pz, sx, sy, sz, rw, rx, ry, rz, opacity, sh0_r, sh0_g, sh0_b, ...]
     *
     * This simplified renderer only reads the first 14 floats per Gaussian
     * (position, scale, rotation, opacity, SH DC component).
     *
     * @param {Float32Array} data   Packed Gaussian attributes.
     * @param {number}       count  Number of Gaussians.
     */
    updateGaussians(data, count) {
        if (!this.#device) return;

        // Ensure GPU buffers are large enough.
        if (count > this.#maxGaussians) {
            this.#allocateBuffers(count);
        }

        this.#numGaussians = count;

        // Repack into the GPU struct layout. Each Gaussian GPU struct:
        //   pos:     vec3f  (12 bytes) + 4-byte pad = 16 bytes
        //   scale:   vec3f  (12 bytes) + 4-byte pad = 16 bytes
        //   rot:     vec4f  (16 bytes)
        //   opacity: f32    (4 bytes) + 4 pad = 8 bytes (to keep sh0 at 16-byte alignment)
        //   sh0:     vec3f  (12 bytes) + 4-byte pad = 16 bytes
        //   Total: 72 bytes (18 floats, padded)
        //
        // To keep things simple we define the GPU struct as:
        //   pos(3) pad(1) scale(3) pad(1) rot(4) opacity(1) pad(1) sh0_r sh0_g sh0_b pad(1)
        //   = 18 floats = 72 bytes
        const FLOATS_PER_GPU_GAUSSIAN = 18;
        const gpuData = new Float32Array(count * FLOATS_PER_GPU_GAUSSIAN);

        // Input stride -- we read 14 floats but the input buffer may have more
        // (full SH coefficients: 3+3+4+1+48 = 59 floats per Gaussian).
        // Detect stride from the total size.
        const inputStride = Math.floor(data.length / count);

        for (let i = 0; i < count; i++) {
            const src = i * inputStride;
            const dst = i * FLOATS_PER_GPU_GAUSSIAN;

            // position xyz
            gpuData[dst + 0] = data[src + 0];
            gpuData[dst + 1] = data[src + 1];
            gpuData[dst + 2] = data[src + 2];
            gpuData[dst + 3] = 0; // pad

            // scale xyz
            gpuData[dst + 4] = data[src + 3];
            gpuData[dst + 5] = data[src + 4];
            gpuData[dst + 6] = data[src + 5];
            gpuData[dst + 7] = 0; // pad

            // rotation wxyz
            gpuData[dst + 8]  = data[src + 6];
            gpuData[dst + 9]  = data[src + 7];
            gpuData[dst + 10] = data[src + 8];
            gpuData[dst + 11] = data[src + 9];

            // opacity
            gpuData[dst + 12] = data[src + 10];
            gpuData[dst + 13] = 0; // pad

            // SH DC component (band-0, first 3 of the 48 SH floats)
            gpuData[dst + 14] = data[src + 11];
            gpuData[dst + 15] = data[src + 12];
            gpuData[dst + 16] = data[src + 13];
            gpuData[dst + 17] = 0; // pad
        }

        this.#device.queue.writeBuffer(this.#gaussianBuffer, 0, gpuData);

        // Initialize index buffer to identity (will be sorted in render()).
        const indices = new Uint32Array(count);
        for (let i = 0; i < count; i++) indices[i] = i;
        this.#device.queue.writeBuffer(this.#indexBuffer, 0, indices);
    }

    // -- render ---------------------------------------------------------------

    /**
     * Render one frame of Gaussian splats.
     *
     * @param {Float32Array} viewMatrix  4x4 column-major view matrix.
     * @param {Float32Array} projMatrix  4x4 column-major projection matrix.
     */
    render(viewMatrix, projMatrix) {
        if (!this.#device || this.#numGaussians === 0) return;

        const width = this.#canvas.width;
        const height = this.#canvas.height;

        // Upload uniforms.
        const uniforms = new Float32Array(16 + 16 + 4); // view(16) + proj(16) + viewport(2) + count(1) + pad(1)
        uniforms.set(viewMatrix, 0);
        uniforms.set(projMatrix, 16);
        uniforms[32] = width;
        uniforms[33] = height;
        // num_gaussians as uint32 in float slot
        const uniformsU32 = new Uint32Array(uniforms.buffer);
        uniformsU32[34] = this.#numGaussians;
        uniformsU32[35] = 0; // pad
        this.#device.queue.writeBuffer(this.#uniformBuffer, 0, uniforms);

        // Render uniforms.
        const renderUniforms = new Float32Array([width, height, 0, 0]);
        this.#device.queue.writeBuffer(this.#renderUniformBuffer, 0, renderUniforms);

        const encoder = this.#device.createCommandEncoder();

        // -- Pass 1: Preprocess (project 3D Gaussians to 2D splats) -----------
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.#preprocessPipeline);
            pass.setBindGroup(0, this.#preprocessBG);
            pass.dispatchWorkgroups(Math.ceil(this.#numGaussians / 256));
            pass.end();
        }

        // -- Pass 2: Sort (bitonic sort by depth, back-to-front) --------------
        this.#encodeSortPasses(encoder);

        // -- Pass 3: Render sorted splats as alpha-blended quads --------------
        {
            const textureView = this.#context.getCurrentTexture().createView();
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(this.#renderPipeline);
            pass.setBindGroup(0, this.#renderBG);
            pass.draw(6, this.#numGaussians); // 6 verts per quad, instanced
            pass.end();
        }

        this.#device.queue.submit([encoder.finish()]);
    }

    // -- private: pipeline creation -------------------------------------------

    #createPipelines() {
        const device = this.#device;

        // -- Preprocess pipeline (compute) ------------------------------------
        this.#preprocessBGL = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        const preprocessModule = device.createShaderModule({ code: PREPROCESS_WGSL });
        this.#preprocessPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.#preprocessBGL] }),
            compute: { module: preprocessModule, entryPoint: 'preprocess' },
        });

        // -- Sort pipeline (compute) ------------------------------------------
        this.#sortBGL = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        const sortModule = device.createShaderModule({ code: SORT_WGSL });
        this.#sortPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.#sortBGL] }),
            compute: { module: sortModule, entryPoint: 'sort_step' },
        });

        // -- Render pipeline (vertex + fragment) ------------------------------
        this.#renderBGL = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX,
                    buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' } },
            ],
        });

        const renderModule = device.createShaderModule({ code: RENDER_WGSL });
        this.#renderPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.#renderBGL] }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.#format,
                    blend: {
                        color: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                    },
                }],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    // -- private: buffer allocation --------------------------------------------

    /**
     * (Re)allocate GPU buffers to hold at least `count` Gaussians.
     * Over-allocates by 25% to reduce reallocations on growing scenes.
     */
    #allocateBuffers(count) {
        const device = this.#device;
        const target = Math.ceil(count * 1.25);

        // Destroy previous buffers.
        this.#gaussianBuffer?.destroy();
        this.#splatBuffer?.destroy();
        this.#indexBuffer?.destroy();
        this.#uniformBuffer?.destroy();
        this.#sortUniformBuffer?.destroy();
        this.#renderUniformBuffer?.destroy();

        // Gaussian input buffer: 18 floats * 4 bytes = 72 bytes per Gaussian.
        const gaussianByteSize = target * 18 * 4;
        this.#gaussianBuffer = device.createBuffer({
            size: gaussianByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Splat2D output buffer: center(2) + axis_a(2) + axis_b(2) + color(4) + depth(1) + pad(1) = 12 floats = 48 bytes.
        const splatByteSize = target * 12 * 4;
        this.#splatBuffer = device.createBuffer({
            size: splatByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Index buffer: one u32 per Gaussian.
        this.#indexBuffer = device.createBuffer({
            size: target * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Uniforms for the preprocess compute shader.
        // view(64) + proj(64) + viewport(8) + count(4) + pad(4) = 144 bytes
        this.#uniformBuffer = device.createBuffer({
            size: 144,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Sort uniforms: num_elements(4) + block_size(4) + sub_block_size(4) + pad(4) = 16 bytes.
        this.#sortUniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Render uniforms: viewport(8) + pad(8) = 16 bytes.
        this.#renderUniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.#maxGaussians = target;

        // Recreate bind groups with the new buffers.
        this.#rebuildBindGroups();
    }

    #rebuildBindGroups() {
        const device = this.#device;

        this.#preprocessBG = device.createBindGroup({
            layout: this.#preprocessBGL,
            entries: [
                { binding: 0, resource: { buffer: this.#gaussianBuffer } },
                { binding: 1, resource: { buffer: this.#splatBuffer } },
                { binding: 2, resource: { buffer: this.#uniformBuffer } },
            ],
        });

        this.#sortBG = device.createBindGroup({
            layout: this.#sortBGL,
            entries: [
                { binding: 0, resource: { buffer: this.#splatBuffer } },
                { binding: 1, resource: { buffer: this.#indexBuffer } },
                { binding: 2, resource: { buffer: this.#sortUniformBuffer } },
            ],
        });

        this.#renderBG = device.createBindGroup({
            layout: this.#renderBGL,
            entries: [
                { binding: 0, resource: { buffer: this.#splatBuffer } },
                { binding: 1, resource: { buffer: this.#indexBuffer } },
                { binding: 2, resource: { buffer: this.#renderUniformBuffer } },
            ],
        });
    }

    // -- private: sorting -----------------------------------------------------

    /**
     * Encode bitonic sort passes into the command encoder.
     * Bitonic sort runs in O(log^2 N) dispatches, each dispatch is O(N) work.
     */
    #encodeSortPasses(encoder) {
        const n = this.#numGaussians;
        if (n <= 1) return;

        // Round up to next power of two for bitonic sort.
        const padN = 1 << Math.ceil(Math.log2(n));

        for (let blockSize = 2; blockSize <= padN; blockSize *= 2) {
            for (let subBlockSize = blockSize / 2; subBlockSize >= 1; subBlockSize /= 2) {
                const sortUniforms = new Uint32Array([n, blockSize, subBlockSize, 0]);
                this.#device.queue.writeBuffer(this.#sortUniformBuffer, 0, sortUniforms);

                const pass = encoder.beginComputePass();
                pass.setPipeline(this.#sortPipeline);
                pass.setBindGroup(0, this.#sortBG);
                pass.dispatchWorkgroups(Math.ceil(n / 256));
                pass.end();
            }
        }
    }
}

/**
 * Check whether WebGPU is available in this browser.
 * @returns {Promise<boolean>}
 */
export async function isWebGPUAvailable() {
    if (!navigator.gpu) return false;
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter !== null;
    } catch {
        return false;
    }
}
