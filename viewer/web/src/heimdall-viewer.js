// heimdall-viewer.js -- Main viewer class for the heimdall volumetric capture pipeline
//
// HeimdallViewer is the top-level entry point. It:
//   - Detects WebGPU capability and selects the appropriate renderer
//     (Gaussian splat via WebGPU, or mesh + VDTM via WebGL fallback)
//   - Connects to live streams (WebRTC) or VOD manifests (HTTP)
//   - Manages a small frame buffer for smooth playback
//   - Exposes a clean public API: play(), pause(), seek(), setViewpoint()

import { StreamClient, PacketType } from './stream-client.js';
import { SplatRenderer, isWebGPUAvailable } from './splat-renderer.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Number of frames to buffer ahead for smooth playback. */
const FRAME_BUFFER_SIZE = 4;

/** Floats per Gaussian in the packed attribute layout. */
const FLOATS_PER_GAUSSIAN = 59; // pos(3) + scale(3) + rot(4) + opacity(1) + sh(48)

// ---------------------------------------------------------------------------
// Minimal math helpers (avoids pulling in gl-matrix for a zero-dep demo)
// ---------------------------------------------------------------------------

function mat4Identity() {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]);
}

function mat4Perspective(fovY, aspect, near, far) {
    const f = 1.0 / Math.tan(fovY / 2);
    const nf = 1.0 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
    ]);
}

function mat4LookAt(eye, center, up) {
    const zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let len = Math.hypot(zx, zy, zz);
    const z0 = zx / len, z1 = zy / len, z2 = zz / len;

    const xx = up[1] * z2 - up[2] * z1;
    const xy = up[2] * z0 - up[0] * z2;
    const xz = up[0] * z1 - up[1] * z0;
    len = Math.hypot(xx, xy, xz);
    const x0 = xx / len, x1 = xy / len, x2 = xz / len;

    const y0 = z1 * x2 - z2 * x1;
    const y1 = z2 * x0 - z0 * x2;
    const y2 = z0 * x1 - z1 * x0;

    return new Float32Array([
        x0, y0, z0, 0,
        x1, y1, z1, 0,
        x2, y2, z2, 0,
        -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]),
        -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]),
        -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]),
        1,
    ]);
}

// ---------------------------------------------------------------------------
// HeimdallViewer
// ---------------------------------------------------------------------------

/**
 * The main viewer class. Manages rendering mode selection, data ingestion
 * (live or VOD), frame buffering, and the render loop.
 *
 * @example
 *   const viewer = new HeimdallViewer(canvas, {
 *       mode: 'live',
 *       signalingUrl: 'https://example.com/rtc/offer',
 *   });
 *   await viewer.init();
 *   viewer.play();
 */
export class HeimdallViewer {

    /** @type {HTMLCanvasElement} */
    #canvas;

    /** @type {'live' | 'vod'} */
    #mode;

    /** @type {string | null} */
    #signalingUrl;

    /** @type {string | null} */
    #manifestUrl;

    /** @type {SplatRenderer | null} */
    #splatRenderer = null;

    /** @type {StreamClient | null} */
    #streamClient = null;

    /** @type {'webgpu' | 'webgl-fallback'} */
    #renderPath = 'webgpu';

    /** @type {boolean} */
    #playing = false;

    /** @type {number | null} */
    #rafId = null;

    // Frame buffer: ring buffer of decoded Gaussian frames waiting for display.
    /** @type {Array<{ frameId: number, timestampMs: number, data: Float32Array, count: number } | null>} */
    #frameBuffer = new Array(FRAME_BUFFER_SIZE).fill(null);

    /** @type {number} Index of the next frame to write into the ring buffer. */
    #writeHead = 0;

    /** @type {number} Index of the next frame to display. */
    #readHead = 0;

    /** @type {number} Number of buffered frames. */
    #bufferedFrames = 0;

    // VOD state.
    /** @type {object | null} Parsed manifest JSON. */
    #manifest = null;

    /** @type {number} Current playback frame for VOD mode. */
    #currentFrame = 0;

    /** @type {boolean} True while a VOD segment fetch is in flight. */
    #fetching = false;

    // Camera / viewpoint state.
    #eyePosition = new Float32Array([0, 0, 3]);
    #lookAt = new Float32Array([0, 0, 0]);
    #upVector = new Float32Array([0, 1, 0]);
    #fovY = Math.PI / 4;

    // Stats for the HUD overlay.
    /** @type {{ fps: number, gaussianCount: number, latencyMs: number, renderPath: string }} */
    stats = {
        fps: 0,
        gaussianCount: 0,
        latencyMs: 0,
        renderPath: 'unknown',
    };
    #frameTimestamps = [];

    // Callback for external stats consumers.
    /** @type {((stats: object) => void) | null} */
    onStats = null;

    /**
     * @param {HTMLCanvasElement} canvas
     * @param {object} opts
     * @param {'live' | 'vod'} opts.mode
     * @param {string} [opts.signalingUrl]   Required for mode='live'.
     * @param {string} [opts.manifestUrl]    Required for mode='vod'.
     */
    constructor(canvas, opts) {
        this.#canvas = canvas;
        this.#mode = opts.mode ?? 'live';
        this.#signalingUrl = opts.signalingUrl ?? null;
        this.#manifestUrl = opts.manifestUrl ?? null;
    }

    // -- public API -----------------------------------------------------------

    /**
     * Initialize the viewer: detect capabilities, create the renderer, and
     * connect to the data source.
     */
    async init() {
        // -- Step 1: Choose render path based on capability --------------------
        const gpuAvailable = await isWebGPUAvailable();
        if (gpuAvailable) {
            this.#renderPath = 'webgpu';
            this.#splatRenderer = new SplatRenderer();
            const ok = await this.#splatRenderer.init(this.#canvas);
            if (!ok) {
                // WebGPU was reported available but init failed; fall back.
                this.#renderPath = 'webgl-fallback';
                this.#splatRenderer = null;
            }
        } else {
            this.#renderPath = 'webgl-fallback';
        }

        this.stats.renderPath = this.#renderPath;

        // If WebGPU is unavailable we initialize a basic WebGL context for the
        // mesh + VDTM fallback. This demo renders a placeholder message; a full
        // implementation would load mesh GLBs and run the VDTM shader.
        if (this.#renderPath === 'webgl-fallback') {
            this.#initFallbackRenderer();
        }

        // -- Step 2: Connect to data source ------------------------------------
        if (this.#mode === 'live') {
            await this.#initLiveStream();
        } else {
            await this.#initVod();
        }
    }

    /** Start playback and the render loop. */
    play() {
        if (this.#playing) return;
        this.#playing = true;
        this.#tick();
    }

    /** Pause playback (render loop continues but frame advance stops). */
    pause() {
        this.#playing = false;
    }

    /**
     * Seek to a specific frame (VOD mode only).
     * @param {number} frame  Zero-based frame number.
     */
    async seek(frame) {
        if (this.#mode !== 'vod' || !this.#manifest) return;
        this.#currentFrame = Math.max(0, Math.min(frame, this.#manifest.total_frames - 1));
        // Flush the buffer and refetch.
        this.#flushBuffer();
        await this.#fetchVodFrames(this.#currentFrame);
    }

    /**
     * Set the virtual camera viewpoint.
     * @param {number[]} position  [x, y, z]
     * @param {number[]} [target]  [x, y, z] look-at point (default: origin)
     */
    setViewpoint(position, target) {
        this.#eyePosition.set(position);
        if (target) this.#lookAt.set(target);
    }

    /** Tear down the viewer and release all resources. */
    destroy() {
        this.#playing = false;
        if (this.#rafId !== null) {
            cancelAnimationFrame(this.#rafId);
            this.#rafId = null;
        }
        this.#streamClient?.disconnect();
        this.#splatRenderer?.destroy();
    }

    // -- private: live stream -------------------------------------------------

    async #initLiveStream() {
        if (!this.#signalingUrl) {
            throw new Error('HeimdallViewer: signalingUrl is required for live mode');
        }

        this.#streamClient = new StreamClient(this.#signalingUrl);

        this.#streamClient.addEventListener('gaussianframe', (e) => {
            this.#handleGaussianPacket(e.detail.header, e.detail.payload);
        });

        this.#streamClient.addEventListener('manifest', (e) => {
            this.#manifest = e.detail.json;
        });

        this.#streamClient.addEventListener('error', (e) => {
            console.warn('[heimdall] stream error:', e.detail.error);
        });

        this.#streamClient.addEventListener('statechange', (e) => {
            console.info('[heimdall] stream state:', e.detail.state);
        });

        await this.#streamClient.connect();
    }

    /**
     * Process a Gaussian packet from the live stream.
     *
     * The payload is an SPZ-compressed blob. In a production build this would
     * go through a WASM SPZ decoder. For this demo, we treat the payload as
     * pre-decoded packed floats (the decoder would be a separate module).
     *
     * A real integration would:
     *   1. Pass payload to spz_decode_wasm(payload) -> Float32Array
     *   2. Feed the result to updateGaussians()
     *
     * Here we stub the decompression and document the interface contract.
     */
    #handleGaussianPacket(header, payload) {
        // --- SPZ decompression stub ------------------------------------------
        // In production:
        //   const decoded = spzDecode(payload);
        //   const { data, count } = decoded;
        //
        // For the demo, we check if the payload happens to be raw float data
        // (useful for testing without the WASM decoder).
        const count = Math.floor(payload.byteLength / (FLOATS_PER_GAUSSIAN * 4));
        if (count === 0) return;

        const data = new Float32Array(payload.buffer, payload.byteOffset, count * FLOATS_PER_GAUSSIAN);

        // Write into the ring buffer.
        this.#frameBuffer[this.#writeHead] = {
            frameId: header.frameId,
            timestampMs: header.timestampMs,
            data,
            count,
        };
        this.#writeHead = (this.#writeHead + 1) % FRAME_BUFFER_SIZE;
        this.#bufferedFrames = Math.min(this.#bufferedFrames + 1, FRAME_BUFFER_SIZE);
    }

    // -- private: VOD ---------------------------------------------------------

    async #initVod() {
        if (!this.#manifestUrl) {
            throw new Error('HeimdallViewer: manifestUrl is required for vod mode');
        }

        const resp = await fetch(this.#manifestUrl);
        if (!resp.ok) {
            throw new Error(`Failed to fetch manifest: ${resp.status}`);
        }
        this.#manifest = await resp.json();
        this.#currentFrame = 0;

        // Pre-fetch the first batch of frames.
        await this.#fetchVodFrames(0);
    }

    /**
     * Fetch VOD frames starting from `startFrame` and load them into the
     * frame buffer.
     */
    async #fetchVodFrames(startFrame) {
        if (!this.#manifest || this.#fetching) return;
        this.#fetching = true;

        try {
            const segments = this.#manifest.segments ?? [];
            const fps = this.#manifest.fps ?? 30;

            // Find the segment containing startFrame.
            for (const seg of segments) {
                if (startFrame < seg.start_frame || startFrame > seg.end_frame) continue;

                const frames = seg.frames ?? [];
                for (const frameMeta of frames) {
                    if (frameMeta.frame < startFrame) continue;
                    if (this.#bufferedFrames >= FRAME_BUFFER_SIZE) break;

                    const url = `${seg.base_url}${frameMeta.file}`;
                    try {
                        const resp = await fetch(url);
                        if (!resp.ok) continue;
                        const arrayBuf = await resp.arrayBuffer();

                        // In production: parse glTF/GLB, extract Gaussian accessor
                        // data or SPZ extension payload, decode SPZ if needed.
                        //
                        // For the demo stub we check for raw float data.
                        const floatCount = arrayBuf.byteLength / 4;
                        const gaussianCount = Math.floor(floatCount / FLOATS_PER_GAUSSIAN);
                        if (gaussianCount === 0) continue;

                        const data = new Float32Array(arrayBuf);
                        this.#frameBuffer[this.#writeHead] = {
                            frameId: frameMeta.frame,
                            timestampMs: Math.round((frameMeta.frame / fps) * 1000),
                            data,
                            count: gaussianCount,
                        };
                        this.#writeHead = (this.#writeHead + 1) % FRAME_BUFFER_SIZE;
                        this.#bufferedFrames = Math.min(this.#bufferedFrames + 1, FRAME_BUFFER_SIZE);
                    } catch (err) {
                        console.warn(`[heimdall] Failed to fetch frame ${frameMeta.frame}:`, err);
                    }
                }
                break; // Only process the matching segment.
            }
        } finally {
            this.#fetching = false;
        }
    }

    // -- private: render loop -------------------------------------------------

    #tick = () => {
        this.#rafId = requestAnimationFrame(this.#tick);

        // Advance to the next buffered frame if playing.
        if (this.#playing && this.#bufferedFrames > 0) {
            const frame = this.#frameBuffer[this.#readHead];
            if (frame) {
                this.#displayFrame(frame);
                this.#frameBuffer[this.#readHead] = null;
                this.#readHead = (this.#readHead + 1) % FRAME_BUFFER_SIZE;
                this.#bufferedFrames--;

                if (this.#mode === 'vod') {
                    this.#currentFrame = frame.frameId + 1;
                }
            }
        }

        // In VOD mode, keep the buffer topped up.
        if (this.#mode === 'vod' && this.#bufferedFrames < FRAME_BUFFER_SIZE / 2) {
            this.#fetchVodFrames(this.#currentFrame);
        }

        // Render the current state (even when paused, so the camera can move).
        this.#renderFrame();

        // Update FPS stats.
        this.#updateStats();
    };

    /**
     * Push a decoded frame to the active renderer.
     */
    #displayFrame(frame) {
        if (this.#splatRenderer?.ready) {
            this.#splatRenderer.updateGaussians(frame.data, frame.count);
        }

        this.stats.gaussianCount = frame.count;
        this.stats.latencyMs = performance.now() - frame.timestampMs;
    }

    /**
     * Issue the draw call for the current frame.
     */
    #renderFrame() {
        // Ensure canvas matches its CSS layout size.
        const dpr = window.devicePixelRatio || 1;
        const rect = this.#canvas.getBoundingClientRect();
        const w = Math.round(rect.width * dpr);
        const h = Math.round(rect.height * dpr);
        if (this.#canvas.width !== w || this.#canvas.height !== h) {
            this.#canvas.width = w;
            this.#canvas.height = h;
        }

        if (this.#renderPath === 'webgpu' && this.#splatRenderer?.ready) {
            const aspect = w / h;
            const view = mat4LookAt(this.#eyePosition, this.#lookAt, this.#upVector);
            const proj = mat4Perspective(this.#fovY, aspect, 0.1, 100.0);
            this.#splatRenderer.render(view, proj);
        }
        // The fallback renderer draws via WebGL and does not need per-frame
        // calls here; it runs its own draw in #initFallbackRenderer or upon
        // receiving mesh data. For the demo we leave the canvas showing the
        // fallback message.
    }

    // -- private: fallback renderer -------------------------------------------

    /**
     * Initialize a basic WebGL context and display a fallback message.
     *
     * A full implementation would load mesh GLBs and run the VDTM shader
     * (see shaders/vdtm.glsl) to project camera video feeds onto the mesh.
     * For this demo we display a placeholder indicating that the VDTM
     * fallback path is active.
     */
    #initFallbackRenderer() {
        const gl = this.#canvas.getContext('webgl2') || this.#canvas.getContext('webgl');
        if (!gl) {
            console.error('[heimdall] Neither WebGPU nor WebGL available');
            return;
        }

        gl.clearColor(0.08, 0.08, 0.12, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        // Draw a simple text overlay via 2D canvas fallback.
        const ctx2d = document.createElement('canvas').getContext('2d');
        if (ctx2d) {
            const c = ctx2d.canvas;
            c.width = this.#canvas.width;
            c.height = this.#canvas.height;
            ctx2d.fillStyle = '#14141a';
            ctx2d.fillRect(0, 0, c.width, c.height);
            ctx2d.fillStyle = '#aaaacc';
            ctx2d.font = '20px monospace';
            ctx2d.textAlign = 'center';
            ctx2d.fillText('heimdall - VDTM fallback (mesh + texture mapping)', c.width / 2, c.height / 2 - 12);
            ctx2d.fillText('WebGPU not available; using WebGL mesh renderer', c.width / 2, c.height / 2 + 16);

            // Copy onto the WebGL canvas as a texture (simplest approach for a demo).
            const tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        }

        console.info('[heimdall] Fallback renderer initialized (mesh + VDTM path)');
    }

    // -- private: frame buffer management -------------------------------------

    #flushBuffer() {
        this.#frameBuffer.fill(null);
        this.#readHead = 0;
        this.#writeHead = 0;
        this.#bufferedFrames = 0;
    }

    // -- private: stats -------------------------------------------------------

    #updateStats() {
        const now = performance.now();
        this.#frameTimestamps.push(now);

        // Keep only timestamps from the last second.
        while (this.#frameTimestamps.length > 0 && this.#frameTimestamps[0] < now - 1000) {
            this.#frameTimestamps.shift();
        }
        this.stats.fps = this.#frameTimestamps.length;

        if (this.onStats) {
            this.onStats(this.stats);
        }
    }
}
