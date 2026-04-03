// stream-client.js -- WebRTC data channel receiver and packet parser for heimdall
//
// Connects to a heimdall streaming server via WebRTC, receives multiplexed
// packets (Gaussian splats + optional video), parses the 18-byte wire headers,
// and emits typed events for downstream consumers.
//
// Wire format (from gs_multiplexer.h):
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

const PACKET_MAGIC = 0x48454D44;
const HEADER_SIZE = 18;

// Packet types matching C++ enum PacketType.
export const PacketType = Object.freeze({
    GaussianKeyframe: 0,
    GaussianDelta:    1,
    VideoFrame:       2,
    Manifest:         3,
});

// Flag bitmask constants.
export const PacketFlags = Object.freeze({
    HasVideo: 1 << 0,
    IsFinal:  1 << 1,
});

/**
 * Parse a single 18-byte packet header from an ArrayBuffer / DataView.
 *
 * @param {DataView} view  DataView over the raw bytes.
 * @param {number}   offset  Byte offset within the view.
 * @returns {{ magic: number, type: number, flags: number, frameId: number,
 *             timestampMs: number, payloadSize: number } | null}
 */
export function parseHeader(view, offset = 0) {
    if (view.byteLength - offset < HEADER_SIZE) return null;

    const magic = view.getUint32(offset, true);
    if (magic !== PACKET_MAGIC) return null;

    return {
        magic,
        type:        view.getUint8(offset + 4),
        flags:       view.getUint8(offset + 5),
        frameId:     view.getUint32(offset + 6, true),
        timestampMs: view.getUint32(offset + 10, true),
        payloadSize: view.getUint32(offset + 14, true),
    };
}

// ---------------------------------------------------------------------------
// StreamClient
// ---------------------------------------------------------------------------

/**
 * StreamClient connects to a heimdall WebRTC signaling endpoint, opens a
 * data channel, and emits parsed packets as events.
 *
 * Events (via addEventListener / removeEventListener):
 *   'gaussianframe'  - { detail: { header, payload: Uint8Array } }
 *   'videoframe'     - { detail: { header, payload: Uint8Array } }
 *   'manifest'       - { detail: { header, json: object } }
 *   'streamend'      - { detail: { header } }
 *   'error'          - { detail: { error } }
 *   'statechange'    - { detail: { state: string } }
 */
export class StreamClient extends EventTarget {

    /** @type {RTCPeerConnection | null} */
    #pc = null;

    /** @type {RTCDataChannel | null} */
    #dc = null;

    /** @type {string} */
    #signalingUrl;

    /** @type {'disconnected' | 'connecting' | 'connected' | 'error'} */
    #state = 'disconnected';

    /** @type {number} Maximum automatic reconnection attempts. */
    #maxReconnectAttempts;

    /** @type {number} Current reconnection attempt counter. */
    #reconnectAttempt = 0;

    /** @type {number | null} Reconnect timer id. */
    #reconnectTimer = null;

    // Reassembly buffer for packets that arrive split across data-channel
    // messages (unlikely with SCTP but we handle it defensively).
    /** @type {Uint8Array | null} */
    #buffer = null;

    // Stats counters.
    /** @type {{ packetsReceived: number, bytesReceived: number, gaussianFrames: number, videoFrames: number, lastTimestampMs: number }} */
    stats = {
        packetsReceived: 0,
        bytesReceived: 0,
        gaussianFrames: 0,
        videoFrames: 0,
        lastTimestampMs: 0,
    };

    /**
     * @param {string} signalingUrl  WebRTC signaling endpoint (HTTP/S).
     * @param {object} [opts]
     * @param {number} [opts.maxReconnectAttempts=5]
     */
    constructor(signalingUrl, opts = {}) {
        super();
        this.#signalingUrl = signalingUrl;
        this.#maxReconnectAttempts = opts.maxReconnectAttempts ?? 5;
    }

    /** Current connection state. */
    get state() { return this.#state; }

    // -- public API -----------------------------------------------------------

    /** Initiate the WebRTC connection. */
    async connect() {
        if (this.#state === 'connecting' || this.#state === 'connected') return;
        this.#setState('connecting');

        try {
            this.#pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
            });

            // Create the data channel that the server expects.
            this.#dc = this.#pc.createDataChannel('heimdall', {
                ordered: true,
                // Large max message size to avoid unnecessary fragmentation.
                maxRetransmits: 3,
            });
            this.#dc.binaryType = 'arraybuffer';
            this.#dc.addEventListener('open',    () => this.#onChannelOpen());
            this.#dc.addEventListener('close',   () => this.#onChannelClose());
            this.#dc.addEventListener('message', (e) => this.#onMessage(e));
            this.#dc.addEventListener('error',   (e) => this.#onChannelError(e));

            this.#pc.addEventListener('iceconnectionstatechange', () => {
                if (this.#pc?.iceConnectionState === 'failed') {
                    this.#handleDisconnect();
                }
            });

            // Create and send the SDP offer to the signaling server.
            const offer = await this.#pc.createOffer();
            await this.#pc.setLocalDescription(offer);

            // Wait for ICE gathering to complete (or timeout).
            await this.#waitForIceGathering(5000);

            const response = await fetch(this.#signalingUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/sdp' },
                body: this.#pc.localDescription.sdp,
            });

            if (!response.ok) {
                throw new Error(`Signaling failed: ${response.status} ${response.statusText}`);
            }

            const answerSdp = await response.text();
            await this.#pc.setRemoteDescription({
                type: 'answer',
                sdp: answerSdp,
            });

        } catch (err) {
            this.#emitError(err);
            this.#setState('error');
            this.#scheduleReconnect();
        }
    }

    /** Gracefully close the connection. */
    disconnect() {
        this.#clearReconnectTimer();
        this.#reconnectAttempt = 0;
        this.#cleanup();
        this.#setState('disconnected');
    }

    // -- private: WebRTC lifecycle --------------------------------------------

    #onChannelOpen() {
        this.#reconnectAttempt = 0;
        this.#setState('connected');
    }

    #onChannelClose() {
        this.#handleDisconnect();
    }

    #onChannelError(event) {
        this.#emitError(event.error ?? new Error('Data channel error'));
    }

    #handleDisconnect() {
        this.#cleanup();
        this.#setState('disconnected');
        this.#scheduleReconnect();
    }

    #cleanup() {
        if (this.#dc) {
            try { this.#dc.close(); } catch { /* ignore */ }
            this.#dc = null;
        }
        if (this.#pc) {
            try { this.#pc.close(); } catch { /* ignore */ }
            this.#pc = null;
        }
        this.#buffer = null;
    }

    // -- private: reconnect logic ---------------------------------------------

    #scheduleReconnect() {
        if (this.#reconnectAttempt >= this.#maxReconnectAttempts) {
            this.#setState('error');
            this.#emitError(new Error(`Gave up after ${this.#maxReconnectAttempts} reconnect attempts`));
            return;
        }
        this.#clearReconnectTimer();
        // Exponential backoff: 1s, 2s, 4s, 8s, 16s ...
        const delayMs = Math.min(1000 * (2 ** this.#reconnectAttempt), 16000);
        this.#reconnectAttempt++;
        this.#reconnectTimer = setTimeout(() => this.connect(), delayMs);
    }

    #clearReconnectTimer() {
        if (this.#reconnectTimer !== null) {
            clearTimeout(this.#reconnectTimer);
            this.#reconnectTimer = null;
        }
    }

    // -- private: ICE gathering -----------------------------------------------

    /** Wait until ICE gathering is complete or timeout expires. */
    #waitForIceGathering(timeoutMs) {
        return new Promise((resolve) => {
            if (this.#pc.iceGatheringState === 'complete') {
                resolve();
                return;
            }
            const timer = setTimeout(resolve, timeoutMs);
            this.#pc.addEventListener('icegatheringstatechange', () => {
                if (this.#pc?.iceGatheringState === 'complete') {
                    clearTimeout(timer);
                    resolve();
                }
            });
        });
    }

    // -- private: message parsing ---------------------------------------------

    /**
     * Handle a binary message from the data channel.
     * Messages may contain one or more complete packets, or a partial packet
     * that must be buffered until the rest arrives.
     */
    #onMessage(event) {
        const incoming = new Uint8Array(event.data);
        this.stats.bytesReceived += incoming.byteLength;

        // Prepend any leftover bytes from the previous message.
        let data;
        if (this.#buffer) {
            data = new Uint8Array(this.#buffer.byteLength + incoming.byteLength);
            data.set(this.#buffer);
            data.set(incoming, this.#buffer.byteLength);
            this.#buffer = null;
        } else {
            data = incoming;
        }

        let offset = 0;

        while (offset < data.byteLength) {
            // Need at least HEADER_SIZE bytes to read a header.
            if (data.byteLength - offset < HEADER_SIZE) {
                this.#buffer = data.slice(offset);
                return;
            }

            const view = new DataView(data.buffer, data.byteOffset + offset, data.byteLength - offset);
            const header = parseHeader(view);
            if (!header) {
                // Lost sync -- discard remaining bytes.
                this.#emitError(new Error('Bad packet magic; stream may be corrupted'));
                this.#buffer = null;
                return;
            }

            const totalPacketSize = HEADER_SIZE + header.payloadSize;
            if (data.byteLength - offset < totalPacketSize) {
                // Incomplete payload -- buffer until next message.
                this.#buffer = data.slice(offset);
                return;
            }

            // Extract payload.
            const payloadStart = offset + HEADER_SIZE;
            const payload = data.slice(payloadStart, payloadStart + header.payloadSize);

            this.#dispatchPacket(header, payload);

            offset += totalPacketSize;
        }
    }

    /**
     * Route a parsed packet to the appropriate event.
     */
    #dispatchPacket(header, payload) {
        this.stats.packetsReceived++;
        this.stats.lastTimestampMs = header.timestampMs;

        // Check for end-of-stream.
        if (header.flags & PacketFlags.IsFinal) {
            this.dispatchEvent(new CustomEvent('streamend', { detail: { header } }));
            return;
        }

        switch (header.type) {
            case PacketType.GaussianKeyframe:
            case PacketType.GaussianDelta:
                this.stats.gaussianFrames++;
                this.dispatchEvent(new CustomEvent('gaussianframe', {
                    detail: { header, payload },
                }));
                break;

            case PacketType.VideoFrame:
                this.stats.videoFrames++;
                this.dispatchEvent(new CustomEvent('videoframe', {
                    detail: { header, payload },
                }));
                break;

            case PacketType.Manifest: {
                let json = null;
                try {
                    const text = new TextDecoder().decode(payload);
                    json = JSON.parse(text);
                } catch (err) {
                    this.#emitError(new Error(`Malformed manifest JSON: ${err.message}`));
                    return;
                }
                this.dispatchEvent(new CustomEvent('manifest', {
                    detail: { header, json },
                }));
                break;
            }

            default:
                // Unknown packet type -- skip silently (forward compatibility).
                break;
        }
    }

    // -- private: helpers -----------------------------------------------------

    #setState(state) {
        if (this.#state === state) return;
        this.#state = state;
        this.dispatchEvent(new CustomEvent('statechange', {
            detail: { state },
        }));
    }

    #emitError(error) {
        this.dispatchEvent(new CustomEvent('error', {
            detail: { error },
        }));
    }
}
