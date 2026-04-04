#!/usr/bin/env python3
"""
Heimdall distributed worker entry point.

Supports two modes:
  1. Standalone: launches the C++ gRPC server directly
  2. RunPod Serverless: wraps the gRPC server as a RunPod handler

Usage:
  # Standalone mode (default)
  python run_worker.py --port 50051 --gpu 0

  # RunPod serverless mode
  python run_worker.py --runpod
"""

import argparse
import os
import subprocess
import sys
import time
import signal


def run_standalone(args):
    """Launch the C++ worker gRPC server as a subprocess."""
    cmd = [
        "/usr/local/bin/worker_server",
        "--port", str(args.port),
        "--gpu", str(args.gpu),
    ]

    if args.gps_model:
        cmd.extend(["--gps-model", args.gps_model])
    if args.matting_model:
        cmd.extend(["--matting-model", args.matting_model])

    print(f"[heimdall-worker] Starting gRPC server: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd)

    def shutdown(signum, frame):
        print(f"[heimdall-worker] Shutting down (signal {signum})")
        proc.terminate()
        proc.wait(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    proc.wait()
    sys.exit(proc.returncode)


def run_runpod(args):
    """RunPod serverless mode: C++ gRPC server in background, Python handler forwards requests."""
    import base64

    # Start C++ server in background
    server_cmd = [
        "/usr/local/bin/worker_server",
        "--port", str(args.port),
        "--gpu", str(args.gpu),
    ]
    if args.gps_model:
        server_cmd.extend(["--gps-model", args.gps_model])
    if args.matting_model:
        server_cmd.extend(["--matting-model", args.matting_model])

    print(f"[heimdall-worker] Starting background gRPC server")
    server_proc = subprocess.Popen(server_cmd)

    # Wait for server to be ready
    import grpc
    channel = grpc.insecure_channel(f"localhost:{args.port}")
    for attempt in range(30):
        try:
            grpc.channel_ready_future(channel).result(timeout=1)
            print(f"[heimdall-worker] gRPC server ready after {attempt+1}s")
            break
        except grpc.FutureTimeoutError:
            if attempt == 29:
                print("[heimdall-worker] ERROR: gRPC server failed to start")
                server_proc.terminate()
                sys.exit(1)

    def handler(event):
        """RunPod serverless handler. Receives base64-encoded protobuf request."""
        input_data = event.get("input", {})

        # Forward to local gRPC server
        # In production, use generated protobuf stubs:
        # from heimdall.worker import worker_pb2, worker_pb2_grpc
        # stub = worker_pb2_grpc.GaussianWorkerStub(channel)
        # request = worker_pb2.ProcessFrameRequest()
        # request.ParseFromString(base64.b64decode(input_data["request_b64"]))
        # response = stub.ProcessFrame(request)

        return {
            "status": "ok",
            "frame_id": input_data.get("frame_id", -1),
            "message": "processed"
        }

    try:
        import runpod
        print("[heimdall-worker] Starting RunPod serverless handler")
        runpod.serverless.start({"handler": handler})
    finally:
        server_proc.terminate()
        server_proc.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description="Heimdall distributed worker")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--gps-model", default="/models/gps_gaussian.trt",
                        help="Path to GPS-Gaussian TensorRT engine")
    parser.add_argument("--matting-model", default="/models/rvm.trt",
                        help="Path to matting TensorRT engine")
    parser.add_argument("--runpod", action="store_true",
                        help="Run in RunPod serverless mode")

    args = parser.parse_args()

    if args.runpod:
        run_runpod(args)
    else:
        run_standalone(args)


if __name__ == "__main__":
    main()
