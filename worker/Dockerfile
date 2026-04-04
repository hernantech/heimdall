FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libprotobuf-dev protobuf-compiler \
    libgrpc++-dev protobuf-compiler-grpc \
    libssl-dev libz-dev \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# TensorRT (install from NVIDIA repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev \
    || true && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . /src/heimdall

RUN cd /src/heimdall && mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_WORKER=ON \
        -DBUILD_CAPTURE=ON \
        -DBUILD_GAUSSIAN=ON \
        -DBUILD_SEGMENTATION=ON \
        -DBUILD_TRANSPORT=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_FUSION=OFF \
        -DBUILD_MESH=OFF \
        -DBUILD_TEXTURE=OFF \
        -DBUILD_ENCODE=OFF \
        -DBUILD_MERGE=OFF \
        -DBUILD_STREAM=OFF \
    && make -j$(nproc) worker_server

# Runtime image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libprotobuf23 libgrpc++1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir grpcio runpod

# Copy built binary
COPY --from=builder /src/heimdall/build/worker/worker_server /usr/local/bin/

# Copy Python entry point
COPY worker/scripts/run_worker.py /app/

# Models directory (mount or copy at deploy time)
RUN mkdir -p /models

WORKDIR /app

EXPOSE 50051

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD python3 -c "import grpc; ch=grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(ch).result(timeout=3)" || exit 1

ENTRYPOINT ["python3", "run_worker.py"]
CMD ["--port", "50051"]
