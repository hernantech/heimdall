project(heimdall_worker LANGUAGES CXX)

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG QUIET)

protobuf_generate_cpp(WORKER_PROTO_SRCS WORKER_PROTO_HDRS proto/worker.proto)

add_library(heimdall_worker
    src/quantizer.cpp
    src/worker_pipeline.cpp
    ${WORKER_PROTO_SRCS}
)

target_include_directories(heimdall_worker PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(heimdall_worker PUBLIC
    heimdall_gaussian
    heimdall_segmentation
    heimdall_transport
    ${Protobuf_LIBRARIES}
)

# Worker gRPC server executable
# add_executable(worker_server src/worker_server_main.cpp)
# target_link_libraries(worker_server heimdall_worker gRPC::grpc++)
