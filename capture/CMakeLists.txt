project(heimdall_capture LANGUAGES CXX CUDA)

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG QUIET)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/capture.proto)

add_library(heimdall_capture
    src/frame_sync.cpp
    src/cuda_ipc_bridge.cu
    ${PROTO_SRCS}
)

target_include_directories(heimdall_capture PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(heimdall_capture PUBLIC
    ${CUDA_LIBRARIES}
    ${Protobuf_LIBRARIES}
)

set_target_properties(heimdall_capture PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
