add_library(heimdall_encode
    src/gltf_writer.cpp
    src/gltf_writer.h
)

target_include_directories(heimdall_encode
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(heimdall_encode PUBLIC cxx_std_17)
