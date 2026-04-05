cmake_minimum_required(VERSION 3.24)
project(heimdall_merge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# heimdall_merge library
# ---------------------------------------------------------------------------
add_library(heimdall_merge
    src/frame_aggregator.cpp
    src/merge_pipeline.cpp
    src/gaussian_merger.cu
)

target_include_directories(heimdall_merge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../gaussian/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../stream/src
)

target_link_libraries(heimdall_merge PUBLIC
    CUDA::cudart
)

# Enable separable compilation for CUDA device-side linking.
set_target_properties(heimdall_merge PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89;90"
    POSITION_INDEPENDENT_CODE ON
)

# Link against sibling modules if they are built as part of the
# parent project. Otherwise these are header-only dependencies.
if(TARGET heimdall_gaussian)
    target_link_libraries(heimdall_merge PUBLIC heimdall_gaussian)
endif()
if(TARGET heimdall_stream)
    target_link_libraries(heimdall_merge PUBLIC heimdall_stream)
endif()
