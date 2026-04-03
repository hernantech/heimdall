project(heimdall_segmentation LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

add_library(heimdall_segmentation
    src/matting.cpp
    src/matting_preprocess.cu
)

target_include_directories(heimdall_segmentation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(heimdall_segmentation PUBLIC
    ${CUDA_LIBRARIES}
)

set_target_properties(heimdall_segmentation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

# TensorRT is expected to be found via CMAKE_PREFIX_PATH or system install.
# The consumer must link -lnvinfer when building the final binary.
