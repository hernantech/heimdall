add_library(heimdall_fusion
    src/fusion.cpp
    src/tsdf_volume.cu
    src/point_cloud_fusion.cu
)

target_include_directories(heimdall_fusion
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(heimdall_fusion
    PUBLIC  ${CUDA_LIBRARIES}
)

set_target_properties(heimdall_fusion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89;90"
    POSITION_INDEPENDENT_CODE ON
)
