project(heimdall_gaussian LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

add_library(heimdall_gaussian
    src/temporal_tracker.cu
)

target_include_directories(heimdall_gaussian PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(heimdall_gaussian PUBLIC
    ${CUDA_LIBRARIES}
    heimdall_capture
)

set_target_properties(heimdall_gaussian PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
