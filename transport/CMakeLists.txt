project(heimdall_transport LANGUAGES CXX)

# SRT library (libsrt) — install via system package or build from source
# Ubuntu: apt install libsrt-openssl-dev
# macOS: brew install srt
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(SRT srt)
endif()

# For GPU decoder, CUDA and NVIDIA Video Codec SDK are required
# Video Codec SDK: https://developer.nvidia.com/video-codec-sdk
# Headers: nvcuvid.h, cuviddec.h

add_library(heimdall_transport INTERFACE)

target_include_directories(heimdall_transport INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link SRT if found
if(SRT_FOUND)
    target_link_libraries(heimdall_transport INTERFACE ${SRT_LIBRARIES})
    target_include_directories(heimdall_transport INTERFACE ${SRT_INCLUDE_DIRS})
endif()
