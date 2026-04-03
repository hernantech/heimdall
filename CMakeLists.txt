cmake_minimum_required(VERSION 3.24)
project(heimdall LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

option(BUILD_CAPTURE    "Build capture module"       ON)
option(BUILD_CALIBRATION "Build calibration tools"   ON)
option(BUILD_SEGMENTATION "Build segmentation module" ON)
option(BUILD_DEPTH      "Build depth estimation"     ON)
option(BUILD_FUSION     "Build fusion module"        ON)
option(BUILD_MESH       "Build mesh processing"      ON)
option(BUILD_TEXTURE    "Build texturing module"     ON)
option(BUILD_ENCODE     "Build encoding/packaging"   ON)
option(BUILD_STREAM     "Build streaming module"     ON)
option(BUILD_TESTS      "Build tests"                ON)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)

if(BUILD_CAPTURE)
    add_subdirectory(capture)
endif()

if(BUILD_CALIBRATION)
    add_subdirectory(calibration)
endif()

if(BUILD_SEGMENTATION)
    add_subdirectory(segmentation)
endif()

if(BUILD_DEPTH)
    add_subdirectory(depth)
endif()

if(BUILD_FUSION)
    add_subdirectory(fusion)
endif()

if(BUILD_MESH)
    add_subdirectory(mesh)
endif()

if(BUILD_TEXTURE)
    add_subdirectory(texture)
endif()

if(BUILD_ENCODE)
    add_subdirectory(encode)
endif()

if(BUILD_STREAM)
    add_subdirectory(stream)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
