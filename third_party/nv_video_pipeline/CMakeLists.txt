cmake_minimum_required(VERSION 3.15)

project(VideoPipeline)

if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(RKNN_SDK_ROOT "${RKNN_SDK_ROOT}" CACHE PATH "rknpu2 SDK root containing include/rknn_api.h and librknnrt.so")
if (NOT RKNN_SDK_ROOT)
    message(FATAL_ERROR "RK3588 build requires an explicit -DRKNN_SDK_ROOT=<path>")
endif()

find_path(RKNN_INCLUDE_DIR rknn_api.h
    PATHS "${RKNN_SDK_ROOT}/include" "${RKNN_SDK_ROOT}/rknpu2/include"
    NO_DEFAULT_PATH)
find_library(RKNN_RUNTIME_LIBRARY NAMES rknnrt librknnrt
    PATHS "${RKNN_SDK_ROOT}/Linux/aarch64" "${RKNN_SDK_ROOT}/lib" "${RKNN_SDK_ROOT}/lib64"
    NO_DEFAULT_PATH)
if (NOT RKNN_INCLUDE_DIR OR NOT RKNN_RUNTIME_LIBRARY)
    message(FATAL_ERROR
        "RKNN_SDK_ROOT must contain rknn_api.h and librknnrt.so. "
        "Example: -DRKNN_SDK_ROOT=/opt/rknn-sdk")
endif()

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio)
if (NOT OpenCV_FOUND)
    # Debian's split libopencv-*-dev packages contain the headers and shared
    # libraries but leave OpenCVConfig.cmake to the much larger libopencv-dev
    # meta-package. Resolve the four components used by this runtime directly
    # so the compile-capable RK3588 image does not pull unused OpenCV modules.
    find_path(OpenCV_INCLUDE_DIRS opencv2/core.hpp PATHS /usr/include/opencv4)
    foreach(component core imgproc imgcodecs videoio)
        string(TOUPPER "${component}" component_upper)
        find_library(OpenCV_${component_upper}_LIBRARY NAMES "opencv_${component}")
        if (NOT OpenCV_${component_upper}_LIBRARY)
            message(FATAL_ERROR "Missing OpenCV ${component} development library")
        endif()
        list(APPEND OpenCV_LIBS "${OpenCV_${component_upper}_LIBRARY}")
    endforeach()
    if (NOT OpenCV_INCLUDE_DIRS)
        message(FATAL_ERROR "Missing OpenCV development headers")
    endif()
endif()
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(CURL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil)
pkg_check_modules(RDKAFKA REQUIRED rdkafka)

option(RKNODE_WITH_RKMPP "Build the Rockchip MPP RTSP decoder" ON)
if (RKNODE_WITH_RKMPP)
    set(ROCKCHIP_MPP_ROOT "${CMAKE_SOURCE_DIR}/3rdparty/rockchip-mpp" CACHE PATH
        "Rockchip MPP snapshot root")
    find_path(ROCKCHIP_MPP_INCLUDE_DIR rk_mpi.h
        PATHS "${ROCKCHIP_MPP_ROOT}/include" NO_DEFAULT_PATH)
    find_library(ROCKCHIP_MPP_LIBRARY NAMES rockchip_mpp
        PATHS "${ROCKCHIP_MPP_ROOT}/lib" NO_DEFAULT_PATH)
    if (NOT ROCKCHIP_MPP_LIBRARY AND EXISTS "${ROCKCHIP_MPP_ROOT}/lib/librockchip_mpp.so.1")
        set(ROCKCHIP_MPP_LIBRARY "${ROCKCHIP_MPP_ROOT}/lib/librockchip_mpp.so.1")
    endif()
    if (NOT ROCKCHIP_MPP_INCLUDE_DIR OR NOT ROCKCHIP_MPP_LIBRARY)
        message(FATAL_ERROR "RKNODE_WITH_RKMPP requires the bundled Rockchip MPP snapshot")
    endif()
endif()

set(RKNN_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src/compat
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/nodes
    ${CMAKE_SOURCE_DIR}/src/objects
    ${CMAKE_SOURCE_DIR}/src/pipelines
    ${CMAKE_SOURCE_DIR}/src/rknn_instance
    ${CMAKE_SOURCE_DIR}/src/utils
    ${OpenCV_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    ${RDKAFKA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIR})

if (RKNODE_WITH_RKMPP)
    list(APPEND RKNN_INCLUDE_DIRS ${ROCKCHIP_MPP_INCLUDE_DIR})
endif()

set(RKNN_SOURCES
    src/utils/CLog.cpp
    src/objects/FrameBoxTarget.cpp
    src/objects/FrameInferenceResult.cpp
    src/pipelines/InstancesManager.cpp
    src/pipelines/pipeline.cpp
    src/nodes/ProcessNode.cpp
    src/nodes/InferNode.cpp
    src/nodes/AnhuanMessage.cpp
    src/nodes/AnalyticsNode.cpp
    src/nodes/ByteTrackNode.cpp
    src/nodes/EventOutputNode.cpp
    src/nodes/track/bytetrack/ByteTracker.cpp
    src/nodes/track/bytetrack/KalmanFilter.cpp
    src/nodes/track/bytetrack/Strack.cpp
    src/nodes/track/bytetrack/TrackUtil.cpp
    src/nodes/VideoCaptureNode.cpp
    src/nodes/JsonOutputNode.cpp
    src/nodes/KafkaOutputNode.cpp
    src/nodes/SecondaryInferNode.cpp
    src/nodes/ZlmSeiOutputNode.cpp
    src/utils/MediaUrl.cpp
    src/utils/SeiPacket.cpp
    src/rknn_instance/DeepLabPostprocess.cpp
    src/rknn_instance/RknnExecutionPool.cpp
    src/rknn_instance/RknnYoloInstance.cpp
    src/rknn_instance/RknnStructuredInstance.cpp)

if (RKNODE_WITH_RKMPP)
    list(APPEND RKNN_SOURCES src/nodes/RkMppCaptureNode.cpp)
endif()

add_library(VideoPipeline_RKNN SHARED ${RKNN_SOURCES})
target_include_directories(VideoPipeline_RKNN PRIVATE ${RKNN_INCLUDE_DIRS})
target_link_directories(VideoPipeline_RKNN PRIVATE ${FFMPEG_LIBRARY_DIRS} ${RDKAFKA_LIBRARY_DIRS})
target_compile_options(VideoPipeline_RKNN PRIVATE -Wall -Wextra -Wpedantic -Wno-sign-compare -Wno-unused-function)
target_link_libraries(VideoPipeline_RKNN PRIVATE
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    ${RDKAFKA_LIBRARIES}
    Eigen3::Eigen
    yaml-cpp
    spdlog::spdlog
    CURL::libcurl
    ${RKNN_RUNTIME_LIBRARY}
    Threads::Threads)

if (RKNODE_WITH_RKMPP)
    target_link_libraries(VideoPipeline_RKNN PRIVATE ${ROCKCHIP_MPP_LIBRARY})
    target_compile_definitions(VideoPipeline_RKNN PRIVATE RKNODE_WITH_RKMPP=1)
endif()

add_executable(rknn_pipeline samples/RknnPipelineMain.cpp)
target_include_directories(rknn_pipeline PRIVATE ${RKNN_INCLUDE_DIRS})
target_link_libraries(
    rknn_pipeline PRIVATE VideoPipeline_RKNN ${JSONCPP_LIBRARIES} ${OpenCV_LIBS}
    yaml-cpp fmt::fmt Threads::Threads)

add_executable(rknn_smoke samples/RknnSmoke.cpp)
target_include_directories(rknn_smoke PRIVATE ${RKNN_INCLUDE_DIR})
target_link_libraries(rknn_smoke PRIVATE ${RKNN_RUNTIME_LIBRARY} Threads::Threads)

add_executable(rknn_instance_probe samples/RknnInstanceProbe.cpp)
target_include_directories(rknn_instance_probe PRIVATE ${RKNN_INCLUDE_DIRS})
target_link_libraries(
    rknn_instance_probe PRIVATE VideoPipeline_RKNN ${JSONCPP_LIBRARIES} ${OpenCV_LIBS}
    yaml-cpp fmt::fmt Threads::Threads)

add_executable(rknn_protocol_probe samples/ProtocolProbe.cpp)
target_include_directories(rknn_protocol_probe PRIVATE ${RKNN_INCLUDE_DIRS})
target_link_libraries(
    rknn_protocol_probe PRIVATE VideoPipeline_RKNN ${JSONCPP_LIBRARIES} ${OpenCV_LIBS} yaml-cpp fmt::fmt)

message(STATUS "VideoPipeline target: RK3588/RKNN")
message(STATUS "RKNN headers: ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN runtime: ${RKNN_RUNTIME_LIBRARY}")
message(STATUS "Rockchip MPP decode: ${RKNODE_WITH_RKMPP}")
