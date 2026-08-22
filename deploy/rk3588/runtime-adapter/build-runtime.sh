#!/bin/sh
set -eu

source_dir="${RKNODE_RUNTIME_SOURCE_DIR:-/opt/rknode/src/nv_video_pipeline}"
build_dir="${RKNODE_RUNTIME_BUILD_DIR:-/tmp/rknode-build}"
build_jobs="${RKNODE_BUILD_JOBS:-2}"

test -f "${source_dir}/CMakeLists.txt"
test -d /opt/rknn-sdk
test -d /opt/rockchip-mpp

cmake -S "${source_dir}" -B "${build_dir}" \
  -DRKNN_SDK_ROOT=/opt/rknn-sdk \
  -DROCKCHIP_MPP_ROOT=/opt/rockchip-mpp \
  -DRKNODE_WITH_RKMPP=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel "${build_jobs}"
"${build_dir}/bin/rknn_protocol_probe"

printf 'RK3588 inference runtime build completed: %s\n' "${build_dir}"
