#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

role="${1:-}"

usage() {
  cat <<'EOF'
Usage: scripts/build_offline_images.sh ROLE

Roles:
  platform              Central API and web images
  trainer-torch-cpu     YOLO/DeepLab CPU trainer
  trainer-paddle-cpu    PPOCR CPU trainer
  trainer-torch-cuda    YOLO/DeepLab CUDA 12.4 trainer
  trainer-paddle-cuda   PPOCR CUDA 12.6 trainer
  converter             RK3588 unified node image (compatibility alias)
  inference             RK3588 unified node image (compatibility alias)
  rk3588                Unified converter and inference image (arm64 only)
EOF
}

if [[ -z "${role}" ]]; then
  usage
  exit 2
fi

default_version() {
  case "$1" in
    platform) echo "2026.08.24" ;;
    trainer-*) echo "2026.08.24" ;;
    converter|inference|rk3588) echo "2026.08.24-business" ;;
    *) echo "2026.08.24" ;;
  esac
}

version="${RKNODE_RELEASE_VERSION:-$(default_version "${role}")}"
revision="${RKNODE_SOURCE_REVISION:-source-tree-${version}}"

normalize_arch() {
  case "$1" in
    x86_64) echo amd64 ;;
    aarch64) echo arm64 ;;
    *) echo "$1" ;;
  esac
}

arch="$(normalize_arch "$(uname -m)")"

build_platform() {
  local zlm_base_image
  zlm_base_image="$(tr -d '[:space:]' < deploy/media/zlm-base-image.lock)"
  if [[ ! "${zlm_base_image}" =~ ^zlmediakit/zlmediakit@sha256:[a-f0-9]{64}$ ]]; then
    echo "ERROR: deploy/media/zlm-base-image.lock does not contain a verified immutable image" >&2
    exit 2
  fi
  docker build -f deploy/Dockerfile.api \
    --build-arg PYTHON_INDEX_URL="${RKNODE_PYTHON_INDEX_URL:-https://pypi.org/simple}" \
    --build-arg "RKNODE_RELEASE_VERSION=${version}" \
    --build-arg "RKNODE_SOURCE_REVISION=${revision}" \
    -t "rknode-platform-api:${version}" .
  docker build -f deploy/Dockerfile.frontend \
    --build-arg "RKNODE_RELEASE_VERSION=${version}" \
    --build-arg "RKNODE_SOURCE_REVISION=${revision}" \
    -t "rknode-platform-web:${version}" .
  docker build -f deploy/media/Dockerfile \
    --build-arg "ZLM_BASE_IMAGE=${zlm_base_image}" \
    --build-arg "RKNODE_RELEASE_VERSION=${version}" \
    --build-arg "RKNODE_SOURCE_REVISION=${revision}" \
    -t "rknode-platform-media:${version}" .
}

build_torch_cpu() {
  docker build -f deploy/Dockerfile.trainer-torch \
    --build-arg BASE_IMAGE="${RKNODE_TRAINER_BASE_IMAGE:-python:3.11-slim-bookworm}" \
    --build-arg DEBIAN_MIRROR="${RKNODE_DEBIAN_MIRROR:-http://deb.debian.org}" \
    --build-arg TORCH_VERSION="${RKNODE_TORCH_VERSION:-2.5.1+cpu}" \
    --build-arg TORCHVISION_VERSION="${RKNODE_TORCHVISION_VERSION:-0.20.1+cpu}" \
    --build-arg TORCH_INDEX_URL="${RKNODE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
    --build-arg EXPECTED_CUDA_VERSION=none \
    --build-arg PYTHON_INDEX_URL="${RKNODE_PYTHON_INDEX_URL:-https://pypi.org/simple}" \
    --build-arg "RKNODE_RELEASE_VERSION=${version}" \
    --build-arg "RKNODE_SOURCE_REVISION=${revision}" \
    --build-arg RKNODE_IMAGE_VARIANT=cpu \
    -t "rknode-trainer-torch-cpu:${version}" .
}

build_paddle_cpu() {
  docker build -f deploy/Dockerfile.trainer-paddle \
    --build-arg BASE_IMAGE="${RKNODE_TRAINER_BASE_IMAGE:-python:3.11-slim-bookworm}" \
    --build-arg DEBIAN_MIRROR="${RKNODE_DEBIAN_MIRROR:-http://deb.debian.org}" \
    --build-arg PADDLE_PACKAGE="${RKNODE_PADDLE_PACKAGE:-paddlepaddle==3.2.2}" \
    --build-arg PADDLE_INDEX_URL="${RKNODE_PADDLE_INDEX_URL:-${RKNODE_PYTHON_INDEX_URL:-https://pypi.org/simple}}" \
    --build-arg EXPECTED_PADDLE_VERSION=3.2.2 \
    --build-arg EXPECTED_PADDLE_CUDA=false \
    --build-arg PYTHON_INDEX_URL="${RKNODE_PYTHON_INDEX_URL:-https://pypi.org/simple}" \
    --build-arg "RKNODE_RELEASE_VERSION=${version}" \
    --build-arg "RKNODE_SOURCE_REVISION=${revision}" \
    --build-arg RKNODE_IMAGE_VARIANT=cpu \
    -t "rknode-trainer-paddle-cpu:${version}" .
}

build_torch_cuda() {
  [[ "${arch}" == "amd64" ]] || { echo "ERROR: CUDA trainer requires amd64" >&2; exit 2; }
  docker build -f deploy/Dockerfile.trainer-torch \
    --build-arg BASE_IMAGE="${RKNODE_CUDA_BASE_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}" \
    --build-arg DEBIAN_MIRROR="${RKNODE_DEBIAN_MIRROR:-http://deb.debian.org}" \
    --build-arg TORCH_VERSION="${RKNODE_TORCH_VERSION:-2.5.1+cu124}" \
    --build-arg TORCHVISION_VERSION="${RKNODE_TORCHVISION_VERSION:-0.20.1+cu124}" \
    --build-arg TORCH_INDEX_URL="${RKNODE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}" \
    --build-arg EXPECTED_CUDA_VERSION=12.4 \
    --build-arg PYTHON_INDEX_URL="${RKNODE_PYTHON_INDEX_URL:-https://pypi.org/simple}" \
    --build-arg "RKNODE_RELEASE_VERSION=${version}" \
    --build-arg "RKNODE_SOURCE_REVISION=${revision}" \
    --build-arg RKNODE_IMAGE_VARIANT=cuda12.4 \
    -t "rknode-trainer-torch-cuda12.4:${version}" .
}

build_paddle_cuda() {
  [[ "${arch}" == "amd64" ]] || { echo "ERROR: CUDA trainer requires amd64" >&2; exit 2; }
  docker build -f deploy/Dockerfile.trainer-paddle \
    --build-arg BASE_IMAGE="${RKNODE_PADDLE_CUDA_BASE_IMAGE:-python:3.11-slim-bookworm}" \
    --build-arg DEBIAN_MIRROR="${RKNODE_DEBIAN_MIRROR:-http://deb.debian.org}" \
    --build-arg PADDLE_PACKAGE="${RKNODE_PADDLE_PACKAGE:-paddlepaddle-gpu==3.2.2}" \
    --build-arg PADDLE_INDEX_URL="${RKNODE_PADDLE_INDEX_URL:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}" \
    --build-arg EXPECTED_PADDLE_VERSION=3.2.2 \
    --build-arg EXPECTED_PADDLE_CUDA=true \
    --build-arg PYTHON_INDEX_URL="${RKNODE_PYTHON_INDEX_URL:-https://pypi.org/simple}" \
    --build-arg "RKNODE_RELEASE_VERSION=${version}" \
    --build-arg "RKNODE_SOURCE_REVISION=${revision}" \
    --build-arg RKNODE_IMAGE_VARIANT=cuda12.6 \
    -t "rknode-trainer-paddle-cuda12.6:${version}" .
}

require_arm64() {
  [[ "${arch}" == "arm64" ]] || { echo "ERROR: RK3588 images must be built on arm64" >&2; exit 2; }
}

build_rk3588() {
  require_arm64
  local image_ref="rknode-rk3588-node:${version}"
  docker build -f deploy/rk3588/Dockerfile.node \
    --target rknode-runtime \
    --build-arg RKNN_TOOLKIT_IMAGE="${RKNODE_CONVERTER_BASE_IMAGE:-rknn_toolkit2:2.3.2-debian12-cp311-aarch64}" \
    --build-arg DEBIAN_MIRROR="${RKNODE_DEBIAN_MIRROR:-http://deb.debian.org}" \
    --build-arg PYTHON_INDEX_URL="${RKNODE_PYTHON_INDEX_URL:-https://pypi.org/simple}" \
    --build-arg "RKNODE_RELEASE_VERSION=${version}" \
    --build-arg "RKNODE_SOURCE_REVISION=${revision}" \
    -t "${image_ref}" .
  if [[ "${RKNODE_COMPACT_RK3588_IMAGE:-true}" == "true" ]]; then
    bash scripts/compact_rk3588_image.sh "${image_ref}"
  fi
}

case "${role}" in
  platform) build_platform ;;
  trainer-torch-cpu) build_torch_cpu ;;
  trainer-paddle-cpu) build_paddle_cpu ;;
  trainer-torch-cuda) build_torch_cuda ;;
  trainer-paddle-cuda) build_paddle_cuda ;;
  converter|inference|rk3588) build_rk3588 ;;
  *) usage; exit 2 ;;
esac
