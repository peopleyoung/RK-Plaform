#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./bundle.env

command -v docker >/dev/null 2>&1 || {
  echo "ERROR: docker is not installed" >&2
  exit 1
}

echo "[1/3] Verifying bundle checksums"
sha256sum -c SHA256SUMS

echo "[2/3] Loading Docker images"
for archive in images/*.tar.gz; do
  echo "Loading ${archive}"
  gzip -dc "${archive}" | docker load
done

normalize_arch() {
  case "$1" in
    x86_64) echo amd64 ;;
    aarch64) echo arm64 ;;
    *) echo "$1" ;;
  esac
}

host_arch="$(normalize_arch "$(uname -m)")"
if [[ "${host_arch}" != "${RKNODE_EXPECTED_ARCH}" ]]; then
  echo "ERROR: bundle requires ${RKNODE_EXPECTED_ARCH}, host is ${host_arch}" >&2
  exit 1
fi

echo "[3/3] Verifying image identity and architecture"
for image in ${RKNODE_IMAGES}; do
  image_arch="$(docker image inspect "${image}" --format '{{.Architecture}}')"
  image_version="$(docker image inspect "${image}" --format '{{index .Config.Labels "org.opencontainers.image.version"}}')"
  offline_ready="$(docker image inspect "${image}" --format '{{index .Config.Labels "io.rknode.offline-ready"}}')"
  if [[ "${image_arch}" != "${RKNODE_EXPECTED_ARCH}" ]]; then
    echo "ERROR: ${image} architecture is ${image_arch}, expected ${RKNODE_EXPECTED_ARCH}" >&2
    exit 1
  fi
  if [[ "${image_version}" != "${RKNODE_BUNDLE_VERSION}" ]]; then
    echo "ERROR: ${image} version is ${image_version}, expected ${RKNODE_BUNDLE_VERSION}" >&2
    exit 1
  fi
  if [[ "${offline_ready}" != "true" ]]; then
    echo "ERROR: ${image} is not marked offline-ready" >&2
    exit 1
  fi
  echo "OK ${image} (${image_arch}, ${image_version})"
done

echo "Bundle ${RKNODE_BUNDLE_NAME} is loaded and ready."
