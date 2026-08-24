#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is not installed" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required to read manifest.json" >&2; exit 1; }

bundle_name="$(python3 ./read-manifest.py bundle)"
bundle_version="$(python3 ./read-manifest.py releaseVersion)"
expected_arch="$(python3 ./read-manifest.py architecture)"
mapfile -t images < <(python3 ./read-manifest.py images --list)

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
if [[ "${host_arch}" != "${expected_arch}" ]]; then
  echo "ERROR: bundle requires ${expected_arch}, host is ${host_arch}" >&2
  exit 1
fi

echo "[3/3] Verifying image identity and architecture"
for image in "${images[@]}"; do
  image_arch="$(docker image inspect "${image}" --format '{{.Architecture}}')"
  image_version="$(docker image inspect "${image}" --format '{{index .Config.Labels "org.opencontainers.image.version"}}')"
  offline_ready="$(docker image inspect "${image}" --format '{{index .Config.Labels "io.rknode.offline-ready"}}')"
  [[ "${image_arch}" == "${expected_arch}" ]] || { echo "ERROR: ${image} architecture is ${image_arch}, expected ${expected_arch}" >&2; exit 1; }
  [[ "${image_version}" == "${bundle_version}" ]] || { echo "ERROR: ${image} version is ${image_version}, expected ${bundle_version}" >&2; exit 1; }
  [[ "${offline_ready}" == "true" ]] || { echo "ERROR: ${image} is not marked offline-ready" >&2; exit 1; }
  echo "OK ${image} (${image_arch}, ${image_version})"
done

echo "Bundle ${bundle_name} is loaded and ready."
