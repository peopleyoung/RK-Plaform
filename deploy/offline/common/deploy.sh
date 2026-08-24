#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

enroll=false
case "${1:-}" in
  "") ;;
  --enroll) enroll=true ;;
  *) echo "Usage: $0 [--enroll]" >&2; exit 2 ;;
esac

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is not installed" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required to read manifest.json" >&2; exit 1; }

expected_arch="$(python3 ./read-manifest.py architecture)"
project="$(python3 ./read-manifest.py composeProject)"
health_kind="$(python3 ./read-manifest.py healthKind)"
mapfile -t images < <(python3 ./read-manifest.py images --list)
mapfile -t compose_files < <(python3 ./read-manifest.py composeFiles --list)

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

for image in "${images[@]}"; do
  docker image inspect "${image}" >/dev/null 2>&1 || {
    echo "ERROR: image ${image} is missing; run ./load-images.sh first" >&2
    exit 1
  }
done

compose_args=()
has_enrollment=false
for compose_file in "${compose_files[@]}"; do
  if [[ "${compose_file}" == *enrollment*.yaml ]]; then
    has_enrollment=true
    [[ "${enroll}" == "true" ]] || continue
  fi
  compose_args+=("-f" "${compose_file}")
done

if [[ "${enroll}" == "true" && "${has_enrollment}" != "true" ]]; then
  echo "ERROR: this bundle does not have an enrollment Compose overlay" >&2
  exit 1
fi

if grep -Eq 'replace-with|CENTRAL_SERVER_IP' "${compose_files[@]}"; then
  echo "ERROR: compose files still contain replace-with or CENTRAL_SERVER_IP placeholders" >&2
  exit 1
fi

if [[ "${enroll}" == "true" ]]; then
  case "${health_kind}" in
    trainer) secret_paths=(./secrets/trainer-enrollment-token) ;;
    converter) secret_paths=(./secrets/converter-enrollment-token) ;;
    inference) secret_paths=(./secrets/inference-enrollment-token) ;;
    rk3588) secret_paths=(./secrets/converter-enrollment-token ./secrets/inference-enrollment-token) ;;
    *) secret_paths=() ;;
  esac
  for secret_path in "${secret_paths[@]}"; do
    if [[ ! -s "${secret_path}" ]]; then
      echo "ERROR: missing non-empty enrollment secret ${secret_path}" >&2
      exit 1
    fi
    mode="$(stat -c %a "${secret_path}")"
    [[ "${mode}" == "600" ]] || { echo "ERROR: ${secret_path} mode is ${mode}; expected 600" >&2; exit 1; }
  done
fi

docker compose -p "${project}" "${compose_args[@]}" config --quiet
docker compose -p "${project}" "${compose_args[@]}" up -d --pull never --no-build
docker compose -p "${project}" "${compose_args[@]}" ps

if [[ "${enroll}" == "true" ]]; then
  echo "Deployment started without .env, pulling, or building images (enrollment: true). Run ./verify.sh --enroll next."
else
  echo "Deployment started without .env, pulling, or building images (enrollment: false). Run ./verify.sh next."
fi
