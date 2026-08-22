#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./bundle.env

if [[ ! -f .env ]]; then
  echo "ERROR: create .env from ${RKNODE_ENV_EXAMPLE} and fill in this node's values" >&2
  exit 1
fi

if grep -Eq '(^|=)(replace-with|replace-|.*CENTRAL_SERVER_IP)' .env; then
  echo "ERROR: .env still contains placeholder values" >&2
  exit 1
fi

set -a
source ./.env
set +a

enrollment_complete="${RKNODE_ENROLLMENT_COMPLETE:-false}"
case "${enrollment_complete}" in
  true|false) ;;
  *)
    echo "ERROR: RKNODE_ENROLLMENT_COMPLETE must be true or false" >&2
    exit 1
    ;;
esac

require_enrollment_file() {
  local variable_name="$1"
  local path="${!variable_name:-}"
  if [[ -z "${path}" || ! -s "${path}" ]]; then
    echo "ERROR: ${variable_name} must point to a non-empty enrollment credential file" >&2
    exit 1
  fi
}

if [[ "${enrollment_complete}" == "false" && " ${RKNODE_COMPOSE_FILES} " == *" compose.enrollment.yaml "* ]]; then
  case "${RKNODE_HEALTH_KIND}" in
    trainer)
      require_enrollment_file RKNODE_ENROLLMENT_TOKEN_PATH
      ;;
    converter)
      require_enrollment_file RKNODE_CONVERTER_ENROLLMENT_TOKEN_PATH
      ;;
    inference)
      require_enrollment_file RKNODE_INFERENCE_ENROLLMENT_TOKEN_PATH
      ;;
    rk3588)
      require_enrollment_file RKNODE_CONVERTER_ENROLLMENT_TOKEN_PATH
      require_enrollment_file RKNODE_INFERENCE_ENROLLMENT_TOKEN_PATH
      ;;
  esac
fi

if [[ "${enrollment_complete}" == "true" ]]; then
  for token_variable in RKNODE_NODE_TOKEN RKNODE_WORKER_TOKEN RKNODE_CONVERTER_TOKEN RKNODE_INFERENCE_TOKEN; do
    if [[ -n "${!token_variable:-}" ]]; then
      echo "ERROR: clear ${token_variable} before entering steady-state deployment" >&2
      exit 1
    fi
  done
fi

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

for image in ${RKNODE_IMAGES}; do
  docker image inspect "${image}" >/dev/null 2>&1 || {
    echo "ERROR: image ${image} is missing; run ./load-images.sh first" >&2
    exit 1
  }
done

compose_args=()
for compose_file in ${RKNODE_COMPOSE_FILES}; do
  if [[ "${enrollment_complete}" == "true" && "${compose_file}" == "compose.enrollment.yaml" ]]; then
    continue
  fi
  compose_args+=("-f" "${compose_file}")
done

docker compose -p "${RKNODE_COMPOSE_PROJECT}" --env-file .env \
  "${compose_args[@]}" config --quiet
docker compose -p "${RKNODE_COMPOSE_PROJECT}" --env-file .env \
  "${compose_args[@]}" up -d --pull never --no-build
docker compose -p "${RKNODE_COMPOSE_PROJECT}" --env-file .env \
  "${compose_args[@]}" ps

echo "Deployment started without pulling or building images (enrollment complete: ${enrollment_complete}). Run ./verify.sh next."
