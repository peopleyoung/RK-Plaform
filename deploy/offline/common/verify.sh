#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

enroll=false
case "${1:-}" in
  "") ;;
  --enroll) enroll=true ;;
  *) echo "Usage: $0 [--enroll]" >&2; exit 2 ;;
esac

project="$(python3 ./read-manifest.py composeProject)"
health_kind="$(python3 ./read-manifest.py healthKind)"
mapfile -t compose_files < <(python3 ./read-manifest.py composeFiles --list)
compose_args=()
for compose_file in "${compose_files[@]}"; do
  if [[ "${compose_file}" == *enrollment*.yaml && "${enroll}" != "true" ]]; then
    continue
  fi
  compose_args+=("-f" "${compose_file}")
done

docker compose -p "${project}" "${compose_args[@]}" ps

verify_node_service() {
  local service="$1"
  local token_mode
  token_mode="$(docker compose -p "${project}" "${compose_args[@]}" exec -T "${service}" stat -c %a /data/state/node-token | tr -d '\r')"
  if [[ "${token_mode}" != "600" ]]; then
    echo "ERROR: ${service} node Token mode is ${token_mode}; expected 600" >&2
    exit 1
  fi
  docker compose -p "${project}" "${compose_args[@]}" exec -T "${service}" python3 -c '
from pathlib import Path
import urllib.request

token = Path("/data/state/node-token").read_text(encoding="utf-8").strip()
request = urllib.request.Request(
    "http://127.0.0.1:10081/health",
    headers={"Authorization": f"Bearer {token}"},
)
with urllib.request.urlopen(request, timeout=10) as response:
    print(response.read().decode("utf-8"))
'
}

case "${health_kind}" in
  platform)
    health_url="http://127.0.0.1:5173/api/v1/ready"
    curl -fsS "${health_url}"
    media_container_id="$(docker compose -p "${project}" "${compose_args[@]}" ps -q media)"
    media_health="$(docker inspect --format '{{.State.Health.Status}}' "${media_container_id}")"
    [[ "${media_health}" == "healthy" ]] || { echo "ERROR: media container health is ${media_health}" >&2; exit 1; }
    nc -z 127.0.0.1 8554
    nc -z 127.0.0.1 8081
    docker compose -p "${project}" "${compose_args[@]}" exec -T api .venv/bin/python -c '
import json
import os
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:8000/api/v1/media-gateways",
    headers={"Authorization": "Bearer " + os.environ["RKNODE_ADMIN_TOKEN"]},
)
with urllib.request.urlopen(request, timeout=10) as response:
    gateways = json.load(response)
builtin = next((item for item in gateways if item["id"] == "gateway_builtin"), None)
if builtin is None or builtin["status"] != "online":
    raise SystemExit("built-in media gateway is not online after authenticated keepalive")
'
    ;;
  trainer) verify_node_service trainer; health_url="trainer container:10081/health" ;;
  converter) verify_node_service converter; health_url="converter container:10081/health" ;;
  inference) verify_node_service inference; health_url="inference container:10081/health" ;;
  rk3588)
    verify_node_service converter
    verify_node_service inference
    health_url="converter and inference container health endpoints"
    ;;
  *) echo "ERROR: unsupported health kind ${health_kind}" >&2; exit 1 ;;
esac

echo
echo "Health check passed: ${health_url}"
if [[ "${health_kind}" != "platform" ]]; then
  echo "Ask the central operator to confirm this endpoint is enrolled + online in System Settings."
fi
