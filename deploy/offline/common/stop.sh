#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
project="$(python3 ./read-manifest.py composeProject)"
mapfile -t compose_files < <(python3 ./read-manifest.py composeFiles --list)
compose_args=()
for compose_file in "${compose_files[@]}"; do
  [[ "${compose_file}" == *enrollment*.yaml ]] && continue
  compose_args+=("-f" "${compose_file}")
done

docker compose -p "${project}" "${compose_args[@]}" down
echo "Containers stopped. Persistent volumes were retained."
