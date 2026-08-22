#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./bundle.env
source ./.env

enrollment_complete="${RKNODE_ENROLLMENT_COMPLETE:-false}"
case "${enrollment_complete}" in
  true|false) ;;
  *)
    echo "ERROR: RKNODE_ENROLLMENT_COMPLETE must be true or false" >&2
    exit 1
    ;;
esac

compose_args=()
for compose_file in ${RKNODE_COMPOSE_FILES}; do
  if [[ "${enrollment_complete}" == "true" && "${compose_file}" == "compose.enrollment.yaml" ]]; then
    continue
  fi
  compose_args+=("-f" "${compose_file}")
done

docker compose -p "${RKNODE_COMPOSE_PROJECT}" --env-file .env \
  "${compose_args[@]}" down

echo "Containers stopped. Persistent volumes were retained."
