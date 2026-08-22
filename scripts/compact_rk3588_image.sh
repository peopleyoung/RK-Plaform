#!/usr/bin/env bash
set -euo pipefail

image_ref="${1:?usage: compact_rk3588_image.sh IMAGE_REF}"
compact_ref="${image_ref}-compact"
source_id="$(docker image inspect "${image_ref}" --format '{{.Id}}')"
version="$(docker image inspect "${image_ref}" --format '{{index .Config.Labels "org.opencontainers.image.version"}}')"
revision="$(docker image inspect "${image_ref}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
container_id=""

cleanup() {
  if [[ -n "${container_id}" ]]; then
    docker rm "${container_id}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

container_id="$(docker create "${image_ref}")"
docker export "${container_id}" | docker import \
  --change 'ENV PYTHONDONTWRITEBYTECODE=1' \
  --change 'ENV PYTHONUNBUFFERED=1' \
  --change 'ENV PYTHONPATH=/opt/rknode' \
  --change 'WORKDIR /opt/rknode' \
  --change 'ENTRYPOINT []' \
  --change 'CMD ["python3", "-m", "workers.node_service.main"]' \
  --change 'EXPOSE 10081' \
  --change 'LABEL org.opencontainers.image.title=RKNode-RK3588-Node' \
  --change "LABEL org.opencontainers.image.version=${version}" \
  --change "LABEL org.opencontainers.image.revision=${revision}" \
  --change 'LABEL io.rknode.component=rk3588-node' \
  --change 'LABEL io.rknode.roles=converter,inference' \
  --change 'LABEL io.rknode.accelerator=rk3588' \
  --change 'LABEL io.rknode.runtime=cpp-rknn' \
  --change 'LABEL io.rknode.inference-business=analytics,bytetrack,secondary-yolo,event-media,kafka,zlm-sei' \
  --change 'LABEL io.rknode.inference-concurrency=context-pool,worker-pool' \
  --change 'LABEL io.rknode.face-capabilities=none' \
  --change 'LABEL io.rknode.build-environment=included' \
  --change 'LABEL io.rknode.offline-ready=true' \
  - "${compact_ref}"

docker tag "${compact_ref}" "${image_ref}"
docker image rm "${source_id}" >/dev/null 2>&1 || true
docker image rm "${compact_ref}" >/dev/null 2>&1 || true
trap - EXIT
cleanup

docker image inspect "${image_ref}" --format 'compacted image={{.RepoTags}} size={{.Size}} version={{index .Config.Labels "org.opencontainers.image.version"}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
