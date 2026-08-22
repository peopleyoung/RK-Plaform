ARG ZLM_BASE_IMAGE
FROM ${ZLM_BASE_IMAGE}

ARG RKNODE_RELEASE_VERSION=dev
ARG RKNODE_SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="RKNode Platform Media" \
      org.opencontainers.image.version="${RKNODE_RELEASE_VERSION}" \
      org.opencontainers.image.revision="${RKNODE_SOURCE_REVISION}" \
      io.rknode.component="platform-media" \
      io.rknode.offline-ready="true"

USER root
COPY --chmod=0444 deploy/media/config.ini.template /opt/rknode-media/config.ini.template
COPY --chmod=0555 deploy/media/render_config.py /usr/local/bin/rknode-render-media-config

WORKDIR /opt/media/bin
ENTRYPOINT ["python3", "/usr/local/bin/rknode-render-media-config"]
# The renderer atomically writes a 0600 config, then execs:
# ./MediaServer -s default.pem -c ../conf/config.ini -l 0
