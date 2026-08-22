#!/bin/sh
set -eu
exec python3 -m workers.inference_agent.runtime_adapter probe
