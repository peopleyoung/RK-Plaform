# RK3588 inference agent

The agent is a pull-based reconciler for one RK3588 board. It never accepts an inbound control connection. The board registers with a one-time token, polls the platform for desired state, downloads artifacts over the authenticated agent API, verifies SHA-256, and reports deployment target state transitions.

Required environment:

```text
RKNODE_API_URL=http://platform.example/api/v1
RKNODE_NODE_ID=inode_...
RKNODE_REGISTRATION_TOKEN=one-time-token
RKNODE_ADAPTERS=yolo_dfl_split_v1,deeplab_logits_v1,ppocr_db_det_v1,ppocr_ctc_rec_v1
```

The one-time registration token is exchanged for a node access token. The agent stores that token and its last successfully applied revision under `RKNODE_STATE_DIR` (default `/var/lib/rknode/state`) with owner-only permissions, so container restarts do not consume a new registration token or lose deployment state. Mount this directory on persistent storage.

Production mode fails closed. Configure `RKNODE_SELF_TEST_COMMAND`, `RKNODE_MODEL_PROBE_COMMAND`, `RKNODE_RUNTIME_COMMAND`, and `RKNODE_RUNTIME_HEALTH_COMMAND`; each command must return zero only after that operation has actually succeeded. Self-test runs before any desired revision is applied, every model is statically probed before the current pipeline is drained, and the activation command receives the complete revision in `RKNODE_RELEASE_CONFIGS`. Health runs after activation and on every unchanged-revision poll. Set `RKNODE_STAGING_ONLY=true` only for an explicit connectivity test; this mode bypasses runtime checks and activation with visible warnings.

The current combined `nv_video_pipeline` build supports `yolo_dfl_split_v1`,
`deeplab_logits_v1`, `ppocr_db_det_v1`, and `ppocr_ctc_rec_v1`. The agent
downloads each release once, probes all releases, and calls the runtime command
once for the desired revision. This prevents a later release from replacing an
earlier release on the same board. An empty `RKNODE_RELEASE_CONFIGS=[]` is a real
activation that stops the local pipeline.

Run on the board:

```bash
python -m workers.inference_agent.main
```
