# RTSP SEI Browser Overlay Design

## Goal

Replace the JPEG upload and MJPEG preview path with a browser player that:

- preserves the original H.264 or H.265 encoded video access units;
- carries RKNode schema-v2 inference results in unregistered-user-data SEI;
- publishes the RTSP stream through ZLMediaKit without video re-encoding;
- lets the browser receive video and SEI over one direct WS-FLV connection;
- synchronizes SEI by presentation timestamp and renders results in Canvas.

The central platform remains the control plane. It configures gateways, issues
credentials, and reports capability and health, but it never relays media bytes.

## Scope

This change covers:

- RK3588 SEI publication limits, diagnostics, and official DeepLab processing;
- managed built-in and external ZLMediaKit gateways;
- gateway, publication, Hook, and browser playback authorization;
- task media configuration and legacy task migration;
- a shared browser player for task details and the four/six-stream wall;
- removal of the complete JPEG/MJPEG preview path;
- online and offline deployment, tests, performance qualification, and rollback.

It does not add audio passthrough, WebRTC, WS-TS, video transcoding, public
Internet exposure for plain WebSocket media, or a metadata side channel.

## Confirmed Decisions

- H.264 is the mandatory first-release codec. H.265 is used unchanged only
  where the browser can play it; no compatibility transcode or MJPEG fallback
  is allowed.
- Publication is video-only. The source RTSP audio track is not retained.
- ZLMediaKit is default-enabled in standard online and offline platform
  deployment. The same gateway contract supports an explicitly configured
  external ZLMediaKit deployment.
- A task selects a managed gateway and stream identifier. Task users do not
  enter arbitrary publication URLs.
- Publish and play use separate, revocable credentials. A node never receives
  a browser credential and a browser never receives a publisher credential.
- IP-and-port-only deployments use `http://` plus `ws://` on a trusted LAN or
  VPN. An HTTPS page must reject a plain-WS gateway instead of downgrading.
- Direct WS-FLV uses exact mpegts.js version `1.8.2`, whose published `gitHead`
  is reviewed commit `791bac89e2ea8cbebe7f7ba247c3b342c09418cc`. Its resolved
  integrity is committed in `package-lock.json`, it is bundled into the
  frontend image, and it is never loaded from a CDN.
- Realtime visual preview is available only to RTSP-input, RKMPP-decoded tasks
  with enabled ZLM SEI output and a healthy gateway. Other tasks retain
  structured results and task health but expose no realtime preview.
- DeepLab follows `scripts/deeplabv3_rknn_infer.py`: resize every class-logit
  plane to source resolution with bilinear interpolation, then apply class-axis
  argmax. The browser renders the resulting class-ID mask and never repeats
  model post-processing.
- The old preview chain is removed completely. There is no hidden fallback or
  mixed-version compatibility mode.

## Alternatives Considered

### Direct WS-FLV with mpegts.js SEI support (selected)

One connection transports both encoded video and its SEI. ZLMediaKit remuxes
RTSP to FLV without decoding or re-encoding video. The browser gets the media
directly from ZLMediaKit, and the platform is not a bandwidth bottleneck.

### WS-TS

WS-TS can preserve elementary stream metadata, but it would require a second
player and compatibility path. It remains a documented future option only.

### WebRTC plus a metadata channel

This can reduce media latency, but it introduces a second synchronization and
authorization path and does not preserve the one-stream SEI contract. It is out
of scope for the first release.

### Burned-in overlay or JPEG preview

Both approaches consume node compute and bandwidth, lose structured overlay
semantics, or add latency. They conflict with the no-reencode requirement and
are rejected.

## Architecture

```text
RTSP camera
    |
    v
RK3588: RKMPP decode -> inference -> original encoded AU + schema-v2 SEI
    |
    | authenticated RTSP publish
    v
ZLMediaKit gateway
    |
    | authenticated direct WS-FLV
    v
Browser: mpegts.js -> <video>
             |
             +-> SEI validator -> PTS queue -> Canvas overlay

Central platform
    +-> gateway configuration and health probes
    +-> on_publish / on_play authorization Hooks
    +-> task capability and short-lived playback descriptors
    `-> ZLMediaKit control calls for lifecycle cleanup
```

There are four implementation boundaries:

1. The node publisher owns access-unit and SEI construction.
2. ZLMediaKit owns RTSP ingest and WS-FLV delivery.
3. The platform owns configuration, authorization, lifecycle, and health.
4. `InferenceStreamPlayer` owns browser compatibility, playback, validation,
   synchronization, and rendering.

No API endpoint proxies the FLV body through the platform.

## Gateway Configuration

A managed gateway record contains:

| Field | Meaning |
| --- | --- |
| `id`, `name` | Stable platform identity and operator label |
| `enabled`, `builtin` | Scheduling flag and deployment ownership |
| `publishHost`, `rtspPort` | Node-reachable RTSP publication origin |
| `playbackHost`, `wsPort` | Browser-reachable WS-FLV origin |
| `apiHost`, `apiPort` | Platform-reachable ZLMediaKit control origin |
| `app` | Fixed allowlisted application name |
| `apiSecretRef` | Restricted control-secret reference |
| `hookSecretRef` | Restricted Hook-identity reference |
| `status` | `disabled`, `probing`, `online`, or `error` |
| `lastProbeAt`, `lastError` | Bounded operational diagnostics |

Host and port fields stay separate because node, browser, and platform routes
can differ across LAN, VPN, NAT, or SSH tunnels. The platform never derives the
browser origin from the publication origin.

Only `online` gateways can be selected for a new desired revision. A probe must
verify the control API plus authenticated `on_publish` and `on_play` Hook
connectivity. APIs expose only whether each secret is configured, never its
value.

## Task Media Contract

The new task contract is:

```json
{
  "media": {
    "zlmSei": {
      "enabled": true,
      "gatewayId": "gateway-id",
      "streamName": "task-stream",
      "reconnectMs": 1000
    }
  }
}
```

`streamName` uses a bounded ASCII identifier grammar and must be unique among
active tasks in the same gateway and application. `reconnectMs` is the first
retry delay, defaults to 1000 ms, and is range-checked from 1000 through 4000
ms. Retry delay doubles and is capped at 4000 ms, so the default sequence is
one, two, and four seconds.

Existing task revisions remain immutable. A task containing only the removed
`outputUri` cannot create a new revision and reports
`media_migration_required` until an administrator chooses a gateway and stream.
The platform must not infer a gateway from a legacy URL because publication and
browser reachability are not equivalent.

The capability response distinguishes task health from preview availability.
Image input, file input, OpenCV decoding, disabled media, invalid stream names,
and unhealthy gateways return a deterministic non-preview capability reason
without degrading inference health.

## Authorization and Secrets

### Publication

For each desired revision, the platform creates an opaque publication token
bound to gateway, task, revision, application, stream, and publisher role. Only
a hash is persisted. The full token is inserted into the node-facing RTSP URL
inside the permission-restricted revision artifact.

ZLMediaKit calls the platform `on_publish` Hook. The platform first validates
the gateway-specific Hook identity, then the token role, hash, binding, current
revision, gateway state, and stream. Any failure denies publication.

### Playback

An authenticated platform user requests a playback descriptor for a capable
task. The platform creates a separate opaque play token, stores only its hash,
and returns the allowlisted WS-FLV URL. The token is valid for connection
establishment for 60 seconds and is bound to the user, gateway, task,
application, stream, and play role.

ZLMediaKit calls `on_play`; the same Hook-identity-first validation applies.
Every reconnect obtains a new token. Publication tokens cannot play and play
tokens cannot publish.

The query string of every RTSP and WS URL is removed before logging or returning
diagnostics. Raw tokens and administrative secrets do not appear in frontend
bundles, normal API responses, task listings, or logs.

Task stop, revision retirement, and gateway disable revoke publication. The
platform also requests ZLMediaKit to close the corresponding media source and
viewers when its control API is reachable. Hook identity or platform failure is
default-deny for new publication and playback.

## Media and SEI Contract

`ZlmSeiOutputNode` retains each original H.264/H.265 access unit and prepends a
single unregistered-user-data SEI NAL. It does not decode or re-encode the video
for publication and does not create an audio track.

The browser accepts only:

- SEI payload type 5;
- UUID `9451ef8f-d241-496a-80ba-6818e24dc04e` as exact bytes;
- valid UTF-8 JSON with `schema_version` equal to 2;
- the current task and revision identity;
- positive, bounded source dimensions and supported result arrays;
- at most 1 MiB of user payload.

The shared envelope continues to carry task and revision identity, frame index,
source width and height, detections, structured results, analytics, and media
fields. Existing JSONL, HTTP, Kafka, and event-media field names do not change.

The reviewed mpegts.js commit emits `SEI_ARRIVED` with `uuid`, `user_data`, and
timestamp-base-adjusted `pts`. FLV timestamps use milliseconds. The project
adapter converts PTS to seconds and hides the upstream event shape from the rest
of the frontend.

Malformed, foreign, stale, unsupported, and over-limit messages are discarded
without logging raw data or changing the visible overlay. Diagnostic counters
are bounded and rate-limited.

## Timestamp Synchronization

The player maintains a PTS-sorted metadata queue. It holds no more than two
seconds or 120 entries and evicts the oldest entry when either limit is reached.

Rendering is driven by `requestVideoFrameCallback`. Its `mediaTime` is compared
with normalized SEI PTS from the same MSE timeline. For each displayed frame,
the renderer selects the newest result not later than the frame plus one-frame
tolerance and discards superseded results. Fixture tests establish the exact
offset and guard against upstream timestamp regressions.

If no valid synchronized result exists for one second of advancing media time,
the player clears Canvas and enters `metadata_degraded`; video continues. A
temporary pause does not age metadata by wall clock. Synchronization never uses
node, gateway, platform, or browser system clocks.

## DeepLab Post-Processing

The node implementation must match the project reference in
`scripts/deeplabv3_rknn_infer.py`:

1. Normalize RKNN output to class-first logits for either NCHW or NHWC output.
2. Bilinearly resize every class-logit plane to the source frame dimensions.
3. Apply class-axis argmax after interpolation.
4. Encode the final source-resolution class-ID mask as `class-rle-v1`.

This replaces the current behavior that performs low-resolution argmax and
then nearest-neighbor preview scaling. The corrected class mask is shared by
SEI, JSONL, HTTP, Kafka, and other structured sinks.

A mask is limited to 3840x2160 decoded pixels and 262144 RLE runs. If an SEI
representation exceeds either limit or the 1 MiB payload limit, the publisher
skips SEI for that frame, increments a rate-limited metric, and preserves all
other sinks. It must not change interpolation order or silently return to
low-resolution argmax.

Canvas uses the deterministic Pascal VOC color map and alpha 0.5. Mask RLE
decoding and raster construction run in a Web Worker; the browser does not
repeat argmax.

## Browser Player and Rendering

`InferenceStreamPlayer` is shared by the task-detail preview and every tile in
the four/six-stream wall. It contains:

- a project-owned mpegts.js adapter;
- compatibility and mixed-content checks;
- playback-session acquisition and retry control;
- UUID/schema/identity/limit validation;
- the bounded PTS synchronization queue;
- a Web Worker for segmentation raster work;
- a Canvas renderer layered over the video element.

Canvas dimensions track the displayed video content rectangle. Coordinate
mapping uses the source dimensions and the same `object-fit: contain` scale and
letterbox offsets as the video. Resizing the tile or entering fullscreen updates
the transform without mutating result coordinates.

The renderer supports:

- primary and secondary YOLO boxes, labels, confidence, track IDs, and parent
  linkage;
- PPOCR detection polygons;
- PPOCR recognition text and confidence;
- DeepLab masks using the Pascal VOC palette and alpha 0.5;
- configured areas and lines plus analytics counts.

The mandatory acceptance browsers are desktop Chrome and Edge, current and
immediately preceding major versions at release time. Firefox, Safari, iOS, and
Android are best effort. Unsupported codec or browser combinations show a
specific state and do not start a fallback.

## Runtime States and Recovery

Player state is one of:

- `unsupported`;
- `waiting_publish`;
- `connecting`;
- `live`;
- `metadata_degraded`;
- `reconnecting`;
- `unauthorized`;
- `codec_unsupported`;
- `stopped`.

Connection failures clear video and overlay, request a new play descriptor, and
retry only the affected player. The default backoff is one, two, and four
seconds, capped at four seconds. A manual reconnect action resets the sequence.

Invalid or missing SEI keeps video live and clears only stale overlays. An
unsupported H.265 decoder enters terminal `codec_unsupported`. An HTTPS page
paired with `ws://` fails before connecting and reports that WSS and a
certificate are required. None of these states changes inference task health.

Under a healthy task and gateway, recovery from a transient disconnect must
complete within five seconds.

## Removal of the Legacy Preview Path

The coordinated release removes:

- `PreviewOutputNode` creation and runtime graph use;
- node JPEG generation and inference-Agent preview polling/upload;
- central JPEG storage, preview sessions, signed frame routes, and MJPEG routes;
- frontend image-stream hooks and `<img>` preview components;
- preview environment variables, examples, tests, and documentation.

Repository-wide negative checks ensure no runtime graph, request, deployment
example, or operator instruction retains the old path. Removal is intentional;
there is no compatibility flag.

## Deployment

The platform Compose stack adds a `media` service using a ZLMediaKit image
pinned by immutable digest. The exact digest becomes a release artifact only
after the end-to-end H.264/SEI fixture passes against it. Online and offline
deployment use the same digest and maintained configuration.

The service includes bounded logs, a health check, RTSP and WS-FLV ports, the
control interface, authenticated Hook URLs, and persistent data only where
required. Standard deployment enables it. External-gateway mode explicitly
disables the built-in service and registers reachable publication, playback,
and control addresses through the same API.

The offline package adds the media image archive and SHA-256 manifest entry.
Image loading and verification scripts validate every archive before Compose
startup. mpegts.js is already inside the frontend image, so an offline browser
does not fetch code from the Internet.

The supported IP-only topology is:

- platform at `http://<platform-ip>:5173`;
- browser media at `ws://<browser-reachable-zlm-ip>:<ws-port>`;
- node publication at
  `rtsp://<node-reachable-zlm-ip>:<rtsp-port>/<app>/<stream>`.

Plain WS must be limited to a trusted LAN or VPN and must not be published to an
untrusted network. SSH tunnels are valid only when each caller can reach the
corresponding configured endpoint.

## Upgrade and Migration

The API, frontend, ZLMediaKit configuration, and RK3588 runtime form one release
because old preview APIs and task media contracts are removed.

Upgrade order is:

1. Stop creation of new revisions and back up the database and configuration.
2. Deploy the matching API, frontend, and media services and run additive
   gateway, secret-reference, publication, session, and capability migrations.
3. Create or verify the default gateway and pass control and Hook probes.
4. Deploy the matching RK3588 node image and verify the new SEI and DeepLab
   behavior.
5. Have an administrator assign gateway and stream values to legacy tasks.
6. Generate new immutable revisions and start tasks.
7. Qualify one stream, then four and six streams, before cleanup.

Legacy `outputUri` values are not auto-converted. Old preview data and old
release images remain until the acceptance gate passes.

## Rollback

Before acceptance, retain the previous complete image set plus the pre-upgrade
database and configuration backup. A rollback stops new tasks and restores the
previous API, frontend, node runtime, and database together. Partial rollback
is unsupported because the contracts intentionally changed.

ZLMediaKit volumes are preserved during rollback. New task configuration must
not be sent to old components; restoring the matching database state handles
that boundary. Destructive cleanup of old images and preview data occurs only
after the new release and rollback drill have passed.

## Verification Strategy

### Node and DeepLab

- H.264/H.265 SEI construction, UUID, EBSP escaping, PTS, size limits, and URL
  redaction;
- DeepLab pixel parity with the Python reference for NCHW, NHWC, rectangular
  inputs, tied logits, and nontrivial class boundaries;
- preservation of JSONL, HTTP, Kafka, and other structured sinks when SEI is
  dropped for size.

### Platform

- gateway CRUD, state transitions, probes, and secret redaction;
- publish/play role isolation, expiry, revision binding, revocation, and Hook
  default-deny behavior;
- capability responses and `media_migration_required` handling;
- lifecycle cleanup and query-string redaction.

### Frontend

- UUID, UTF-8, schema, identity, dimension, array, and size validation;
- RLE worker decoding and Pascal VOC rendering at alpha 0.5;
- PTS queue ordering, eviction, frame selection, and one-second stale clearing;
- contain-mode coordinates for boxes, polygons, masks, lines, and areas;
- all player states, independent retries, and mixed-content rejection.

### End to End

A recorded H.264 stream with known RKNode SEI is published through the exact
pinned ZLMediaKit image and pulled by exact mpegts.js version `1.8.2` at the
reviewed commit. The test compares UUID, JSON payload, and PTS, then verifies
video and Canvas pixels in a real browser. It also confirms that media bytes do
not pass through the API.

Chrome and Edge current/current-minus-one validation covers one, four, and six
streams and interruption of one tile. H.265 is tested on supported and
unsupported environments without transcoding.

### Performance and Hardware

On a trusted gigabit LAN with H.264:

- one YOLO/OCR stream has node-publish-to-corresponding-display P95 at or below
  one second;
- each stream in a four/six-tile wall has P95 at or below 1.5 seconds;
- overlay/video skew is at most one displayed frame;
- a transient disconnect recovers within five seconds.

RK3588 hardware results report P50/P95, official DeepLab post-processing time,
SEI size, sustainable frame rate, and drops. If official DeepLab processing
misses the generic limit, the release documents qualified resolution and frame
rate combinations; it never changes the algorithm. Host-only tests cannot be
reported as board acceptance.

## Acceptance Gate

The release is complete only when:

- the pinned H.264 ZLMediaKit/WS-FLV/SEI fixture passes;
- every supported overlay type renders against recorded fixtures;
- security, migration, and negative legacy-path tests pass;
- frontend build and mandatory browser tests pass;
- online and offline deployments use the same verified media image digest;
- RK3588 hardware validation and the latency matrix are recorded;
- coordinated upgrade and full-release rollback are demonstrated;
- old preview and image cleanup happens only after those checks succeed.
