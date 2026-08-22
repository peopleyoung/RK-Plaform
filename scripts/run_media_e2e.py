#!/usr/bin/env python3
"""Run the disposable ZLMediaKit, FastAPI, Vite, and Chromium media gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.platform_api.app import create_app  # noqa: E402
from backend.platform_api.media_service import MediaService  # noqa: E402
from backend.platform_api.settings import Settings  # noqa: E402
from scripts.lock_zlm_base_image import require_repo_digest  # noqa: E402

FIXTURE_ROOT = ROOT / "tests" / "media"
TASK_ID = "media-e2e-task"
REVISION = 1
STREAM_NAME = "media-e2e"
SEI_UUID = "9451ef8f-d241-496a-80ba-6818e24dc04e"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_http(url: str, *, headers: dict[str, str] | None = None, timeout: int = 30) -> Any:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=2) as response:
                body = response.read()
                return json.loads(body) if body else None
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(0.25)
    path = urllib.parse.urlsplit(url).path
    raise RuntimeError(f"HTTP readiness timed out for {path}") from last_error


def wait_http_ok(url: str, *, timeout: int = 30) -> None:
    """Wait for a non-JSON resource such as the Vite fixture page."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                response.read(1)
                return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(0.25)
    path = urllib.parse.urlsplit(url).path
    raise RuntimeError(f"HTTP readiness timed out for {path}") from last_error


def request_json(
    url: str,
    *,
    token: str | None = None,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> Any:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=5) as response:
        body = response.read()
        return json.loads(body) if body else None


def wait_for_media(url: str, timeout: int = 20) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        try:
            response = request_json(url)
            if isinstance(response, dict):
                latest = response
                if response.get("data"):
                    return response
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    return latest


def fixture_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        FIXTURE_ROOT / "compose.yaml",
        FIXTURE_ROOT / "config.ini",
        FIXTURE_ROOT / "fixture-page.html",
        FIXTURE_ROOT / "fixture-player.ts",
        FIXTURE_ROOT / "browser-check.mjs",
        Path(__file__),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def render_zlm_config(target: Path, values: dict[str, str]) -> None:
    template = Template((FIXTURE_ROOT / "config.ini").read_text(encoding="utf-8"))
    target.write_text(template.substitute(values), encoding="utf-8")
    target.chmod(0o600)


def fixture_envelope() -> str:
    payload = {
        "schema_version": 2,
        "task_id": TASK_ID,
        "revision": REVISION,
        "frame_index": 1,
        "width": 640,
        "height": 360,
        "primary_instance": "detector",
        "detections": [
            {
                "x": 64,
                "y": 36,
                "w": 160,
                "h": 90,
                "label": "fixture",
                "confidence": 0.95,
                "class_id": 1,
                "track_id": 7,
            }
        ],
        "detection_results": {},
        "result_type": "segmentation",
        "result": {
            "width": 640,
            "height": 360,
            "source_width": 640,
            "source_height": 360,
            "encoding": "class-rle-v1",
            "labels": ["background", "fixture"],
            "runs": [[0, 57600], [1, 115200], [0, 57600]],
        },
        "structured_results": {},
        "analytics": {},
        "media": {"fixture": True},
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def ffmpeg_filter_value(value: str) -> str:
    if "'" in value:
        raise ValueError("FFmpeg SEI fixture values must not contain single quotes")
    escaped = value.replace("\\", "\\\\").replace(":", "\\:")
    return f"'{escaped}'"


def subprocess_log(path: Path) -> Any:
    return path.open("a", encoding="utf-8")


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def redact_logs(root: Path, secrets_to_redact: list[str]) -> None:
    for path in root.glob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for value in secrets_to_redact:
            text = text.replace(value, "[REDACTED]")
        path.write_text(text, encoding="utf-8")
        path.chmod(0o600)


def locate_ffmpeg(image: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            image,
            "-c",
            "command -v ffmpeg || test ! -x /opt/media/bin/ffmpeg || echo /opt/media/bin/ffmpeg",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    path = result.stdout.strip().splitlines()
    if not path:
        raise RuntimeError("candidate image does not contain FFmpeg")
    return path[-1]


def generate_h264(image: str, ffmpeg: str, work_dir: Path, log_path: Path) -> None:
    sei_value = ffmpeg_filter_value(f"{SEI_UUID}+{fixture_envelope()}")
    bsf = f"h264_metadata=sei_user_data={sei_value}"
    with subprocess_log(log_path) as log:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                ffmpeg,
                "-v",
                f"{work_dir}:/fixture",
                image,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:rate=30",
                "-t",
                "6",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                "30",
                "-bf",
                "0",
                "-bsf:v",
                bsf,
                "-f",
                "mpegts",
                "-y",
                "/fixture/source.ts",
            ],
            check=True,
            stdout=log,
            stderr=log,
            timeout=90,
        )


def context_settings(
    work_dir: Path,
    *,
    api_port: int,
    rtsp_port: int,
    http_port: int,
    admin_token: str,
    worker_token: str,
    api_secret: str,
    hook_identity: str,
) -> Settings:
    return Settings(
        data_dir=work_dir / "data",
        database_url=f"sqlite:///{work_dir / 'platform.db'}",
        model_profiles_path=ROOT / "config" / "model_profiles.json",
        admin_token=admin_token,
        worker_token=worker_token,
        direct_dispatch_enabled=False,
        media_builtin_enabled=True,
        media_publish_host="127.0.0.1",
        media_playback_host="127.0.0.1",
        media_rtsp_port=rtsp_port,
        media_ws_port=http_port,
        media_api_host="127.0.0.1",
        media_api_port=http_port,
        zlm_api_secret=api_secret,
        zlm_hook_identity=hook_identity,
        cors_origins=f"http://127.0.0.1:{api_port}",
    )


def settings_environment(settings: Settings, api_port: int) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "RKNODE_ENVIRONMENT": "test",
        "RKNODE_DATA_DIR": str(settings.data_dir),
        "RKNODE_DATABASE_URL": settings.resolved_database_url,
        "RKNODE_MODEL_PROFILES_PATH": str(settings.model_profiles_path),
        "RKNODE_ADMIN_TOKEN": settings.admin_token.get_secret_value(),
        "RKNODE_WORKER_TOKEN": settings.worker_token.get_secret_value(),
        "RKNODE_DIRECT_DISPATCH_ENABLED": "false",
        "RKNODE_MEDIA_BUILTIN_ENABLED": "true",
        "RKNODE_MEDIA_PUBLISH_HOST": str(settings.media_publish_host),
        "RKNODE_MEDIA_PLAYBACK_HOST": str(settings.media_playback_host),
        "RKNODE_MEDIA_RTSP_PORT": str(settings.media_rtsp_port),
        "RKNODE_MEDIA_WS_PORT": str(settings.media_ws_port),
        "RKNODE_MEDIA_API_HOST": settings.media_api_host,
        "RKNODE_MEDIA_API_PORT": str(settings.media_api_port),
        "RKNODE_ZLM_API_SECRET": settings.zlm_api_secret.get_secret_value(),
        "RKNODE_ZLM_HOOK_IDENTITY": settings.zlm_hook_identity.get_secret_value(),
        "RKNODE_CORS_ORIGINS": f"http://127.0.0.1:{api_port}",
    }


def write_record(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(image: str, record_path: Path) -> bool:
    image = require_repo_digest(image)
    inspected = json.loads(
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )[0]
    if image not in inspected.get("RepoDigests", []):
        raise RuntimeError("the local image does not expose the requested RepoDigest")

    work_dir = Path(tempfile.mkdtemp(prefix="rknode-media-e2e-"))
    work_dir.chmod(0o700)
    project = "rknode_media_" + secrets.token_hex(6)
    ports = {"api": free_port(), "rtsp": free_port(), "http": free_port(), "vite": free_port()}
    values = {
        "RKNODE_ZLM_API_SECRET": secrets.token_hex(32),
        "RKNODE_ZLM_HOOK_IDENTITY": secrets.token_hex(32),
        "RKNODE_MEDIA_FIXTURE_API_PORT": str(ports["api"]),
    }
    admin_token = secrets.token_urlsafe(32)
    worker_token = secrets.token_urlsafe(32)
    all_secrets = [
        values["RKNODE_ZLM_API_SECRET"],
        values["RKNODE_ZLM_HOOK_IDENTITY"],
        admin_token,
        worker_token,
    ]
    config_path = work_dir / "config.ini"
    render_zlm_config(config_path, values)
    settings = context_settings(
        work_dir,
        api_port=ports["vite"],
        rtsp_port=ports["rtsp"],
        http_port=ports["http"],
        admin_token=admin_token,
        worker_token=worker_token,
        api_secret=values["RKNODE_ZLM_API_SECRET"],
        hook_identity=values["RKNODE_ZLM_HOOK_IDENTITY"],
    )
    compose_environment = {
        **os.environ,
        "RKNODE_ZLM_CANDIDATE_IMAGE": image,
        "RKNODE_MEDIA_FIXTURE_RTSP_PORT": str(ports["rtsp"]),
        "RKNODE_MEDIA_FIXTURE_HTTP_PORT": str(ports["http"]),
        "RKNODE_MEDIA_FIXTURE_CONFIG": str(config_path),
        "RKNODE_ZLM_API_SECRET": values["RKNODE_ZLM_API_SECRET"],
    }
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(FIXTURE_ROOT / "compose.yaml"),
    ]
    api_process: subprocess.Popen[str] | None = None
    vite_process: subprocess.Popen[str] | None = None
    publisher_process: subprocess.Popen[str] | None = None
    success = False
    checks: dict[str, bool] = {
        "directWsFlv": False,
        "h264VideoNonblank": False,
        "playAuthorized": False,
        "publishAuthorized": False,
        "seiArrived": False,
    }
    details: dict[str, object] = {}
    try:
        api_log = subprocess_log(work_dir / "api.log")
        api_process = subprocess.Popen(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                "-m",
                "uvicorn",
                "backend.platform_api.app:create_app",
                "--factory",
                "--host",
                "0.0.0.0",
                "--port",
                str(ports["api"]),
            ],
            cwd=ROOT,
            env=settings_environment(settings, ports["vite"]),
            stdout=api_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_http(f"http://127.0.0.1:{ports['api']}/api/v1/ready")

        with subprocess_log(work_dir / "compose.log") as log:
            subprocess.run(
                [*compose, "up", "-d"],
                check=True,
                cwd=ROOT,
                env=compose_environment,
                stdout=log,
                stderr=log,
                timeout=60,
            )
        wait_http(
            f"http://127.0.0.1:{ports['http']}/index/api/getServerConfig?"
            + urllib.parse.urlencode({"secret": values["RKNODE_ZLM_API_SECRET"]})
        )
        deadline = time.monotonic() + 20
        gateway: dict[str, object] | None = None
        while time.monotonic() < deadline:
            try:
                gateway = request_json(
                    f"http://127.0.0.1:{ports['api']}/api/v1/media-gateways/gateway_builtin/probe",
                    token=admin_token,
                    method="POST",
                )
                if gateway.get("status") == "online":
                    break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.5)
        if gateway is None or gateway.get("status") != "online":
            raise RuntimeError("built-in gateway did not become online")

        ffmpeg = locate_ffmpeg(image)
        generate_h264(image, ffmpeg, work_dir, work_dir / "ffmpeg-generate.log")
        service = MediaService(create_app(settings).state.context)
        publish_token = service.issue_publish_credential(
            gateway_id="gateway_builtin",
            task_id=TASK_ID,
            revision=REVISION,
            app="live",
            stream_name=STREAM_NAME,
            principal="media-e2e-publisher",
        )
        play_token, _ = service.issue_play_credential(
            gateway_id="gateway_builtin",
            task_id=TASK_ID,
            revision=REVISION,
            app="live",
            stream_name=STREAM_NAME,
            principal="media-e2e-player",
        )
        all_secrets.extend((publish_token, play_token))
        publish_url = (
            f"rtsp://127.0.0.1:{ports['rtsp']}/live/{STREAM_NAME}?"
            + urllib.parse.urlencode({"publishToken": publish_token})
        )
        publisher_name = project + "_publisher"
        publisher_log = subprocess_log(work_dir / "ffmpeg-publish.log")
        publisher_process = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                publisher_name,
                "--network",
                "host",
                "--entrypoint",
                ffmpeg,
                "-v",
                f"{work_dir}:/fixture:ro",
                image,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-re",
                "-stream_loop",
                "-1",
                "-i",
                "/fixture/source.ts",
                "-c:v",
                "copy",
                "-rtsp_transport",
                "tcp",
                "-f",
                "rtsp",
                publish_url,
            ],
            stdout=publisher_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        media_query = urllib.parse.urlencode(
            {
                "secret": values["RKNODE_ZLM_API_SECRET"],
                "schema": "rtsp",
                "vhost": "__defaultVhost__",
                "app": "live",
                "stream": STREAM_NAME,
            }
        )
        media = wait_for_media(
            f"http://127.0.0.1:{ports['http']}/index/api/getMediaList?{media_query}",
            timeout=20,
        )
        checks["publishAuthorized"] = bool(media.get("data"))
        if not checks["publishAuthorized"]:
            raise RuntimeError("authenticated RTSP publication was not registered")

        vite_log = subprocess_log(work_dir / "vite.log")
        vite_process = subprocess.Popen(
            [
                "npm",
                "exec",
                "vite",
                "--",
                "--host",
                "127.0.0.1",
                "--port",
                str(ports["vite"]),
                "--strictPort",
            ],
            cwd=ROOT,
            stdout=vite_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_http_ok(f"http://127.0.0.1:{ports['vite']}/tests/media/fixture-page.html")
        stream_url = (
            f"ws://127.0.0.1:{ports['http']}/live/{STREAM_NAME}.live.flv?"
            + urllib.parse.urlencode({"playToken": play_token})
        )
        fixture_query = urllib.parse.urlencode(
            {"streamUrl": stream_url, "taskId": TASK_ID, "revision": str(REVISION)}
        )
        browser = subprocess.run(
            [
                "node",
                str(FIXTURE_ROOT / "browser-check.mjs"),
                f"http://127.0.0.1:{ports['vite']}/tests/media/fixture-page.html?{fixture_query}",
                str(work_dir / "browser-failure.png"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=45,
        )
        (work_dir / "browser.log").write_text(
            browser.stdout + "\n" + browser.stderr, encoding="utf-8"
        )
        if browser.returncode != 0:
            raise RuntimeError("browser media assertion failed")
        browser_result = json.loads(browser.stdout)
        checks["directWsFlv"] = str(ports["http"]) in browser_result["websocketPorts"]
        checks["h264VideoNonblank"] = browser_result.get("h264VideoNonblank") is True
        checks["seiArrived"] = browser_result.get("seiArrived") is True
        checks["playAuthorized"] = browser_result.get("passed") is True
        details = {
            "overlayCoordinates": browser_result.get("overlayCoordinates") is True,
            "segmentationRendered": browser_result.get("segmentationRendered") is True,
            "videoFrameCount": int(browser_result.get("videoFrameCount", 0)),
            "seiCount": int(browser_result.get("seiCount", 0)),
            "latencySamplesMs": browser_result.get("latencySamplesMs", []),
            "overlaySkewFrames": browser_result.get("overlaySkewFrames", []),
            "reconnectDurationMs": float(browser_result.get("reconnectDurationMs", 0)),
            "maxQueueDepth": int(browser_result.get("maxQueueDepth", 1)),
            "maxQueueAgeMs": float(browser_result.get("maxQueueAgeMs", 0)),
        }
        success = all(checks.values()) and all(
            details[name] is True for name in ("overlayCoordinates", "segmentationRendered")
        )
        if not success:
            raise RuntimeError("one or more browser media checks failed")
    except Exception as error:
        details = {**details, "failureKind": type(error).__name__}
    finally:
        stop_process(publisher_process)
        if publisher_process is not None:
            subprocess.run(
                ["docker", "rm", "-f", project + "_publisher"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        stop_process(vite_process)
        stop_process(api_process)
        try:
            with subprocess_log(work_dir / "compose.log") as log:
                subprocess.run(
                    [*compose, "logs", "--no-color"],
                    cwd=ROOT,
                    env=compose_environment,
                    stdout=log,
                    stderr=log,
                    timeout=20,
                )
                subprocess.run(
                    [*compose, "down", "--volumes", "--remove-orphans"],
                    cwd=ROOT,
                    env=compose_environment,
                    stdout=log,
                    stderr=log,
                    timeout=30,
                )
        finally:
            redact_logs(work_dir, all_secrets)

    record = {
        "schemaVersion": 1,
        "image": image,
        "verifiedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "fixtureHash": fixture_hash(),
        "passed": success,
        "checks": checks,
        "details": details,
    }
    write_record(record_path, record)
    if success:
        shutil.rmtree(work_dir)
    else:
        print(f"Media E2E failed; redacted logs preserved at {work_dir}", file=sys.stderr)
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image")
    parser.add_argument("--record", type=Path, default=ROOT / "var" / "media-e2e.json")
    parser.add_argument("--candidate-gate", action="store_true")
    args = parser.parse_args()
    image = args.image
    if image is None:
        image = (
            (ROOT / "deploy" / "media" / "zlm-base-image.lock").read_text(encoding="utf-8").strip()
        )
    try:
        passed = run(image, args.record.resolve())
    except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
