from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "third_party" / "training"

EXPECTED_TRAINING_SOURCES = {
    "yolov5": ("d25a07534c14f44296f9444bab2aa5c601cdaaab", "train.py"),
    "yolov6": ("0e7c2d5a93f6d49ed5ab6f005ccdd9d9bbd3db9b", "tools/train.py"),
    "yolov7": ("c2d39f4db6b82800ce6c61740be5ef82854c5d3e", "train.py"),
    "yolov8": ("4674fe6e003dfbc5f2250d3b39dd31faaf7a9877", "ultralytics/__init__.py"),
    "yolov10": ("81f32c4ee396e679b489ff786faa0f9fa0eec298", "ultralytics/__init__.py"),
    "yolo11": ("0692e9297670acf4cc6d0cec773d7a9493cb8a5f", "ultralytics/__init__.py"),
    "segmentation_models_pytorch": (
        "420ce84b0c2df0286fa9bb2bd1499eea625c9b33",
        "segmentation_models_pytorch/decoders/deeplabv3/model.py",
    ),
    "paddleocr": ("8cce9b6fd7ccb50226d0c38f94054d81c29b8184", "tools/train.py"),
}


def test_training_source_snapshots_match_lock_and_dockerfiles() -> None:
    lock_lines = {
        name: revision
        for name, revision in (
            line.split() for line in (TRAINING_ROOT / "SOURCES.lock").read_text().splitlines()
        )
    }
    assert lock_lines == {
        name: revision for name, (revision, _) in EXPECTED_TRAINING_SOURCES.items()
    }

    for name, (revision, entrypoint) in EXPECTED_TRAINING_SOURCES.items():
        source_root = TRAINING_ROOT / name
        assert (source_root / entrypoint).is_file()
        assert revision == lock_lines[name]
        assert not (source_root / ".git").exists()

    dockerfiles = [
        ROOT / "deploy" / "Dockerfile.trainer-torch",
        ROOT / "deploy" / "Dockerfile.trainer-paddle",
    ]
    combined = "\n".join(path.read_text() for path in dockerfiles)
    for name in EXPECTED_TRAINING_SOURCES:
        assert f"COPY third_party/training/{name} " in combined
    assert not re.search(r"\bgit\s+(clone|fetch|checkout)\b", combined)

    torch_dockerfile = dockerfiles[0].read_text()
    assert (
        "ARG SEGMENTATION_MODELS_PYTORCH_COMMIT="
        f"{EXPECTED_TRAINING_SOURCES['segmentation_models_pytorch'][0]}"
    ) in torch_dockerfile
    assert "/opt/frameworks/segmentation_models_pytorch" in torch_dockerfile
    assert "segmentation-models-pytorch==" not in torch_dockerfile
    assert "--no-build-isolation" in torch_dockerfile
    segmentation_root = TRAINING_ROOT / "segmentation_models_pytorch"
    assert (segmentation_root / "LICENSE").is_file()
    assert (segmentation_root / "pyproject.toml").is_file()
    assert (
        segmentation_root / "segmentation_models_pytorch" / "__version__.py"
    ).read_text().strip() == '__version__ = "0.5.0"'

    weight_root = TRAINING_ROOT / "weights"
    checksums = {
        filename: checksum
        for checksum, filename in (
            line.split() for line in (weight_root / "SHA256SUMS").read_text().splitlines()
        )
    }
    assert checksums == {
        "yolov8n.pt": "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"
    }
    for filename, checksum in checksums.items():
        assert hashlib.sha256((weight_root / filename).read_bytes()).hexdigest() == checksum
    assert "COPY third_party/training/weights /opt/weights" in torch_dockerfile
    assert "RKNODE_PRETRAINED_WEIGHTS_ROOT=/opt/weights" in torch_dockerfile


def test_rk3588_inference_image_builds_only_from_repository_sources() -> None:
    runtime_root = ROOT / "third_party" / "nv_video_pipeline"
    required = [
        "CMakeLists.txt",
        "src/rknn_instance/RknnStructuredInstance.cpp",
        "src/rknn_instance/RknnYoloInstance.cpp",
        "samples/RknnPipelineMain.cpp",
        "samples/RknnInstanceProbe.cpp",
        "samples/ProtocolProbe.cpp",
        "3rdparty/rknpu2/include/rknn_api.h",
        "3rdparty/rknpu2/Linux/aarch64/librknnrt.so",
        "3rdparty/rknpu2/LICENSE",
        "3rdparty/rockchip-mpp/include/rk_mpi.h",
        "3rdparty/rockchip-mpp/lib/librockchip_mpp.so.1",
    ]
    for relative in required:
        assert (runtime_root / relative).is_file()

    dockerfile = (ROOT / "deploy" / "rk3588" / "Dockerfile.node").read_text()
    assert "COPY third_party/nv_video_pipeline /opt/rknode/src/nv_video_pipeline" in dockerfile
    assert "RKNODE_SDK_ROOT=/opt/rknode/src/nv_video_pipeline/3rdparty/rknpu2" not in dockerfile
    assert "-DRKNN_SDK_ROOT=/opt/rknode/src/nv_video_pipeline/3rdparty/rknpu2" in dockerfile
    assert (
        "-DROCKCHIP_MPP_ROOT=/opt/rknode/src/nv_video_pipeline/3rdparty/rockchip-mpp"
        in dockerfile
    )
    assert "COPY workers /opt/rknode/workers" in dockerfile
    assert "COPY deploy/rk3588/runtime-adapter" in dockerfile
    assert "build-essential cmake pkg-config" in dockerfile
    assert "fastapi==0.141.1 uvicorn==0.52.0" in dockerfile
    assert 'io.rknode.build-environment="included"' in dockerfile
    assert 'io.rknode.roles="converter,inference"' in dockerfile
    assert 'io.rknode.face-capabilities="none"' in dockerfile
    assert "from rknn.api import RKNN" in dockerfile
    assert "FROM scratch" in dockerfile
    assert not re.search(r"(?:^|\s)/(?:home|workspace)/", dockerfile)
    assert not re.search(r"\bgit\s+(clone|fetch|checkout)\b", dockerfile)

    build_script = (
        ROOT / "deploy" / "rk3588" / "runtime-adapter" / "build-runtime.sh"
    ).read_text()
    assert "RKNODE_WITH_RKMPP=ON" in build_script
    assert "RKNODE_RUNTIME_SOURCE_DIR" in build_script
    assert "rknn_protocol_probe" in build_script


def test_node_compose_build_contexts_resolve_to_repository_root() -> None:
    compose_files = [
        ROOT / "deploy" / "nodes" / "trainer" / "compose.yaml",
        ROOT / "deploy" / "nodes" / "rk3588" / "compose.yaml",
        ROOT / "deploy" / "rk3588" / "compose.yaml",
    ]
    for compose_file in compose_files:
        text = compose_file.read_text()
        assert "./app/workers" not in text
        assert "/home/" not in text

    assert "context: ../../.." in compose_files[0].read_text()
    assert "context: ../../.." in compose_files[1].read_text()
    assert "context: ../.." in compose_files[2].read_text()


def test_online_trainer_compose_supports_legacy_and_enrollment_credentials() -> None:
    compose_root = ROOT / "deploy" / "nodes" / "trainer"
    base = yaml.safe_load((compose_root / "compose.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load(
        (compose_root / "compose.enrollment.yaml").read_text(encoding="utf-8")
    )
    base_environment = base["services"]["trainer"]["environment"]
    assert base_environment["RKNODE_NODE_TOKEN"] == "${RKNODE_NODE_TOKEN:-}"
    assert base_environment["RKNODE_NODE_TOKEN_FILE"] == (
        "${RKNODE_NODE_TOKEN_FILE:-/data/state/node-token}"
    )
    assert "RKNODE_WORKER_TOKEN" not in base_environment

    service = overlay["services"]["trainer"]
    environment = service["environment"]
    assert environment["RKNODE_ENDPOINT_ID"] == (
        "${RKNODE_ENDPOINT_ID:?set endpoint ID from the platform}"
    )
    assert environment["RKNODE_PLATFORM_URL"] == (
        "${RKNODE_PLATFORM_URL:?set central platform URL}"
    )
    assert environment["RKNODE_ENROLLMENT_TOKEN_FILE"] == (
        "/run/secrets/rknode-enrollment-token"
    )
    assert environment["RKNODE_NODE_TOKEN_FILE"] == "/data/state/node-token"
    assert service["secrets"] == ["rknode-enrollment-token"]
    assert overlay["secrets"]["rknode-enrollment-token"]["file"] == (
        "${RKNODE_ENROLLMENT_TOKEN_PATH:?set enrollment token file path}"
    )


def test_online_rk3588_compose_isolates_converter_and_inference_enrollment() -> None:
    compose_root = ROOT / "deploy" / "nodes" / "rk3588"
    base = yaml.safe_load((compose_root / "compose.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load(
        (compose_root / "compose.enrollment.yaml").read_text(encoding="utf-8")
    )

    expected = {
        "converter": (
            "${RKNODE_CONVERTER_ENDPOINT_ID:?set converter endpoint ID from the platform}",
            "rknode-converter-enrollment-token",
            "${RKNODE_CONVERTER_ENROLLMENT_TOKEN_PATH:?set converter enrollment token file path}",
            "${RKNODE_CONVERTER_TOKEN:-}",
        ),
        "inference": (
            "${RKNODE_INFERENCE_ENDPOINT_ID:?set inference endpoint ID from the platform}",
            "rknode-inference-enrollment-token",
            "${RKNODE_INFERENCE_ENROLLMENT_TOKEN_PATH:?set inference enrollment token file path}",
            "${RKNODE_INFERENCE_TOKEN:-}",
        ),
    }
    for service_name, (endpoint_id, secret_name, secret_path, legacy_token) in expected.items():
        base_service = base["services"][service_name]
        base_environment = base_service["environment"]
        assert base_environment["RKNODE_NODE_TOKEN"] == legacy_token
        assert "RKNODE_WORKER_TOKEN" not in base_environment
        service = overlay["services"][service_name]
        environment = service["environment"]
        assert environment["RKNODE_ENDPOINT_ID"] == endpoint_id
        assert environment["RKNODE_PLATFORM_URL"] == (
            "${RKNODE_PLATFORM_URL:?set central platform URL}"
        )
        assert environment["RKNODE_ENROLLMENT_TOKEN_FILE"] == (
            "/run/secrets/rknode-enrollment-token"
        )
        assert environment["RKNODE_NODE_TOKEN_FILE"] == "/data/state/node-token"
        assert service["secrets"] == [
            {"source": secret_name, "target": "rknode-enrollment-token"}
        ]
        assert overlay["secrets"][secret_name]["file"] == secret_path

    assert base["services"]["converter"]["ports"] == [
        "${RKNODE_CONVERTER_HOST_PORT:-10081}:10081"
    ]
    assert base["services"]["inference"]["ports"] == [
        "${RKNODE_INFERENCE_HOST_PORT:-10082}:10081"
    ]
    assert base["services"]["converter"]["volumes"] == ["converter-data:/data"]
    assert "inference-data:/data" in base["services"]["inference"]["volumes"]


def test_rk3588_inference_compose_owns_media_device_contract() -> None:
    compose_services = [
        (ROOT / "deploy" / "rk3588" / "compose.yaml", "inference-agent"),
        (ROOT / "deploy" / "nodes" / "rk3588" / "compose.yaml", "inference"),
        (
            ROOT / "deploy" / "offline" / "rk3588" / "compose.inference.yaml",
            "inference",
        ),
    ]
    required = (
        "systempaths=unconfined",
        "/dev/dri/card0:/dev/dri/card0",
        "/dev/dri/renderD128:/dev/dri/renderD128",
        "/dev/mpp_service:/dev/mpp_service",
        "/dev/rga:/dev/rga",
        "/dev/dma_heap:/dev/dma_heap",
        "/sys/firmware/devicetree/base:/sys/firmware/devicetree/base:ro",
    )

    for path, service_name in compose_services:
        text = path.read_text()
        marker = f"  {service_name}:"
        assert marker in text
        inference_service = text.split(marker, maxsplit=1)[1]
        for value in required:
            assert value in inference_service

        converter_service = text.split(marker, maxsplit=1)[0]
        assert "systempaths=unconfined" not in converter_service


def test_operator_documentation_uses_unified_node_enrollment() -> None:
    operator_documents = (
        ROOT / "README.md",
        ROOT / "docs" / "simple-node-deployment.md",
        ROOT / "docs" / "system-guide.md",
        ROOT / "docs" / "offline-deployment.md",
    )
    required_terms = {
        "节点宿主机 IP / 域名",
        "RKNODE_ENDPOINT_ID",
        "RKNODE_PLATFORM_URL",
        "RKNODE_ENROLLMENT_TOKEN_FILE",
        "RKNODE_NODE_TOKEN_FILE",
        "pending",
        "claimed",
        "enrolled",
        "compose.enrollment.yaml",
        "172.16.66.249:10081",
        "172.30.82.12:10081",
        "172.30.82.12:10082",
        "172.29.0.1:11081",
        "172.29.0.1:11082",
    }
    for path in operator_documents:
        text = path.read_text(encoding="utf-8")
        assert required_terms <= {term for term in required_terms if term in text}
        legacy_marker = "## 旧版静态 Token 迁移"
        assert legacy_marker in text
        legacy_start = text.index(legacy_marker)
        for assignment in ("RKNODE_NODE_TOKEN=", "RKNODE_WORKER_TOKEN="):
            assert text.find(assignment) in {-1} or text.find(assignment) > legacy_start
        for boundary in (
            "不得暴露到公网",
            "VPN",
            "HTTPS",
            "SSH 隧道",
            "不得保存 SSH 密码",
        ):
            assert boundary in text


def test_emergency_tunnel_example_is_explicitly_temporary() -> None:
    unit = (
        ROOT / "deploy" / "systemd" / "rknode-node-tunnel.service.example"
    ).read_text(encoding="utf-8")
    assert "EMERGENCY ROLLBACK ONLY" in unit
    assert "172.29.0.1:11081:127.0.0.1:10081" in unit
    assert "172.29.0.1:11082:127.0.0.1:10082" in unit
    assert "BatchMode=yes" in unit
