from __future__ import annotations

import re
from pathlib import Path

import yaml
from scripts.package_offline_bundle import copy_template

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_ROOT = ROOT / "deploy" / "offline"


def test_template_version_rewrite_does_not_duplicate_business_suffix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.yaml"
    destination = tmp_path / "destination.yaml"
    source.write_text(
        "BASE_IMAGE=example:2026.08.15\n"
        "BUSINESS_IMAGE=example:2026.08.15-business\n",
        encoding="utf-8",
    )

    copy_template(source, destination, "2026.08.15-business")

    assert destination.read_text(encoding="utf-8") == (
        "BASE_IMAGE=example:2026.08.15-business\n"
        "BUSINESS_IMAGE=example:2026.08.15-business\n"
    )


def test_offline_compose_files_never_build_or_pull() -> None:
    compose_files = sorted(OFFLINE_ROOT.glob("**/compose*.yaml"))
    assert len(compose_files) == 13
    for path in compose_files:
        text = path.read_text(encoding="utf-8")
        assert "build:" not in text
        assert "${" not in text

    runtime_compose_files = (
        OFFLINE_ROOT / "platform" / "compose.yaml",
        OFFLINE_ROOT / "trainer" / "compose.yaml",
        OFFLINE_ROOT / "rk3588" / "compose.converter.yaml",
        OFFLINE_ROOT / "rk3588" / "compose.inference.yaml",
    )
    for path in runtime_compose_files:
        text = path.read_text(encoding="utf-8")
        assert "pull_policy: never" in text
        assert re.search(r"image: rknode-.+:2026\.08\.24", text)

    trainer = (OFFLINE_ROOT / "trainer" / "compose.yaml").read_text(encoding="utf-8")
    assert 'command: ["python", "-m", "workers.node_service.main"]' in trainer


def test_offline_enrollment_overlays_match_the_online_bootstrap_contract() -> None:
    trainer = yaml.safe_load(
        (OFFLINE_ROOT / "trainer" / "compose.enrollment.yaml").read_text(
            encoding="utf-8"
        )
    )
    trainer_environment = trainer["services"]["trainer"]["environment"]
    assert trainer_environment["RKNODE_ENROLLMENT_TOKEN_FILE"] == (
        "/run/secrets/rknode-enrollment-token"
    )
    assert trainer["secrets"]["rknode-enrollment-token"]["file"] == (
        "./secrets/trainer-enrollment-token"
    )

    rk3588 = yaml.safe_load(
        (OFFLINE_ROOT / "rk3588" / "compose.enrollment.yaml").read_text(
            encoding="utf-8"
        )
    )
    converter = rk3588["services"]["converter"]
    inference = rk3588["services"]["inference"]
    assert converter["secrets"] != inference["secrets"]
    assert set(rk3588["secrets"]) == {
        "rknode-converter-enrollment-token",
        "rknode-inference-enrollment-token",
    }
    assert rk3588["secrets"]["rknode-converter-enrollment-token"]["file"] == (
        "./secrets/converter-enrollment-token"
    )
    assert rk3588["secrets"]["rknode-inference-enrollment-token"]["file"] == (
        "./secrets/inference-enrollment-token"
    )


def test_offline_compose_profiles_use_fixed_release_tags_and_no_env_files() -> None:
    version = (OFFLINE_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert not list((ROOT / "deploy").glob("**/*.env.example"))
    profiles = sorted(OFFLINE_ROOT.glob("**/compose*.yaml"))
    assert profiles
    for path in profiles:
        text = path.read_text(encoding="utf-8")
        assert ":latest" not in text
        assert "${" not in text
        images = re.findall(r"^\s+image:\s+(.+)$", text, re.MULTILINE)
        for image in images:
            expected_version = "2026.08.24-business" if path.parent.name == "rk3588" else version
            assert image.endswith(f":{expected_version}")
        assert not re.search(r"\b[a-f0-9]{48,}\b", text)


def test_all_delivery_images_have_offline_oci_labels() -> None:
    dockerfiles = (
        "deploy/Dockerfile.api",
        "deploy/Dockerfile.frontend",
        "deploy/media/Dockerfile",
        "deploy/Dockerfile.trainer-torch",
        "deploy/Dockerfile.trainer-paddle",
        "deploy/rk3588/Dockerfile.node",
    )
    for relative in dockerfiles:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "ARG RKNODE_RELEASE_VERSION=dev" in text
        assert 'org.opencontainers.image.version="${RKNODE_RELEASE_VERSION}"' in text
        assert 'io.rknode.offline-ready="true"' in text


def test_paddle_image_pins_native_dependency_stack_and_self_tests() -> None:
    dockerfile = (ROOT / "deploy" / "Dockerfile.trainer-paddle").read_text(encoding="utf-8")
    constraints = (ROOT / "deploy" / "paddleocr-constraints.txt").read_text(
        encoding="utf-8"
    )

    assert "python:3.11-slim-bookworm" in dockerfile
    assert "--constraint /opt/frameworks/paddleocr-constraints.txt" in dockerfile
    assert "for attempt in 1 2 3" in dockerfile
    assert "from ppocr.modeling.architectures import build_model" in dockerfile
    assert "EXPECTED_PADDLE_VERSION" in dockerfile
    assert 'if [ "${EXPECTED_PADDLE_CUDA}" = "true" ]' in dockerfile
    assert "version('paddlepaddle-gpu')" in dockerfile
    assert "nvidia-cuda-runtime-cu12" in dockerfile
    assert "assert not paddle.device.is_compiled_with_cuda()" in dockerfile
    assert '--index "${PADDLE_INDEX_URL}"' in dockerfile
    assert "--index-strategy unsafe-best-match" in dockerfile
    assert '"${PADDLE_PACKAGE}"' in dockerfile
    assert "protobuf==3.20.2" in constraints
    assert "scipy==1.10.1" in constraints


def test_torch_image_prevents_framework_dependency_drift() -> None:
    dockerfile = (ROOT / "deploy" / "Dockerfile.trainer-torch").read_text(
        encoding="utf-8"
    )
    build_script = (ROOT / "scripts" / "build_offline_images.sh").read_text(
        encoding="utf-8"
    )

    assert '"timm>=0.9"' in dockerfile
    assert "--no-deps" in dockerfile
    assert "torch.__version__ == '${TORCH_VERSION}'" in dockerfile
    assert "torchvision.__version__ == '${TORCHVISION_VERSION}'" in dockerfile
    assert "actual == expected" in dockerfile
    assert "--build-arg EXPECTED_CUDA_VERSION=12.4" in build_script
    assert "UV_NO_CACHE=1" in dockerfile
    assert "--no-cache" in dockerfile
    assert "FROM scratch" in dockerfile
    assert "GIT_PYTHON_REFRESH=quiet" in dockerfile


def test_paddle_image_does_not_retain_package_cache_layers() -> None:
    dockerfile = (ROOT / "deploy" / "Dockerfile.trainer-paddle").read_text(
        encoding="utf-8"
    )
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "UV_NO_CACHE=1" in dockerfile
    assert "--no-cache" in dockerfile
    assert "FROM scratch" in dockerfile
    assert "third_party/training/paddleocr/doc" in dockerignore
    assert "third_party/training/paddleocr/ppocr" not in dockerignore


def test_paddle_cuda_delivery_uses_the_pinned_cu126_stack() -> None:
    build_script = (ROOT / "scripts" / "build_offline_images.sh").read_text(
        encoding="utf-8"
    )
    dockerfile = (ROOT / "deploy" / "Dockerfile.trainer-paddle").read_text(
        encoding="utf-8"
    )
    packager = (ROOT / "scripts" / "package_offline_bundle.py").read_text(
        encoding="utf-8"
    )
    compose_profile = (OFFLINE_ROOT / "trainer" / "compose.paddle-cuda.yaml").read_text(
        encoding="utf-8"
    )

    assert "python:3.11-slim-bookworm" in build_script
    assert "https://www.paddlepaddle.org.cn/packages/stable/cu126/" in build_script
    assert "rknode-trainer-paddle-cuda12.6" in build_script
    assert "rknode-trainer-paddle-cuda12.6" not in packager
    assert "rknode-trainer-paddle-cuda12.6" in compose_profile
    assert "paddlepaddle-gpu==3.2.2" in build_script
    assert "version('paddlepaddle-gpu')" in dockerfile
    assert "libpaddle.so" in dockerfile


def test_bundle_runtime_scripts_enforce_no_network_deployment() -> None:
    deploy = (OFFLINE_ROOT / "common" / "deploy.sh").read_text(encoding="utf-8")
    verify = (OFFLINE_ROOT / "common" / "verify.sh").read_text(encoding="utf-8")
    loader = (OFFLINE_ROOT / "common" / "load-images.sh").read_text(encoding="utf-8")
    assert "--pull never --no-build" in deploy
    assert "docker image inspect" in deploy
    assert "read-manifest.py architecture" in deploy
    assert "--enroll" in deploy
    assert "*enrollment*.yaml" in deploy
    assert "./secrets/trainer-enrollment-token" in deploy
    assert "./secrets/converter-enrollment-token" in deploy
    assert "./secrets/inference-enrollment-token" in deploy
    assert "source ./.env" not in deploy
    assert "source ./bundle.env" not in deploy
    assert "sha256sum -c SHA256SUMS" in loader
    assert 'io.rknode.offline-ready' in loader
    assert 'org.opencontainers.image.version' in loader
    assert "docker compose" in verify
    assert "exec -T" in verify
    assert "stat -c %a /data/state/node-token" in verify
    assert "urllib.request" in verify
    assert "enrolled + online" in verify
    assert "--enroll" in verify
    stop = (OFFLINE_ROOT / "common" / "stop.sh").read_text(encoding="utf-8")
    assert "read-manifest.py composeProject" in stop
    assert "${RKNODE_NODE_TOKEN}" not in verify
    assert "${RKNODE_CONVERTER_TOKEN}" not in verify
    assert "${RKNODE_INFERENCE_TOKEN}" not in verify


def test_bundle_packager_covers_the_two_release_delivery_archives() -> None:
    packager = (ROOT / "scripts" / "package_offline_bundle.py").read_text(encoding="utf-8")
    expected = {
        "platform-amd64",
        "converter-rk3588-arm64",
        "inference-rk3588-arm64",
        "rk3588-node-arm64",
    }
    for bundle_name in expected:
        assert f'"{bundle_name}": BundleSpec(' in packager
    assert "shutil.rmtree(destination)" in packager
    assert '"requiresNetworkDuringDeploy": False' in packager
    assert '"secretsIncluded": False' in packager
    assert '"healthKind": spec.health_kind' in packager
    assert '"composeProject": spec.project' in packager
    assert 'destination / "bundle.env"' not in packager
    assert "env_example" not in packager
    assert '"--allow-cross-arch"' in packager
    assert "image_arch != spec.arch" in packager
    assert "strict=True" not in packager


def test_platform_bundle_contains_api_web_and_media_images() -> None:
    from scripts.package_offline_bundle import SPECS

    platform = SPECS["platform-amd64"]
    assert platform.images == (
        "rknode-platform-api:{version}",
        "rknode-platform-web:{version}",
        "rknode-platform-media:{version}",
    )
    compose = yaml.safe_load(
        (OFFLINE_ROOT / "platform" / "compose.yaml").read_text(encoding="utf-8")
    )
    assert set(compose["services"]) == {"api", "frontend", "media"}
    assert compose["services"]["frontend"]["ports"] == ["5173:80"]
    assert compose["services"]["media"]["pull_policy"] == "never"


def test_rk3588_composes_have_no_legacy_preview_path() -> None:
    paths = (
        ROOT / "deploy" / "rk3588" / "compose.yaml",
        ROOT / "deploy" / "nodes" / "rk3588" / "compose.yaml",
        OFFLINE_ROOT / "rk3588" / "compose.inference.yaml",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "rknode_preview" not in text
        assert "preview.jpg" not in text
        assert "preview.mjpeg" not in text
        assert "/data/preview" not in text


def test_every_offline_node_bundle_includes_the_enrollment_overlay() -> None:
    from scripts.package_offline_bundle import SPECS

    for name, spec in SPECS.items():
        if name != "platform-amd64":
            assert "enrollment" in spec.compose_files[-1]


def test_rk3588_delivery_uses_one_image_for_two_container_roles() -> None:
    dockerfile = (ROOT / "deploy" / "rk3588" / "Dockerfile.node").read_text(
        encoding="utf-8"
    )
    compose = (ROOT / "deploy" / "nodes" / "rk3588" / "compose.yaml").read_text(
        encoding="utf-8"
    )
    build_script = (ROOT / "scripts" / "build_offline_images.sh").read_text(
        encoding="utf-8"
    )

    assert 'io.rknode.roles="converter,inference"' in dockerfile
    assert 'io.rknode.face-capabilities="none"' in dockerfile
    assert 'CMD ["python3", "-m", "workers.node_service.main"]' in dockerfile
    assert compose.count("deploy/rk3588/Dockerfile.node") == 1
    assert compose.count("rknode-rk3588-node:2026.08.25-business") == 2
    assert 'RKNODE_NODE_KIND: converter' in compose
    assert 'RKNODE_NODE_KIND: inference' in compose
    assert 'rknode-rk3588-node:${version}' in build_script
    assert '--target rknode-runtime' in build_script
    compact = (ROOT / "scripts" / "compact_rk3588_image.sh").read_text(encoding="utf-8")
    assert 'docker export "${container_id}"' in compact
    assert 'docker import' in compact
    assert 'org.opencontainers.image.revision' in compact
