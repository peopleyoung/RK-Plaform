#!/usr/bin/env python3
"""Export tagged Docker images and no-build Compose files as an offline bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_ROOT = ROOT / "deploy" / "offline"
TEMPLATE_VERSION = (OFFLINE_ROOT / "VERSION").read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class BundleSpec:
    arch: str
    images: tuple[str, ...]
    template_dir: str
    compose_files: tuple[str, ...]
    health_kind: str
    project: str
    default_version: str


SPECS = {
    "platform-amd64": BundleSpec(
        "amd64",
        (
            "rknode-platform-api:{version}",
            "rknode-platform-web:{version}",
            "rknode-platform-media:{version}",
        ),
        "platform",
        ("compose.yaml",),
        "platform",
        "rknode-platform",
        "2026.08.26",
    ),
    "converter-rk3588-arm64": BundleSpec(
        "arm64",
        ("rknode-rk3588-node:{version}",),
        "rk3588",
        ("compose.converter.yaml", "compose.enrollment.converter.yaml"),
        "converter",
        "rknode-converter",
        "2026.08.26-business",
    ),
    "inference-rk3588-arm64": BundleSpec(
        "arm64",
        ("rknode-rk3588-node:{version}",),
        "rk3588",
        ("compose.inference.yaml", "compose.enrollment.inference.yaml"),
        "inference",
        "rknode-inference",
        "2026.08.26-business",
    ),
    "rk3588-node-arm64": BundleSpec(
        "arm64",
        ("rknode-rk3588-node:{version}",),
        "rk3588",
        ("compose.converter.yaml", "compose.inference.yaml", "compose.enrollment.yaml"),
        "rk3588",
        "rknode-rk3588",
        "2026.08.26-business",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_arch(value: str) -> str:
    return {"x86_64": "amd64", "aarch64": "arm64"}.get(value, value)


def inspect_image(image: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", image], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)[0]


def export_image(image: str, destination: Path) -> None:
    process = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("docker save did not expose stdout")
    with gzip.open(destination, "wb", compresslevel=1) as compressed:
        shutil.copyfileobj(process.stdout, compressed, length=1024 * 1024)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ["docker", "save", image])


def copy_template(source: Path, destination: Path, version: str) -> None:
    text = source.read_text(encoding="utf-8")
    version_pattern = re.compile(r"(?<=:)\d{4}\.\d{2}\.\d{2}(?:-business)?")
    text = version_pattern.sub(version, text)
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", choices=sorted(SPECS))
    parser.add_argument(
        "--version",
        default=None,
        help="override the role's release version",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "release" / "offline")
    parser.add_argument("--directory-only", action="store_true")
    parser.add_argument(
        "--allow-cross-arch",
        action="store_true",
        help=(
            "allow docker save on a host with a different architecture; image "
            "architecture and release labels are still validated"
        ),
    )
    args = parser.parse_args()

    spec = SPECS[args.bundle]
    version = args.version or spec.default_version
    host_arch = normalize_arch(platform.machine())
    if host_arch != spec.arch and not args.allow_cross_arch:
        raise SystemExit(
            f"ERROR: {args.bundle} must be packaged on {spec.arch}; current host is "
            f"{host_arch}. Use --allow-cross-arch only for an already-built target image."
        )

    image_names = tuple(image.format(version=version) for image in spec.images)
    inspected = []
    for image in image_names:
        metadata = inspect_image(image)
        image_arch = metadata["Architecture"]
        labels = metadata.get("Config", {}).get("Labels") or {}
        if image_arch != spec.arch:
            raise SystemExit(f"ERROR: {image} architecture is {image_arch}, expected {spec.arch}")
        if labels.get("org.opencontainers.image.version") != version:
            raise SystemExit(f"ERROR: {image} does not carry release version {version}")
        if labels.get("io.rknode.offline-ready") != "true":
            raise SystemExit(f"ERROR: {image} is not marked offline-ready")
        inspected.append(metadata)

    bundle_name = f"rknode-{args.bundle}-{version}"
    destination = args.output.resolve() / bundle_name
    if destination.exists():
        raise SystemExit(f"ERROR: output already exists: {destination}")
    destination.mkdir(parents=True)
    images_dir = destination / "images"
    images_dir.mkdir()

    template_root = OFFLINE_ROOT / spec.template_dir
    for compose_file in spec.compose_files:
        copy_template(template_root / compose_file, destination / compose_file, version)

    for script_name in ("load-images.sh", "deploy.sh", "verify.sh", "stop.sh"):
        target = destination / script_name
        shutil.copy2(OFFLINE_ROOT / "common" / script_name, target)
        target.chmod(0o755)
    target = destination / "read-manifest.py"
    shutil.copy2(OFFLINE_ROOT / "common" / "read-manifest.py", target)
    target.chmod(0o755)
    if spec.health_kind == "platform":
        target = destination / "configure-media-secrets.py"
        shutil.copy2(ROOT / "scripts" / "configure_media_secrets.py", target)
        target.chmod(0o755)

    if len(image_names) != len(inspected):
        raise RuntimeError("Docker image inspection returned an incomplete result")
    manifest_images = []
    for index, image in enumerate(image_names):
        metadata = inspected[index]
        archive_name = image.replace("/", "_").replace(":", "_") + ".tar.gz"
        archive = images_dir / archive_name
        print(f"Exporting {image} -> {archive}", flush=True)
        export_image(image, archive)
        labels = metadata.get("Config", {}).get("Labels") or {}
        manifest_images.append(
            {
                "tag": image,
                "id": metadata["Id"],
                "architecture": metadata["Architecture"],
                "os": metadata["Os"],
                "archive": f"images/{archive_name}",
                "archiveBytes": archive.stat().st_size,
                "archiveSha256": sha256(archive),
                "component": labels.get("io.rknode.component"),
                "variant": labels.get("io.rknode.variant"),
                "sourceRevision": labels.get("org.opencontainers.image.revision"),
            }
        )

    manifest = {
        "schemaVersion": 1,
        "bundle": bundle_name,
        "releaseVersion": version,
        "architecture": spec.arch,
        "healthKind": spec.health_kind,
        "composeProject": spec.project,
        "images": manifest_images,
        "composeFiles": list(spec.compose_files),
        "requiresNetworkDuringDeploy": False,
        "persistentDataIncluded": False,
        "secretsIncluded": False,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    checksum_paths = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (destination / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(destination)}\n" for path in checksum_paths),
        encoding="utf-8",
    )

    enrollment_commands = (
        "./deploy.sh --enroll\n./verify.sh --enroll"
        if spec.health_kind != "platform"
        else "./deploy.sh\n./verify.sh"
    )
    enrollment_note = (
        "For first-time node enrollment, create the documented secret file and use --enroll."
        if spec.health_kind != "platform"
        else "For platform deployment, edit the Compose anchors before starting."
    )
    steady_state_note = (
        "After the node is `enrolled + online`, run `./deploy.sh && ./verify.sh` without\n"
        "`--enroll`. Confirm the container no longer mounts the enrollment secret, then delete\n"
        "that one-time file and verify one more stop/deploy cycle."
        if spec.health_kind != "platform"
        else "The platform bundle has no enrollment overlay; keep the Compose anchors protected\n"
        "and run `./deploy.sh && ./verify.sh`."
    )
    readme = f"""# {bundle_name}

This is a no-network deployment bundle for `{spec.arch}`. It contains Docker images,
Compose configuration, checksums, and no credentials or persistent business data.

```bash
sha256sum -c SHA256SUMS
./load-images.sh
# Edit compose*.yaml: platform address, endpoint IDs, node names and ports.
# {enrollment_note}
{enrollment_commands}
```

{steady_state_note}

`deploy.sh` always uses `--pull never --no-build`. Runtime metadata comes from
`manifest.json`; no `.env` or `bundle.env` file is used. Stop containers while retaining
data with `./stop.sh`. Do not delete Compose volumes during an upgrade or rollback.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")

    # README is intentionally outside SHA256SUMS generation above; append its checksum last.
    with (destination / "SHA256SUMS").open("a", encoding="utf-8") as checksums:
        checksums.write(f"{sha256(destination / 'README.md')}  README.md\n")

    if args.directory_only:
        print(f"Bundle directory: {destination}")
    else:
        archive_path = args.output.resolve() / f"{bundle_name}.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(destination, arcname=bundle_name)
        archive_digest = sha256(archive_path)
        shutil.rmtree(destination)
        print(f"Bundle archive: {archive_path}")
        print(f"Bundle archive SHA256: {archive_digest}")


if __name__ == "__main__":
    main()
