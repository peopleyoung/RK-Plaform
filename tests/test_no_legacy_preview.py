from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (
    ROOT / "backend",
    ROOT / "workers" / "inference_agent",
    ROOT / "third_party" / "nv_video_pipeline" / "src",
    ROOT / "deploy",
)
CURRENT_OPERATOR_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "deployment-guide.md",
    ROOT / "docs" / "operations-guide.md",
    ROOT / "third_party" / "nv_video_pipeline" / "CMakeLists.txt",
    ROOT / "third_party" / "nv_video_pipeline" / "RK3588_RUNTIME.md",
)
FORBIDDEN = (
    "Preview" + "OutputNode",
    "RKNODE_" + "PREVIEW_",
    "preview" + ".jpg",
    "preview" + ".mjpeg",
    "upload" + "_preview",
    "Inference" + "PreviewStore",
    "useInference" + "PreviewStream",
)
TEXT_SUFFIXES = {
    ".cpp",
    ".h",
    ".md",
    ".py",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}


def _scanned_files() -> list[Path]:
    files: set[Path] = set()
    for root in RUNTIME_ROOTS:
        if not root.exists():
            continue
        files.update(
            path
            for path in root.rglob("*")
            if path.is_file()
            and (path.suffix in TEXT_SUFFIXES or path.name.startswith(".env"))
        )
    files.update(path for path in CURRENT_OPERATOR_DOCS if path.exists())
    return sorted(files)


def test_legacy_preview_chain_is_absent_from_current_runtime_and_docs() -> None:
    violations: list[str] = []
    for path in _scanned_files():
        content = path.read_text(encoding="utf-8", errors="replace")
        for symbol in FORBIDDEN:
            if symbol in content:
                violations.append(f"{path.relative_to(ROOT)}: {symbol}")

    assert violations == []
