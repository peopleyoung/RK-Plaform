from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from backend.platform_api.contracts import TrainingJobCreate
from backend.platform_api.profiles import ModelProfileRegistry

from workers.trainer.dataset import write_resolved_yolo_config


class AdapterConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandStep:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]

    def __post_init__(self) -> None:
        if not self.argv or any(not item for item in self.argv):
            raise ValueError("Command arguments must be non-empty strings")


@dataclass(frozen=True)
class AdapterPlan:
    steps: tuple[CommandStep, ...]
    onnx_source: Path
    checkpoint_source: Path | None = None
    auxiliary_artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True)
class TrainingTask:
    job_id: str
    request: TrainingJobCreate
    dataset_dir: Path
    output_dir: Path
    labels: tuple[str, ...]


@dataclass(frozen=True)
class AdapterEnvironment:
    python: str
    project_root: Path
    framework_roots: dict[str, Path]

    @classmethod
    def from_env(cls) -> AdapterEnvironment:
        root = Path(__file__).parents[2].resolve()
        names = {
            "yolov5": "RKNODE_YOLOV5_ROOT",
            "yolov6": "RKNODE_YOLOV6_ROOT",
            "yolov7": "RKNODE_YOLOV7_ROOT",
            "yolov8": "RKNODE_YOLOV8_ROOT",
            "yolov10": "RKNODE_YOLOV10_ROOT",
            "yolo11": "RKNODE_YOLO11_ROOT",
            "paddleocr": "RKNODE_PADDLEOCR_ROOT",
        }
        paths = {
            name: Path(value).resolve()
            for name, env_name in names.items()
            if (value := os.getenv(env_name))
        }
        return cls(
            python=os.getenv("RKNODE_TRAINER_PYTHON", sys.executable),
            project_root=root,
            framework_roots=paths,
        )

    def require_root(self, name: str, entrypoint: str) -> Path:
        root = self.framework_roots.get(name)
        if root is None:
            raise AdapterConfigurationError(
                f"Training framework '{name}' is not configured; set RKNODE_{name.upper()}_ROOT"
            )
        expected = root / entrypoint
        if not expected.is_file():
            raise AdapterConfigurationError(f"Framework entrypoint does not exist: {expected}")
        return root


class TrainingAdapter(ABC):
    profile_id: str

    def __init__(self, profiles: ModelProfileRegistry, environment: AdapterEnvironment) -> None:
        self.profiles = profiles
        self.environment = environment

    @abstractmethod
    def plan(self, task: TrainingTask) -> AdapterPlan:
        raise NotImplementedError

    @staticmethod
    def dataset_file(dataset_dir: Path, relative_name: str) -> Path:
        matches = list(dataset_dir.rglob(relative_name))
        if len(matches) != 1:
            raise AdapterConfigurationError(
                f"Dataset must contain exactly one '{relative_name}', found {len(matches)}"
            )
        return matches[0]


class YoloAdapter(TrainingAdapter):
    profile_id = "yolo-detect"

    def plan(self, task: TrainingTask) -> AdapterPlan:
        contract = self.profiles.variant_contract(self.profile_id, task.request.variant)
        task.output_dir.mkdir(parents=True, exist_ok=True)
        data_yaml = write_resolved_yolo_config(
            self.dataset_file(task.dataset_dir, "data.yaml"),
            task.dataset_dir,
            task.output_dir / "data-resolved.yaml",
        )
        variant = task.request.variant
        if variant.startswith("yolov5"):
            return self._yolov5(task, data_yaml)
        if variant.startswith("yolov6"):
            return self._yolov6(task, data_yaml)
        if variant.startswith("yolov7"):
            return self._yolov7(task, data_yaml)
        if variant.startswith("yolov8"):
            return self._ultralytics(task, data_yaml, "yolov8", contract.opset, "rknn")
        if variant.startswith("yolov10"):
            return self._ultralytics(task, data_yaml, "yolov10", contract.opset, "rknn")
        if variant.startswith("yolo11"):
            return self._ultralytics(task, data_yaml, "yolo11", contract.opset, "rknn")
        raise AdapterConfigurationError(f"No YOLO adapter for variant '{variant}'")

    def _yolov5(self, task: TrainingTask, data_yaml: Path) -> AdapterPlan:
        root = self.environment.require_root("yolov5", "train.py")
        run_dir = task.output_dir / "train"
        checkpoint = run_dir / "weights" / "best.pt"
        train_arguments = [
            "--data",
            str(data_yaml),
            "--epochs",
            str(task.request.hyperparameters.epochs),
            "--batch-size",
            str(task.request.hyperparameters.batch_size),
            "--imgsz",
            str(max(task.request.resolution.width, task.request.resolution.height)),
            "--device",
            self._device(task),
            "--project",
            str(task.output_dir),
            "--name",
            "train",
            "--exist-ok",
            "--seed",
            str(task.request.hyperparameters.seed),
        ]
        if task.request.hyperparameters.optimizer != "auto":
            train_arguments.extend(("--optimizer", task.request.hyperparameters.optimizer))
        prepare_steps, hyperparameters = self._legacy_hyperparameters(
            task,
            root / "data/hyps/hyp.scratch-low.yaml",
            "yaml",
            include_optimizer=False,
        )
        if hyperparameters is not None:
            train_arguments.extend(("--hyp", str(hyperparameters)))
        if task.request.hyperparameters.pretrained:
            train_arguments.extend(("--weights", f"{task.request.variant}.pt"))
        else:
            train_arguments.append("--weights=")
            train_arguments.extend(("--cfg", str(root / "models/yolov5s.yaml")))
        train = self._step("train", root, "train.py", *train_arguments)
        export = self._step(
            "export",
            root,
            "export.py",
            "--rknpu",
            "--weights",
            str(checkpoint),
            "--imgsz",
            str(task.request.resolution.height),
            str(task.request.resolution.width),
            "--batch-size",
            "1",
            "--opset",
            "12",
        )
        return AdapterPlan(
            (*prepare_steps, train, export),
            checkpoint.with_suffix(".onnx"),
            checkpoint,
            (checkpoint.parent / "RK_anchors.txt",),
        )

    def _yolov6(self, task: TrainingTask, data_yaml: Path) -> AdapterPlan:
        root = self.environment.require_root("yolov6", "tools/train.py")
        run_dir = task.output_dir / "train"
        checkpoint = run_dir / "weights" / "best_ckpt.pt"
        source_config = root / (
            "configs/yolov6s_finetune.py"
            if task.request.hyperparameters.pretrained
            else "configs/yolov6s.py"
        )
        prepare_steps, resolved_config = self._legacy_hyperparameters(task, source_config, "python")
        train = self._step(
            "train",
            root,
            "tools/train.py",
            "--data-path",
            str(data_yaml),
            "--conf-file",
            str(resolved_config or source_config),
            "--epochs",
            str(task.request.hyperparameters.epochs),
            "--batch-size",
            str(task.request.hyperparameters.batch_size),
            "--img-size",
            str(max(task.request.resolution.width, task.request.resolution.height)),
            "--device",
            self._device(task),
            "--output-dir",
            str(task.output_dir),
            "--name",
            "train",
        )
        export = self._step(
            "export",
            root,
            "deploy/RKNN/export_onnx_for_rknn.py",
            "--weights",
            str(checkpoint),
            "--img-size",
            str(task.request.resolution.height),
            str(task.request.resolution.width),
            "--batch-size",
            "1",
            "--device",
            self._device(task),
        )
        return AdapterPlan(
            (*prepare_steps, train, export), checkpoint.with_suffix(".onnx"), checkpoint
        )

    def _yolov7(self, task: TrainingTask, data_yaml: Path) -> AdapterPlan:
        root = self.environment.require_root("yolov7", "train.py")
        run_dir = task.output_dir / "train"
        checkpoint = run_dir / "weights" / "best.pt"
        train_size = max(task.request.resolution.width, task.request.resolution.height)
        weights = (
            ("--weights", "yolov7-tiny.pt")
            if task.request.hyperparameters.pretrained
            else ("--weights=",)
        )
        prepare_steps, hyperparameters = self._legacy_hyperparameters(
            task, root / "data/hyp.scratch.p5.yaml", "yaml"
        )
        extra_arguments = ("--hyp", str(hyperparameters)) if hyperparameters is not None else ()
        train = self._step(
            "train",
            root,
            "train.py",
            "--data",
            str(data_yaml),
            *weights,
            *extra_arguments,
            "--cfg",
            str(root / "cfg/training/yolov7-tiny.yaml"),
            "--epochs",
            str(task.request.hyperparameters.epochs),
            "--batch-size",
            str(task.request.hyperparameters.batch_size),
            "--img-size",
            str(train_size),
            str(train_size),
            "--device",
            self._device(task),
            "--project",
            str(task.output_dir),
            "--name",
            "train",
            "--exist-ok",
        )
        export = self._step(
            "export",
            root,
            "export.py",
            "--rknpu",
            "--weights",
            str(checkpoint),
            "--img-size",
            str(task.request.resolution.height),
            str(task.request.resolution.width),
            "--batch-size",
            "1",
        )
        return AdapterPlan(
            (*prepare_steps, train, export),
            checkpoint.with_suffix(".onnx"),
            checkpoint,
            (checkpoint.parent / "RK_anchors.txt",),
        )

    def _ultralytics(
        self,
        task: TrainingTask,
        data_yaml: Path,
        framework: str,
        opset: int,
        export_format: str,
    ) -> AdapterPlan:
        root = self.environment.require_root(framework, "ultralytics/__init__.py")
        script = self.environment.project_root / "workers/trainer/scripts/yolo_ultralytics.py"
        model_path = task.output_dir / "model.onnx"
        checkpoint = task.output_dir / "best.pt"
        arguments = [
            self.environment.python,
            str(script),
            "--repo-root",
            str(root),
            "--variant",
            task.request.variant,
            "--data",
            str(data_yaml),
            "--output-model",
            str(model_path),
            "--output-checkpoint",
            str(checkpoint),
            "--width",
            str(task.request.resolution.width),
            "--height",
            str(task.request.resolution.height),
            "--epochs",
            str(task.request.hyperparameters.epochs),
            "--batch-size",
            str(task.request.hyperparameters.batch_size),
            "--device",
            self._device(task),
            "--opset",
            str(opset),
            "--export-format",
            export_format,
            "--optimizer",
            task.request.hyperparameters.optimizer,
            "--seed",
            str(task.request.hyperparameters.seed),
        ]
        if task.request.hyperparameters.learning_rate is not None:
            arguments.extend(("--learning-rate", str(task.request.hyperparameters.learning_rate)))
        if not task.request.hyperparameters.pretrained:
            arguments.append("--no-pretrained")
        step = CommandStep(
            name="train_export",
            argv=tuple(arguments),
            cwd=root,
            env={},
        )
        return AdapterPlan((step,), model_path, checkpoint)

    def _legacy_hyperparameters(
        self,
        task: TrainingTask,
        source: Path,
        config_format: str,
        *,
        include_optimizer: bool = True,
    ) -> tuple[tuple[CommandStep, ...], Path | None]:
        hyperparameters = task.request.hyperparameters
        requested_optimizer = hyperparameters.optimizer if include_optimizer else "auto"
        if hyperparameters.learning_rate is None and requested_optimizer == "auto":
            return (), None
        suffix = "py" if config_format == "python" else "yaml"
        output = task.output_dir / f"legacy-hyperparameters.{suffix}"
        script = (
            self.environment.project_root / "workers/trainer/scripts/prepare_yolo_legacy_config.py"
        )
        arguments = [
            self.environment.python,
            str(script),
            "--source",
            str(source),
            "--output",
            str(output),
            "--format",
            config_format,
            "--optimizer",
            requested_optimizer,
        ]
        if hyperparameters.learning_rate is not None:
            arguments.extend(("--learning-rate", str(hyperparameters.learning_rate)))
        return (
            CommandStep(
                "prepare_hyperparameters", tuple(arguments), self.environment.project_root, {}
            ),
        ), output

    def _step(self, name: str, root: Path, entrypoint: str, *arguments: str) -> CommandStep:
        return CommandStep(
            name=name,
            argv=(self.environment.python, str(root / entrypoint), *arguments),
            cwd=root,
            env={},
        )

    @staticmethod
    def _device(task: TrainingTask) -> str:
        return "0" if task.request.accelerator == "cuda" else "cpu"


class DeepLabAdapter(TrainingAdapter):
    profile_id = "deeplabv3plus"

    def plan(self, task: TrainingTask) -> AdapterPlan:
        script = self.environment.project_root / "workers/trainer/scripts/train_deeplab.py"
        model_path = task.output_dir / "model.onnx"
        checkpoint = task.output_dir / "best.pt"
        arguments = [
            self.environment.python,
            str(script),
            "--dataset",
            str(task.dataset_dir),
            "--variant",
            task.request.variant,
            "--classes",
            str(len(task.labels)),
            "--output-model",
            str(model_path),
            "--output-checkpoint",
            str(checkpoint),
            "--width",
            str(task.request.resolution.width),
            "--height",
            str(task.request.resolution.height),
            "--epochs",
            str(task.request.hyperparameters.epochs),
            "--batch-size",
            str(task.request.hyperparameters.batch_size),
            "--device",
            "cuda" if task.request.accelerator == "cuda" else "cpu",
            "--optimizer",
            task.request.hyperparameters.optimizer,
            "--seed",
            str(task.request.hyperparameters.seed),
        ]
        if task.request.hyperparameters.learning_rate is not None:
            arguments.extend(("--learning-rate", str(task.request.hyperparameters.learning_rate)))
        if not task.request.hyperparameters.pretrained:
            arguments.append("--no-pretrained")
        step = CommandStep(
            name="train_export",
            argv=tuple(arguments),
            cwd=self.environment.project_root,
            env={},
        )
        return AdapterPlan((step,), model_path, checkpoint)


class PpocrAdapter(TrainingAdapter):
    def __init__(
        self,
        profiles: ModelProfileRegistry,
        environment: AdapterEnvironment,
        profile_id: str,
    ) -> None:
        super().__init__(profiles, environment)
        self.profile_id = profile_id

    def plan(self, task: TrainingTask) -> AdapterPlan:
        root = self.environment.require_root("paddleocr", "tools/train.py")
        config = self._config(root, task.request.variant)
        train_label = self.dataset_file(task.dataset_dir, "train.txt")
        val_label = self.dataset_file(task.dataset_dir, "val.txt")
        train_dir = task.output_dir / "train"
        inference_dir = task.output_dir / "inference"
        export_checkpoint = task.output_dir / "export-checkpoint"
        resolved_config = task.output_dir / "ppocr-resolved.yml"
        dynamic_model_path = task.output_dir / "dynamic.onnx"
        model_path = task.output_dir / "model.onnx"
        prepare_script = (
            self.environment.project_root / "workers/trainer/scripts/prepare_ppocr_config.py"
        )
        prepare_arguments = [
            self.environment.python,
            str(prepare_script),
            "--source",
            str(config),
            "--output",
            str(resolved_config),
            "--profile-id",
            self.profile_id,
            "--dataset",
            str(task.dataset_dir),
            "--train-label",
            str(train_label),
            "--val-label",
            str(val_label),
            "--save-dir",
            str(train_dir),
            "--width",
            str(task.request.resolution.width),
            "--height",
            str(task.request.resolution.height),
            "--epochs",
            str(task.request.hyperparameters.epochs),
            "--batch-size",
            str(task.request.hyperparameters.batch_size),
            "--device",
            "cuda" if task.request.accelerator == "cuda" else "cpu",
            "--optimizer",
            task.request.hyperparameters.optimizer,
            "--seed",
            str(task.request.hyperparameters.seed),
        ]
        if task.request.hyperparameters.learning_rate is not None:
            prepare_arguments.extend(
                ("--learning-rate", str(task.request.hyperparameters.learning_rate))
            )
        if not task.request.hyperparameters.pretrained:
            prepare_arguments.append("--no-pretrained")
        prepare = CommandStep(
            "prepare_config",
            tuple(prepare_arguments),
            self.environment.project_root,
            {},
        )
        train = CommandStep(
            "train",
            (
                self.environment.python,
                str(root / "tools/train.py"),
                "-c",
                str(resolved_config),
            ),
            root,
            {},
        )
        select_checkpoint = CommandStep(
            "select_checkpoint",
            (
                self.environment.python,
                str(
                    self.environment.project_root
                    / "workers/trainer/scripts/select_ppocr_checkpoint.py"
                ),
                "--train-dir",
                str(train_dir),
                "--output-prefix",
                str(export_checkpoint),
            ),
            self.environment.project_root,
            {},
        )
        export = CommandStep(
            "export",
            (
                self.environment.python,
                str(root / "tools/export_model.py"),
                "-c",
                str(resolved_config),
                "-o",
                f"Global.pretrained_model={export_checkpoint}",
                f"Global.save_inference_dir={inference_dir}",
            ),
            root,
            {},
        )
        static_shape = (
            f'{{"x":[1,3,{task.request.resolution.height},{task.request.resolution.width}]}}'
        )
        convert = CommandStep(
            "export_onnx",
            (
                "paddle2onnx",
                "--model_dir",
                str(inference_dir),
                "--model_filename",
                "inference.pdmodel",
                "--params_filename",
                "inference.pdiparams",
                "--save_file",
                str(dynamic_model_path),
                "--opset_version",
                "14",
                "--enable_onnx_checker",
                "True",
            ),
            root,
            {},
        )
        optimize = CommandStep(
            "fix_static_shape",
            (
                self.environment.python,
                "-m",
                "paddle2onnx.optimize",
                "--input_model",
                str(dynamic_model_path),
                "--output_model",
                str(model_path),
                "--input_shape_dict",
                static_shape,
            ),
            root,
            {},
        )
        return AdapterPlan(
            (prepare, train, select_checkpoint, export, convert, optimize),
            model_path,
            export_checkpoint.with_suffix(".pdparams"),
        )

    def _config(self, root: Path, variant: str) -> Path:
        env_name = f"RKNODE_{variant.upper()}_CONFIG"
        override = os.getenv(env_name)
        if override:
            config = Path(override).resolve()
        else:
            relative = {
                "ppocrv3_det": "configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml",
                "ppocrv4_det": "configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml",
                "ppocrv3_rec": "configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml",
                "ppocrv4_rec": "configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml",
            }.get(variant)
            if relative is None:
                raise AdapterConfigurationError(f"No PPOCR config mapping for '{variant}'")
            config = root / relative
        if not config.is_file():
            raise AdapterConfigurationError(
                f"PPOCR config does not exist: {config}; set {env_name} for this checkout"
            )
        return config


class AdapterRegistry:
    def __init__(self, profiles: ModelProfileRegistry, environment: AdapterEnvironment) -> None:
        self._adapters: dict[str, TrainingAdapter] = {
            "yolo-detect": YoloAdapter(profiles, environment),
            "deeplabv3plus": DeepLabAdapter(profiles, environment),
            "ppocr-det": PpocrAdapter(profiles, environment, "ppocr-det"),
            "ppocr-rec": PpocrAdapter(profiles, environment, "ppocr-rec"),
        }

    def get(self, profile_id: str) -> TrainingAdapter:
        try:
            return self._adapters[profile_id]
        except KeyError as error:
            raise AdapterConfigurationError(
                f"No training adapter for profile '{profile_id}'"
            ) from error
