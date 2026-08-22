from __future__ import annotations

from workers.common.config import WorkerConfig
from workers.common.runtime import WorkerRuntime

from .app import InferenceController
from .config import NodeServiceSettings


def build_runtime(settings: NodeServiceSettings) -> WorkerRuntime | None:
    if settings.kind == "inference":
        return None
    if not settings.token:
        raise ValueError("node Token must be resolved before building the Worker runtime")
    worker_config = WorkerConfig.from_env(token_override=settings.token)
    if (
        worker_config.name != settings.name
        or worker_config.kind != settings.kind
        or worker_config.accelerator != settings.accelerator
        or set(worker_config.capabilities) != set(settings.capabilities)
    ):
        raise ValueError("RKNODE_WORKER_* settings must match RKNODE_NODE_* settings")
    if settings.kind == "converter":
        from workers.converter.executor import ConversionExecutor

        return WorkerRuntime(worker_config, ConversionExecutor())
    from backend.platform_api.profiles import ModelProfileRegistry

    from workers.trainer.adapters import AdapterEnvironment
    from workers.trainer.executor import TrainingExecutor

    environment = AdapterEnvironment.from_env()
    profiles = ModelProfileRegistry(environment.project_root / "config/model_profiles.json")
    return WorkerRuntime(worker_config, TrainingExecutor(profiles, environment))


def build_inference(settings: NodeServiceSettings) -> InferenceController | None:
    if settings.kind != "inference":
        return None
    from workers.inference_agent.direct import DirectInferenceController

    return DirectInferenceController()
