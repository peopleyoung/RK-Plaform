from __future__ import annotations

import logging

from backend.platform_api.profiles import ModelProfileRegistry

from workers.common.config import WorkerConfig
from workers.common.runtime import WorkerRuntime
from workers.trainer.adapters import AdapterEnvironment
from workers.trainer.executor import TrainingExecutor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = WorkerConfig.from_env()
    if config.kind != "trainer" or config.accelerator not in {"cpu", "cuda"}:
        raise ValueError("Trainer requires kind=trainer and accelerator=cpu|cuda")
    environment = AdapterEnvironment.from_env()
    profiles_path = environment.project_root / "config/model_profiles.json"
    executor = TrainingExecutor(
        ModelProfileRegistry(profiles_path),
        environment,
    )
    WorkerRuntime(config, executor).run_forever()


if __name__ == "__main__":
    main()
