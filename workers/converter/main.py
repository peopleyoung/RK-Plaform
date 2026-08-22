from __future__ import annotations

import logging

from workers.common.config import WorkerConfig
from workers.common.runtime import WorkerRuntime
from workers.converter.executor import ConversionExecutor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = WorkerConfig.from_env()
    if config.kind != "converter" or config.accelerator != "rk3588":
        raise ValueError("Converter requires kind=converter and accelerator=rk3588")
    WorkerRuntime(config, ConversionExecutor()).run_forever()


if __name__ == "__main__":
    main()
