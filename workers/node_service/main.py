from __future__ import annotations

import logging
from dataclasses import replace

import uvicorn

from .app import create_node_app
from .config import NodeServiceSettings
from .enrollment import resolve_node_token
from .factory import build_inference, build_runtime


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = NodeServiceSettings.from_env()
    settings = replace(settings, token=resolve_node_token(settings))
    app = create_node_app(
        settings,
        runtime=build_runtime(settings),
        inference=build_inference(settings),
    )
    uvicorn.run(app, host=settings.host, port=settings.port, access_log=True)


if __name__ == "__main__":
    main()
