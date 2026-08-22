from __future__ import annotations

import logging

from .agent import build_agent_from_env


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    build_agent_from_env().run()


if __name__ == "__main__":
    main()
