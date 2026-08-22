from __future__ import annotations

from dataclasses import dataclass

from .database import Database
from .media_secrets import MediaSecretStore
from .node_secrets import NodeSecretStore
from .profiles import ModelProfileRegistry
from .settings import Settings
from .state_machine import JobStateMachine
from .storage import FileStorage


@dataclass(frozen=True)
class AppContext:
    settings: Settings
    database: Database
    storage: FileStorage
    profiles: ModelProfileRegistry
    jobs: JobStateMachine
    node_secrets: NodeSecretStore
    media_secrets: MediaSecretStore
