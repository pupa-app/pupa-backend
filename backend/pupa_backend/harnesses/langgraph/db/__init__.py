"""Persistence for the deepagents harness — savers plus the `/db` router.

Public surface:
    open_persistence: async context manager yielding a ready
        (checkpointer, store) pair for the app lifespan.
    router:           FastAPI APIRouter over the checkpointer.
"""

from .connection import open_persistence
from .routes import router

__all__ = ["open_persistence", "router"]
