"""Open the LangGraph checkpointer + store for the lifetime of the app.

`open_persistence` is the whole interface: it opens the underlying async
connections, runs each saver's `setup()`, and yields the pair for the caller
to hand to the graph and the `/db` routes. The URL *scheme* picks the backend
(see :mod:`pupa_backend.db_config`); a `None` URL yields the in-memory saver,
which is what the test suite uses.

Routes talk to LangGraph's own checkpointer API directly — there is no wrapper
layer here to keep in sync with it.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from pupa_backend.db_config import (
    SCHEME_POSTGRES,
    SCHEME_SQLITE,
    scheme_of,
    sqlite_path,
)

logger = logging.getLogger(__name__)

_POOL_KWARGS = {"autocommit": True, "prepare_threshold": 0}


def thread_config(thread_id: str, checkpoint_id: str | None = None) -> dict:
    """The `configurable` dict LangGraph's checkpointer API takes."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        },
    }


@asynccontextmanager
async def open_persistence(
    checkpointer_url: str | None,
    store_url: str | None,
) -> AsyncIterator[tuple[BaseCheckpointSaver, BaseStore]]:
    """Yield a ready `(checkpointer, store)` pair for the app lifespan."""
    async with _checkpointer(checkpointer_url) as checkpointer:
        async with _store(store_url) as store:
            logger.info(
                "persistence ready: checkpointer=%s, store=%s",
                type(checkpointer).__name__,
                type(store).__name__,
            )
            yield checkpointer, store


@asynccontextmanager
async def _checkpointer(url: str | None) -> AsyncIterator[BaseCheckpointSaver]:
    if url is None:
        yield MemorySaver()
        return

    scheme = scheme_of(url)

    if scheme == SCHEME_SQLITE:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(str(sqlite_path(url))) as saver:
            await saver.setup()
            yield saver
        return

    if scheme == SCHEME_POSTGRES:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        async with AsyncConnectionPool(conninfo=url, kwargs=_POOL_KWARGS, open=False) as pool:
            await pool.open()
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            yield saver
        return

    raise ValueError(f"Unsupported checkpointer scheme: {scheme!r}")


@asynccontextmanager
async def _store(url: str | None) -> AsyncIterator[BaseStore]:
    if url is None:
        yield InMemoryStore()
        return

    scheme = scheme_of(url)

    if scheme == SCHEME_SQLITE:
        # LangGraph ships no async SQLite store, so the store is ephemeral on a
        # SQLite deploy even though the checkpointer is not.
        logger.warning(
            "LangGraph has no AsyncSqliteStore; falling back to InMemoryStore. "
            "Set DATABASE_URL to a postgresql:// URL for a persistent store."
        )
        yield InMemoryStore()
        return

    if scheme == SCHEME_POSTGRES:
        from langgraph.store.postgres.aio import AsyncPostgresStore
        from psycopg_pool import AsyncConnectionPool

        async with AsyncConnectionPool(conninfo=url, kwargs=_POOL_KWARGS, open=False) as pool:
            await pool.open()
            store = AsyncPostgresStore(pool)
            await store.setup()
            yield store
        return

    raise ValueError(f"Unsupported store scheme: {scheme!r}")
