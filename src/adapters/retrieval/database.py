"""SearchBackend abstraction.

Retrievers in this library are algorithm definitions; the choice of vector DB
is an orthogonal concern. ``SearchBackend`` is the narrow interface every
retriever depends on. ``WeaviateBackend`` is the only built-in implementation
today, but adding a new one means writing one class — not editing every
retriever.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from src.adapters.retrieval.models import ObjectFromDB


@runtime_checkable
class SearchBackend(Protocol):
    """Synchronous search interface every retriever depends on."""

    def search(
        self,
        query: str,
        collection_name: str,
        target_property_name: str,
        retrieved_k: int,
        *,
        return_property_name: Optional[str] = None,
        return_vector: bool = False,
        return_score: bool = False,
        search_type: str = "hybrid",
        hybrid_alpha: Optional[float] = None,
    ) -> list[ObjectFromDB]:
        ...


@runtime_checkable
class AsyncSearchBackend(Protocol):
    """Asynchronous mirror of :class:`SearchBackend`.

    Uses ``asearch`` (not ``search``) so a single object can satisfy both
    protocols without an awaitable / non-awaitable name collision.
    """

    async def asearch(
        self,
        query: str,
        collection_name: str,
        target_property_name: str,
        retrieved_k: int,
        *,
        return_property_name: Optional[str] = None,
        return_vector: bool = False,
        return_score: bool = False,
        search_type: str = "hybrid",
        hybrid_alpha: Optional[float] = None,
    ) -> list[ObjectFromDB]:
        ...


class WeaviateBackend:
    """SearchBackend backed by a Weaviate client.

    The client is held on the backend instance, so the per-call kwargs that
    used to leak through every retriever (``weaviate_client=...``) are no
    longer needed. Sync and async clients can be combined on one backend
    instance so a single object satisfies both Protocols.
    """

    def __init__(
        self,
        sync_client=None,
        async_client=None,
    ) -> None:
        self._sync_client = sync_client
        self._async_client = async_client

    @property
    def sync_client(self):
        if self._sync_client is None:
            raise RuntimeError(
                "WeaviateBackend was constructed without a sync client; "
                "pass sync_client=... or use the async path."
            )
        return self._sync_client

    @property
    def async_client(self):
        if self._async_client is None:
            raise RuntimeError(
                "WeaviateBackend was constructed without an async client; "
                "pass async_client=... or use the sync path."
            )
        return self._async_client

    def search(
        self,
        query: str,
        collection_name: str,
        target_property_name: str,
        retrieved_k: int,
        *,
        return_property_name: Optional[str] = None,
        return_vector: bool = False,
        return_score: bool = False,
        search_type: str = "hybrid",
        hybrid_alpha: Optional[float] = None,
    ) -> list[ObjectFromDB]:
        from src.adapters.retrieval.weaviate_database import weaviate_search_tool

        return weaviate_search_tool(
            query=query,
            collection_name=collection_name,
            target_property_name=target_property_name,
            weaviate_client=self.sync_client,
            return_property_name=return_property_name,
            retrieved_k=retrieved_k,
            return_vector=return_vector,
            return_score=return_score,
            search_type=search_type,
            hybrid_alpha=hybrid_alpha,
        )

    async def asearch(
        self,
        query: str,
        collection_name: str,
        target_property_name: str,
        retrieved_k: int,
        *,
        return_property_name: Optional[str] = None,
        return_vector: bool = False,
        return_score: bool = False,
        search_type: str = "hybrid",
        hybrid_alpha: Optional[float] = None,
    ) -> list[ObjectFromDB]:
        from src.adapters.retrieval.weaviate_database import (
            async_weaviate_search_tool,
        )

        return await async_weaviate_search_tool(
            query=query,
            collection_name=collection_name,
            target_property_name=target_property_name,
            weaviate_async_client=self.async_client,
            return_property_name=return_property_name,
            retrieved_k=retrieved_k,
            return_score=return_score,
            return_vector=return_vector,
            search_type=search_type,
            hybrid_alpha=hybrid_alpha,
        )


__all__ = ["SearchBackend", "AsyncSearchBackend", "WeaviateBackend"]
