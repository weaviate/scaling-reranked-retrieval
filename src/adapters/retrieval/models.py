"""Data models shared across the retrieval adapters."""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal


class ObjectFromDB(BaseModel):
    object_id: str
    content: str
    relevance_rank: Optional[int] = None
    relevance_score: Optional[float] = None
    vector: Optional[list[float]] = None
    source_query: Optional[str] = None


class AgentRAGResponse:
    """Retrieval response envelope; consumers read ``.sources`` off it."""

    def __init__(self, final_answer: str = "", sources: List[ObjectFromDB] = None,
                 searches: Optional[List[str]] = None, aggregations: Optional[List] = None,
                 usage: Optional[Dict[str, Any]] = None, **kwargs):
        self.final_answer = final_answer
        self.sources = sources or []
        self.searches = searches
        self.aggregations = aggregations
        self.usage = usage or {}
        for key, value in kwargs.items():
            setattr(self, key, value)


class RerankerClient(BaseModel):
    name: Literal["cohere", "voyage", "zerank"]
    client: Any


class RerankItem(BaseModel):
    index: int
    relevance_score: float
