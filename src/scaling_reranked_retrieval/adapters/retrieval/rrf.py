from typing import List, Dict, Optional
from collections import defaultdict
from scaling_reranked_retrieval.adapters.retrieval.models import ObjectFromDB, RerankItem

def fuse_rrf(
    rankings: Dict[str, List[RerankItem]],
    top_k: int,
    rrf_k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> List[RerankItem]:
    """Fuse multiple rankings using Reciprocal Rank Fusion."""
    weights = weights or {}

    valid_rankings = {name: items for name, items in rankings.items() if items}

    if len(valid_rankings) == 1:
        return list(valid_rankings.values())[0][:top_k]

    if len(valid_rankings) == 0:
        return []

    scores: Dict[int, float] = {}
    for name, items in valid_rankings.items():
        w = weights.get(name, 1.0 / len(valid_rankings))
        for rank, item in enumerate(items):
            scores[item.index] = scores.get(item.index, 0.0) + w / (rrf_k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [RerankItem(index=idx, relevance_score=score) for idx, score in fused]

def reciprocal_rank_fusion(
    result_sets: List[List[ObjectFromDB]],
    k: int = 60,
    top_k: Optional[int] = None
) -> List[ObjectFromDB]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion."""
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, ObjectFromDB] = {}

    for result_set in result_sets:
        for rank, obj in enumerate(result_set, start=1):
            doc_id = obj.object_id

            rrf_scores[doc_id] += 1.0 / (rank + k)

            # Keep first occurrence of each doc.
            if doc_id not in doc_map:
                doc_map[doc_id] = obj

    sorted_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for new_rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], start=1):
        obj = doc_map[doc_id]
        results.append(ObjectFromDB(
            object_id=obj.object_id,
            content=obj.content,
            relevance_rank=new_rank,
            relevance_score=rrf_score,
            vector=obj.vector,
            source_query=obj.source_query
        ))
    
    return results