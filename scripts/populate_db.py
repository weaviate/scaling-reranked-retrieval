#!/usr/bin/env python3
"""Populate Weaviate with one dataset's corpus (LIVE Weaviate writes).

Ingests a dataset from the DATASETS registry (e.g. bright/robotics, ~62k
docs) into the collection the experiment retrieves against (scaling_reranked_retrieval.config:
BrightRobotics_Default), using qab's own dataset loader and collection spec
so the schema matches what run_search_eval expects: a searchable "content"
text property, a filterable dataset_id, and a text2vec_weaviate vector over
Snowflake/snowflake-arctic-embed-l-v2.0 (scaling_reranked_retrieval.config.EMBEDDING_MODEL).
Embeddings are computed server-side by Weaviate Embeddings, so only
WEAVIATE_URL / WEAVIATE_API_KEY are required.

Safe by default: refuses to touch an existing collection unless --recreate.

Usage:
    uv run python scripts/populate_db.py --dataset robotics
    uv run python scripts/populate_db.py --dataset robotics --recreate
"""
from __future__ import annotations

import argparse
import os

from scaling_reranked_retrieval.adapters import qab

qab.setup()

from query_agent_benchmarking.internal.adapters.clients.weaviate_client import (  # noqa: E402
    get_weaviate_client,
)
from query_agent_benchmarking.internal.adapters.database.database_loader import (  # noqa: E402
    _batch_insert,
    _drop_and_create_collection,
    _load_documents,
    get_vector_config,
)
from query_agent_benchmarking.internal.adapters.database.database_registry import (  # noqa: E402
    resolve_spec,
)
from query_agent_benchmarking.internal.adapters.database.naming import (  # noqa: E402
    add_tag_to_name,
)

from scaling_reranked_retrieval.config import DATASETS, EMBEDDING_MODEL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        required=True,
        help="Dataset slug from scaling_reranked_retrieval.config.DATASETS to ingest.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the collection if it already exists before ingesting.",
    )
    args = parser.parse_args()

    missing = [v for v in ("WEAVIATE_URL", "WEAVIATE_API_KEY") if not os.getenv(v)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    cfg = DATASETS[args.dataset]
    spec = resolve_spec(cfg.qab_name)

    # The experiment addresses cfg.collection directly; qab's loader derives
    # names as spec name + "_Default" tag. Refuse to ingest under any other
    # name than the one the retrieval harness will query.
    expected = add_tag_to_name(spec.name_fn(cfg.qab_name), "Default")
    if expected != cfg.collection:
        raise SystemExit(
            f"Spec-derived collection name {expected!r} does not match "
            f"DATASETS[{args.dataset!r}].collection {cfg.collection!r}."
        )

    client = get_weaviate_client()
    try:
        if client.collections.exists(cfg.collection) and not args.recreate:
            raise SystemExit(
                f"Collection {cfg.collection!r} already exists. "
                "Pass --recreate to drop and re-ingest."
            )

        print(f"Loading corpus for {cfg.qab_name} ...")
        objects = _load_documents(cfg.qab_name)
        print(f"Loaded {len(objects)} documents.")

        print(f"Creating collection {cfg.collection!r} ({EMBEDDING_MODEL}) ...")
        _drop_and_create_collection(
            client,
            cfg.collection,
            properties=spec.properties,
            vector_config=get_vector_config(EMBEDDING_MODEL),
            recreate=args.recreate,
            multi_tenancy_config=spec.multi_tenancy_config,
        )

        inserted = _batch_insert(
            client,
            collection=cfg.collection,
            items=objects,
            item_to_props=spec.item_to_props,
            tenant_id_field=spec.tenant_id_field,
        )

        total = (
            client.collections.get(cfg.collection)
            .aggregate.over_all(total_count=True)
            .total_count
        )
        print(f"Inserted {inserted} objects; collection reports {total} objects.")
        if total != len(objects):
            raise SystemExit(
                f"Object count mismatch: corpus has {len(objects)}, "
                f"collection has {total}. Re-run with --recreate."
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
