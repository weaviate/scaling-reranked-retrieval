"""Experiment-wide configuration: dataset registry, model ids, constants.

Importable with zero side effects and zero provider/qab imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


@dataclass(frozen=True)
class DatasetConfig:
    """Per-dataset wiring for one --dataset slug.

    extra_base_metrics: metric specs appended on top of qab's profile defaults
    and the recall cutoffs in build_extra_metrics (adds nDCG@10 for IRPAPERS).
    """

    qab_name: str
    collection: str
    target_property: str = "content"
    results_subdir: str = ""
    extra_base_metrics: tuple[dict, ...] = ()


# Collections must already be ingested into Weaviate.
DATASETS: dict[str, DatasetConfig] = {
    "biology":       DatasetConfig("bright/biology",       "BrightBiology_Default",     results_subdir="bright_biology"),
    "earth_science": DatasetConfig("bright/earth_science", "BrightEarthScience_Default", results_subdir="bright_earth_science"),
    "economics":     DatasetConfig("bright/economics",     "BrightEconomics_Default",   results_subdir="bright_economics"),
    "psychology":    DatasetConfig("bright/psychology",    "BrightPsychology_Default",  results_subdir="bright_psychology"),
    "robotics":      DatasetConfig("bright/robotics",      "BrightRobotics_Default",    results_subdir="bright_robotics"),
    # IRPAPERS: qab's profile is recall@1/5/20 only, so nDCG@10 is added via
    # extra_base_metrics for reporting parity.
    "irpapers_text": DatasetConfig(
        "irpapers-text-only",
        "IRPapersTextOnly_Default",
        target_property="content",
        results_subdir="irpapers_text",
        extra_base_metrics=({"name": "nDCG", "params": {"k": 10}},),
    ),
}

# BRIGHT subsets in the paper's canonical order.
BRIGHT_SUBSETS = ["biology", "earth_science", "economics", "psychology", "robotics"]

DEFAULT_RETRIEVED_K = 100
DEFAULT_RERANKED_K = 20
RANDOM_SEED = 42
RRF_K = 60

# The k the score caches were collected at, and the derive sweep over it.
CACHE_K = 2000
K_VALUES = (100, 200, 500, 1000, 2000)

# Per-provider reranker model overrides (pin the model version recorded in the experiment).
MODEL_OVERRIDES = {
    "cohere": "rerank-v4.0-pro",
    "voyage": "rerank-2.5",
    "zerank": "zerank-2",
}

PROVIDERS = ("cohere", "voyage", "zerank")
# Single-letter Venn labels, in PROVIDERS order.
PROV_LETTER = {"cohere": "c", "voyage": "v", "zerank": "z"}

EMBEDDING_MODEL = "weaviate/Snowflake/snowflake-arctic-embed-l-v2.0"


def get_results_dir(dataset_slug: str) -> Path:
    return RESULTS_DIR / DATASETS[dataset_slug].results_subdir
