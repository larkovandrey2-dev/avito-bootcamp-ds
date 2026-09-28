from dataclasses import dataclass, field
from pathlib import Path


BASE_SOURCES = (
    "Lexical",
    "Giga",
    "Giga_local",
    "Description",
    "Title_params",
    "Giga_local_deep",
    "Description_local",
    "Giga_geo50",
)
ALL_SOURCES = BASE_SOURCES + ("Geo_redirect",)


@dataclass(frozen=True)
class PipelineConfig:
    root: Path
    data_dir: Path
    cache_dir: Path
    artifacts_dir: Path
    output: Path
    seed: int = 42
    top_k: int = 50
    embedding_model: str = "ai-sage/Giga-Embeddings-instruct-480M-0826"
    embedding_revision: str | None = "1763d603adac8057bd6482a708001b0ed2e0a903"
    query_prefix: str = (
        "Instruct: Given a Russian service search query, retrieve relevant "
        "service advertisements\nQuery: "
    )
    catboost_params: dict = field(
        default_factory=lambda: {
            "loss_function": "YetiRank",
            "depth": 6,
            "learning_rate": 0.05,
            # Frozen v3/v4 value from local experiments. The research log does not
            # say that 158 came from early stopping.
            "iterations": 158,
            "random_seed": 42,
        }
    )

    @classmethod
    def from_root(cls, root: Path, data_dir: Path, output: Path):
        return cls(
            root=root.resolve(),
            data_dir=data_dir.resolve(),
            cache_dir=(root / "cache").resolve(),
            artifacts_dir=(root / "artifacts").resolve(),
            output=output.resolve(),
        )
