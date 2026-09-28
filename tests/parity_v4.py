"""Небольшая проверка, что рефакторинг не поменял frozen v4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ALL_SOURCES
from src.features import CandidateFeatureBuilder, FieldInteractionBuilder, FINAL_FEATURE_NAMES
from src.features import build_final_matrix
from src.retrieval import load_source
from src.utils import normalize_text


GOLDEN_PATH = ROOT / "tests" / "golden" / "v4_sample.npz"
QUERY_INDICES = np.array([0, 17], dtype=np.int32)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def load_frames():
    local_items = pd.read_pickle(
        ROOT / "cache" / "reranker_frozen" / "fixed_item_corpus_ordered.pkl"
    )
    benchmark_items = pd.read_parquet(ROOT / "data" / "benchmark_items.parquet")
    frames = {
        "train": pd.read_pickle(ROOT / "cache" / "learned_reranker" / "reranker_train_queries.pkl"),
        "internal": pd.read_pickle(ROOT / "cache" / "research_v3" / "reranker_internal_val.pkl"),
        "dev": pd.read_pickle(ROOT / "cache" / "reranker_frozen" / "development_examples.pkl"),
        "benchmark": pd.read_parquet(ROOT / "data" / "benchmark_queries.parquet"),
    }
    return local_items, benchmark_items, frames


def local_sources(item_index: pd.Index):
    learned = ROOT / "cache" / "learned_reranker"
    sprint = ROOT / "cache" / "sprint_v2"
    research = ROOT / "cache" / "research_v3"
    final = ROOT / "cache" / "final_v3"
    heavy = ROOT / "cache" / "heavy_reranking_v1"

    sources = {
        "train": {
            "Lexical": load_source(learned / "reranker_train_lexical_top200.npz"),
            "Giga": load_source(learned / "reranker_train_giga480_top200.npz"),
            "Giga_local": load_source(sprint / "giga480_local_train_top100.npz"),
            "Description": load_source(sprint / "description_train_top200.npz"),
            "Title_params": load_source(sprint / "title_params_train_top200.npz"),
            "Giga_local_deep": load_source(final / "giga_local_train_top200.npz"),
            "Description_local": load_source(final / "description_local_train_top100.npz"),
            "Giga_geo50": load_source(final / "giga_geo50_train_top100.npz"),
        },
        "internal": {
            "Lexical": load_source(research / "lexical_internal_top200.npz"),
            "Giga": load_source(research / "giga_internal_top500.npz", 200),
            "Giga_local": load_source(research / "giga_local_internal_top300.npz", 100),
            "Description": load_source(research / "description_internal_top400.npz", 200),
            "Title_params": load_source(research / "title_params_internal_top400.npz", 200),
            "Giga_local_deep": load_source(research / "giga_local_internal_top300.npz", 200),
            "Description_local": load_source(research / "description_local_internal_top100.npz"),
            "Giga_geo50": load_source(research / "giga_geo50_internal_top100.npz"),
        },
        "dev": {
            "Lexical": load_source(
                ROOT / "cache" / "lexical_v1_dev_top200.npz", item_index=item_index
            ),
            "Giga": load_source(ROOT / "cache" / "giga480_dev_top200.npz", item_index=item_index),
            "Giga_local": load_source(sprint / "giga480_local_dev_top100.npz"),
            "Description": load_source(sprint / "description_dev_top200.npz"),
            "Title_params": load_source(sprint / "title_params_dev_top200.npz"),
            "Giga_local_deep": load_source(research / "giga_local_dev_top300.npz", 200),
            "Description_local": load_source(research / "description_local_dev_top100.npz"),
            "Giga_geo50": load_source(research / "giga_geo50_dev_top100.npz"),
        },
    }

    for split in ("train", "internal", "dev"):
        geo = load_npz(heavy / f"{split}_giga_geo_redirect_top100.npz")
        sources[split]["Geo_redirect"] = (geo["rankings"], geo["scores"])

    return sources


def benchmark_sources():
    benchmark = ROOT / "cache" / "benchmark_v2"
    final = ROOT / "cache" / "final_v3"
    heavy = ROOT / "cache" / "heavy_reranking_v1"
    sources = {
        "benchmark": {
            "Lexical": load_source(benchmark / "lexical_benchmark_top200.npz"),
            "Giga": load_source(benchmark / "giga480_benchmark_top200.npz"),
            "Giga_local": load_source(benchmark / "giga480_local_benchmark_top100.npz"),
            "Description": load_source(benchmark / "description_benchmark_top200.npz"),
            "Title_params": load_source(benchmark / "title_params_benchmark_top200.npz"),
            "Giga_local_deep": load_source(final / "giga_local_benchmark_top200.npz"),
            "Description_local": load_source(final / "description_local_benchmark_top100.npz"),
            "Giga_geo50": load_source(final / "giga_geo50_benchmark_top100.npz"),
        }
    }
    geo = load_npz(heavy / "benchmark_giga_geo_redirect_top100.npz")
    sources["benchmark"]["Geo_redirect"] = (geo["rankings"], geo["scores"])
    return sources


def transition_summary():
    summary = pd.read_parquet(
        ROOT / "cache" / "heavy_reranking_v1" / "geo_redirect_transition_summary.parquet"
    )
    return {
        row.search_location_id: {
            "top1_probability": row.top1_probability,
            "entropy": row.entropy,
        }
        for row in summary.itertuples(index=False)
    }


def make_builder(items, frames, sources):
    vectorizer = TfidfVectorizer(
        dtype=np.float32,
        min_df=3,
        max_features=60000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    vectorizer.fit(normalize_text(items.item_infm_params_text))
    base = CandidateFeatureBuilder(items, sources, frames, vectorizer, ALL_SOURCES)
    token_idf = joblib.load(ROOT / "cache" / "cheap_signal_ablation_v1" / "token_idf.joblib")
    fields = FieldInteractionBuilder(items, token_idf)
    return base, fields


def collect_current_sample():
    local_items, benchmark_items, frames = load_frames()
    local_index = pd.Index(local_items.item_id.astype(str))
    local_source_data = local_sources(local_index)
    benchmark_source_data = benchmark_sources()
    summary = transition_summary()

    local_base, local_fields = make_builder(
        local_items,
        {name: frames[name] for name in ("train", "internal", "dev")},
        local_source_data,
    )
    benchmark_base, benchmark_fields = make_builder(
        benchmark_items,
        {"benchmark": frames["benchmark"]},
        benchmark_source_data,
    )

    validation_model = CatBoostRanker()
    validation_model.load_model(ROOT / "cache" / "cheap_signal_ablation_v1" / "E4.cbm")
    benchmark_model = CatBoostRanker()
    benchmark_model.load_model(ROOT / "artifacts" / "final_v4_e4_yetirank.cbm")

    result = {"feature_names": np.asarray(FINAL_FEATURE_NAMES, dtype=str)}
    for split in ("train", "internal", "dev", "benchmark"):
        if split == "benchmark":
            items = benchmark_items
            base = benchmark_base
            fields = benchmark_fields
            geo = load_npz(
                ROOT / "cache" / "heavy_reranking_v1" / "benchmark_giga_geo_redirect_top100.npz"
            )
            model = benchmark_model
        else:
            items = local_items
            base = local_base
            fields = local_fields
            geo = load_npz(
                ROOT / "cache" / "heavy_reranking_v1" / f"{split}_giga_geo_redirect_top100.npz"
            )
            model = validation_model

        item_ids = items.item_id.astype(str).to_numpy()
        for query_index in QUERY_INDICES:
            candidate_positions, matrix = build_final_matrix(
                base,
                fields,
                geo,
                summary,
                split,
                int(query_index),
            )
            predictions = model.predict(matrix).astype(np.float64)
            order = np.lexsort((candidate_positions, -predictions))
            prefix = f"{split}_{query_index}"
            result[f"{prefix}_candidate_positions"] = candidate_positions
            result[f"{prefix}_candidate_ids"] = item_ids[candidate_positions]
            result[f"{prefix}_features"] = matrix
            result[f"{prefix}_predictions"] = predictions
            result[f"{prefix}_top50"] = item_ids[candidate_positions[order[:50]]]
    return result


def write_reference() -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_PATH, **collect_current_sample())
    print(f"Эталон сохранён: {GOLDEN_PATH}")


def compare_with_reference() -> dict:
    current = collect_current_sample()
    # Эталон создан локально до рефакторинга. item_id в pandas хранится как
    # object, поэтому NumPy сохраняет эти две строковые таблицы в object-массиве.
    with np.load(GOLDEN_PATH, allow_pickle=True) as golden:
        candidate_ids_identical = True
        top50_identical = True
        feature_max_abs_diff = 0.0
        prediction_max_abs_diff = 0.0

        for name, value in current.items():
            expected = golden[name]
            if name.endswith("candidate_ids") or name.endswith("candidate_positions"):
                candidate_ids_identical &= np.array_equal(value, expected)
            elif name.endswith("top50"):
                top50_identical &= np.array_equal(value, expected)
            elif name.endswith("features"):
                feature_max_abs_diff = max(
                    feature_max_abs_diff,
                    float(np.max(np.abs(value - expected))),
                )
            elif name.endswith("predictions"):
                prediction_max_abs_diff = max(
                    prediction_max_abs_diff,
                    float(np.max(np.abs(value - expected))),
                )

        feature_names_identical = np.array_equal(current["feature_names"], golden["feature_names"])

    report = {
        "candidate_ids_identical": bool(candidate_ids_identical),
        "feature_names_identical": bool(feature_names_identical),
        "feature_matrix_max_abs_diff": feature_max_abs_diff,
        "prediction_max_abs_diff": prediction_max_abs_diff,
        "top50_identical": bool(top50_identical),
    }
    report["passed"] = all(
        [
            report["candidate_ids_identical"],
            report["feature_names_identical"],
            report["feature_matrix_max_abs_diff"] < 1e-7,
            report["prediction_max_abs_diff"] < 1e-7,
            report["top50_identical"],
        ]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()
    if args.write_reference:
        write_reference()
        return
    print(json.dumps(compare_with_reference(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
