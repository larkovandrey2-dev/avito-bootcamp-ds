from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ALL_SOURCES, BASE_SOURCES, PipelineConfig
from .data import (
    build_internal,
    build_ranker_train,
    build_validation_and_corpus,
    load_data,
    validate_data_dir,
)
from .features import (
    CandidateFeatureBuilder,
    FieldInteractionBuilder,
    FINAL_FEATURE_NAMES,
    build_final_matrix,
    build_token_idf,
)
from .geo import behavioral_geo_candidates, build_transition_table
from .ranking import retrieval100_selection, train_yetirank, validate_submission
from .retrieval import (
    encode_giga,
    exact_dense_topk,
    geo50_dense,
    load_source,
    restricted_dense,
    restricted_sparse,
    save_source,
    sparse_topk,
)
from .utils import atomic_json, normalize_text


def _source_paths(folder: Path):
    return {name: folder / f"{name.lower()}_top.npz" for name in BASE_SOURCES}


def _load_npz_dict(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _transition_summary_map(summary: pd.DataFrame):
    return {
        row.search_location_id: {
            "top1_probability": row.top1_probability,
            "entropy": row.entropy,
        }
        for row in summary.itertuples(index=False)
    }


def _load_or_encode_embeddings(
    path: Path,
    texts,
    config: PipelineConfig,
    query_prefix: str = "",
):
    if path.exists():
        return np.load(path, mmap_mode="r")

    embeddings = encode_giga(
        texts,
        config.embedding_model,
        query_prefix,
        revision=config.embedding_revision,
    )
    np.save(path, embeddings)
    return embeddings


def _build_sparse_sources(
    paths,
    query_frame,
    item_frame,
    query_locations,
    item_locations,
):
    if not paths["Lexical"].exists():
        vectorizer = TfidfVectorizer(
            dtype=np.float32,
            min_df=3,
            max_features=60000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        item_matrix = vectorizer.fit_transform(normalize_text(item_frame.item_title_raw))
        query_matrix = vectorizer.transform(query_frame.search_query.fillna(""))
        save_source(paths["Lexical"], *sparse_topk(query_matrix, item_matrix, 200))

    if not paths["Description"].exists() or not paths["Description_local"].exists():
        vectorizer = TfidfVectorizer(
            dtype=np.float32,
            min_df=3,
            max_features=100000,
            sublinear_tf=True,
        )
        descriptions = normalize_text(item_frame.item_description_raw).str.slice(0, 2000)
        item_matrix = vectorizer.fit_transform(descriptions)
        query_matrix = vectorizer.transform(query_frame.search_query.fillna(""))
        save_source(paths["Description"], *sparse_topk(query_matrix, item_matrix, 200))
        local_description = restricted_sparse(
            query_matrix,
            item_matrix,
            query_locations,
            item_locations,
            100,
        )
        save_source(paths["Description_local"], *local_description)

    if not paths["Title_params"].exists():
        titles = item_frame.item_title_raw.fillna("").astype(str)
        params = normalize_text(item_frame.item_infm_params_text).str.slice(0, 1200)
        documents = titles.str.cat(params, sep=" ")
        vectorizer = TfidfVectorizer(
            dtype=np.float32,
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        )
        item_matrix = vectorizer.fit_transform(documents)
        query_matrix = vectorizer.transform(query_frame.search_query.fillna(""))
        save_source(paths["Title_params"], *sparse_topk(query_matrix, item_matrix, 200))


def _build_dense_sources(
    paths,
    query_frame,
    item_frame,
    query_embeddings,
    item_embeddings,
    query_locations,
    item_locations,
):
    if not paths["Giga"].exists():
        save_source(
            paths["Giga"],
            *exact_dense_topk(query_embeddings, item_embeddings, 200),
        )

    if paths["Giga_local_deep"].exists():
        deep_local = load_source(paths["Giga_local_deep"])
    else:
        deep_local = restricted_dense(
            query_embeddings,
            item_embeddings,
            query_locations,
            item_locations,
            200,
        )
        save_source(paths["Giga_local_deep"], *deep_local)

    if not paths["Giga_local"].exists():
        save_source(
            paths["Giga_local"],
            deep_local[0][:, :100],
            deep_local[1][:, :100],
        )

    if not paths["Giga_geo50"].exists():
        geo50 = geo50_dense(
            query_frame,
            item_frame,
            query_embeddings,
            item_embeddings,
            100,
        )
        save_source(paths["Giga_geo50"], *geo50)


def _load_or_build_behavioral_geo(
    path,
    query_frame,
    item_frame,
    query_embeddings,
    item_embeddings,
    transition_table,
):
    if not path.exists():
        geo = behavioral_geo_candidates(
            query_frame,
            item_frame,
            query_embeddings,
            item_embeddings,
            transition_table,
            100,
        )
        np.savez_compressed(path, **geo)
    return _load_npz_dict(path)


def build_candidate_cache(
    config: PipelineConfig,
    scope: str,
    query_frame: pd.DataFrame,
    item_frame: pd.DataFrame,
    train: pd.DataFrame,
    transition_table: pd.DataFrame,
):
    """Строит и сохраняет девять источников кандидатов для одного корпуса."""
    del train  # История уже представлена готовой transition_table.
    folder = config.cache_dir / "v4" / scope
    folder.mkdir(parents=True, exist_ok=True)
    paths = _source_paths(folder)

    item_embeddings = _load_or_encode_embeddings(
        folder / "item_embeddings.npy",
        item_frame.item_title_raw.fillna(""),
        config,
    )
    query_embeddings = _load_or_encode_embeddings(
        folder / "query_embeddings.npy",
        query_frame.search_query.fillna(""),
        config,
        config.query_prefix,
    )

    item_locations = item_frame.item_location_id.to_numpy()
    query_locations = query_frame.search_location_id.to_numpy()

    _build_sparse_sources(
        paths,
        query_frame,
        item_frame,
        query_locations,
        item_locations,
    )
    _build_dense_sources(
        paths,
        query_frame,
        item_frame,
        query_embeddings,
        item_embeddings,
        query_locations,
        item_locations,
    )
    geo_path = folder / "geo_redirect_top100.npz"
    geo = _load_or_build_behavioral_geo(
        geo_path,
        query_frame,
        item_frame,
        query_embeddings,
        item_embeddings,
        transition_table,
    )

    sources = {scope: {name: load_source(path) for name, path in paths.items()}}
    sources[scope]["Geo_redirect"] = (geo["rankings"], geo["scores"])
    return sources, geo


def _feature_builders(items, query_frame, sources, scope, token_idf):
    filter_vectorizer = TfidfVectorizer(
        dtype=np.float32,
        min_df=3,
        max_features=60000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    filter_vectorizer.fit(normalize_text(items.item_infm_params_text))
    base_builder = CandidateFeatureBuilder(
        items,
        sources,
        {scope: query_frame},
        filter_vectorizer,
        ALL_SOURCES,
    )
    field_builder = FieldInteractionBuilder(items, token_idf)
    return base_builder, field_builder


def _load_or_build_token_idf(config: PipelineConfig, corpus: pd.DataFrame):
    path = config.cache_dir / "v4" / "token_idf.joblib"
    if path.exists():
        return joblib.load(path)
    token_idf = build_token_idf(corpus)
    joblib.dump(token_idf, path, compress=3)
    return token_idf


def build_training_matrix(
    config,
    train,
    train_queries,
    corpus,
    transition_table,
    transition_summary,
):
    """Строит retrieval100-группы и итоговую матрицу обучения YetiRank."""
    output = config.cache_dir / "v4" / "training_matrix.npz"
    if output.exists():
        return _load_npz_dict(output)

    sources, geo = build_candidate_cache(
        config,
        "train",
        train_queries,
        corpus,
        train,
        transition_table,
    )
    token_idf = _load_or_build_token_idf(config, corpus)
    base_builder, field_builder = _feature_builders(
        corpus, train_queries, sources, "train", token_idf
    )
    summary_map = _transition_summary_map(transition_summary)
    item_index = pd.Index(corpus.item_id.astype(str))
    sampler = joblib.load(config.artifacts_dir / "retrieval100_sampler.joblib")

    feature_blocks = []
    label_blocks = []
    group_blocks = []

    for query_index in range(len(train_queries)):
        candidate_positions, matrix = build_final_matrix(
            base_builder,
            field_builder,
            geo,
            summary_map,
            "train",
            query_index,
        )
        positive_id = str(train_queries.item_id.iloc[query_index])
        positive_position = item_index.get_indexer([positive_id])[0]
        labels = (candidate_positions == positive_position).astype(np.int8)
        if not labels.any():
            continue

        _, _, source_values, present_matrix = base_builder.build("train", query_index)
        legacy_features = np.column_stack(
            [source_values[name] for name in sampler["features"]]
        ).astype(np.float32)
        scaled = sampler["scaler"].transform(legacy_features)
        baseline_scores = sampler["model"].predict_proba(scaled)[:, 1]
        selected_rows = retrieval100_selection(labels, baseline_scores, present_matrix)

        feature_blocks.append(matrix[selected_rows])
        label_blocks.append(labels[selected_rows])
        group_blocks.append(np.full(len(selected_rows), query_index, np.int32))

    result = {
        "X": np.concatenate(feature_blocks),
        "y": np.concatenate(label_blocks),
        "g": np.concatenate(group_blocks),
    }
    np.savez_compressed(output, **result)
    return result


def load_or_train_ranker(config, training=None, retrain=False):
    model_path = config.artifacts_dir / "final_v4_e4_yetirank.cbm"
    if model_path.exists() and not retrain:
        model = CatBoostRanker()
        model.load_model(model_path)
        return model

    if training is None:
        raise ValueError("Для переобучения нужна training matrix")

    model = train_yetirank(
        training["X"],
        training["y"],
        training["g"],
        config.catboost_params,
    )
    model.save_model(model_path)
    return model


def _consolidate_cached_chunks(config, queries, items):
    folder = config.cache_dir / "final_v4_e4" / "benchmark_chunks"
    chunk_paths = sorted(folder.glob("q_*.npz"))
    if not chunk_paths:
        return None

    query_indices = []
    top_positions = []
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as part:
            query_indices.append(part["query_index"])
            top_positions.append(part["top_positions"])

    query_indices = np.concatenate(query_indices)
    top_positions = np.concatenate(top_positions)
    if not np.array_equal(query_indices, np.arange(len(queries))):
        return None

    item_ids = items.item_id.astype(str).to_numpy()
    answers = [" ".join(item_ids[row]) for row in top_positions]
    return pd.DataFrame({"query_id": queries.query_id.astype(str), "answer": answers})


def predict_benchmark(
    config,
    model,
    train,
    queries,
    items,
    transition_table,
    transition_summary,
):
    """Создаёт top-50 для benchmark или читает уже проверенные куски."""
    cached = _consolidate_cached_chunks(config, queries, items)
    if cached is not None:
        return cached, "cached benchmark chunks"

    sources, geo = build_candidate_cache(
        config,
        "benchmark",
        queries,
        items,
        train,
        transition_table,
    )
    idf_path = config.cache_dir / "v4" / "token_idf.joblib"
    if not idf_path.exists():
        raise FileNotFoundError("Не найден token_idf.joblib: сначала подготовьте training matrix")

    token_idf = joblib.load(idf_path)
    base_builder, field_builder = _feature_builders(items, queries, sources, "benchmark", token_idf)
    summary_map = _transition_summary_map(transition_summary)
    item_ids = items.item_id.astype(str).to_numpy()
    answers = []

    for query_index in range(len(queries)):
        candidate_positions, matrix = build_final_matrix(
            base_builder,
            field_builder,
            geo,
            summary_map,
            "benchmark",
            query_index,
        )
        scores = model.predict(matrix)
        order = np.lexsort((candidate_positions, -scores))[: config.top_k]
        answers.append(" ".join(item_ids[candidate_positions[order]]))

    submission = pd.DataFrame({"query_id": queries.query_id.astype(str), "answer": answers})
    return submission, "fresh benchmark inference"


def evaluate_split(
    config,
    scope,
    query_frame,
    items,
    train,
    model,
    transitions,
    summary,
):
    """Пересчитывает single-positive Hit@50 и candidate coverage."""
    sources, geo = build_candidate_cache(config, scope, query_frame, items, train, transitions)
    token_idf = _load_or_build_token_idf(config, items)
    base_builder, field_builder = _feature_builders(items, query_frame, sources, scope, token_idf)
    summary_map = _transition_summary_map(summary)
    item_index = pd.Index(items.item_id.astype(str))
    positive_positions = item_index.get_indexer(query_frame.item_id.astype(str))
    hits = np.zeros(len(query_frame), bool)
    coverage = np.zeros(len(query_frame), bool)

    for query_index in range(len(query_frame)):
        candidate_positions, matrix = build_final_matrix(
            base_builder,
            field_builder,
            geo,
            summary_map,
            scope,
            query_index,
        )
        positive_position = positive_positions[query_index]
        coverage[query_index] = (candidate_positions == positive_position).any()
        scores = model.predict(matrix)
        order = np.lexsort((candidate_positions, -scores))[:50]
        hits[query_index] = (candidate_positions[order] == positive_position).any()

    return {
        "hit_at_50": float(hits.mean()),
        "candidate_coverage": float(coverage.mean()),
        "queries": len(query_frame),
    }


def check_environment(config):
    paths = validate_data_dir(config.data_dir)
    model_path = config.artifacts_dir / "final_v4_e4_yetirank.cbm"
    chunk_folder = config.cache_dir / "final_v4_e4" / "benchmark_chunks"
    return {
        "data": {name: str(path) for name, path in paths.items()},
        "model": str(model_path),
        "model_exists": model_path.exists(),
        "feature_count": len(FINAL_FEATURE_NAMES),
        "cached_benchmark_chunks": len(list(chunk_folder.glob("q_*.npz"))),
    }


def _load_or_build_splits(config, train):
    split_dir = config.cache_dir / "v4" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    development_path = split_dir / "development.pkl"
    corpus_path = split_dir / "corpus.pkl"
    ranker_train_path = split_dir / "ranker_train.pkl"
    paths = (development_path, corpus_path, ranker_train_path)

    if all(path.exists() for path in paths):
        development = pd.read_pickle(development_path)
        corpus = pd.read_pickle(corpus_path)
        ranker_train = pd.read_pickle(ranker_train_path)
        return development, corpus, ranker_train

    development, _, corpus = build_validation_and_corpus(train, config.seed)
    ranker_train = build_ranker_train(train, development, corpus, seed=config.seed)
    development.to_pickle(development_path)
    corpus.to_pickle(corpus_path)
    ranker_train.to_pickle(ranker_train_path)
    return development, corpus, ranker_train


def _load_or_build_geo_history(config, ranker_train, corpus):
    transition_path = config.cache_dir / "v4" / "transition.parquet"
    summary_path = config.cache_dir / "v4" / "transition_summary.parquet"
    if transition_path.exists() and summary_path.exists():
        transitions = pd.read_parquet(transition_path)
        summary = pd.read_parquet(summary_path)
        return transitions, summary

    transitions, summary = build_transition_table(ranker_train, corpus)
    transitions.to_parquet(transition_path, index=False)
    summary.to_parquet(summary_path, index=False)
    atomic_json(
        config.cache_dir / "v4" / "transition_manifest.json",
        {
            "history": "ranker_train only",
            "rows": len(ranker_train),
            "seed": config.seed,
        },
    )
    return transitions, summary


def _evaluate_local_splits(
    config,
    train,
    development,
    ranker_train,
    corpus,
    model,
    transitions,
    transition_summary,
):
    internal = build_internal(
        train,
        development,
        ranker_train,
        corpus,
        seed=config.seed,
    )
    return {
        "internal": evaluate_split(
            config,
            "internal",
            internal,
            corpus,
            train,
            model,
            transitions,
            transition_summary,
        ),
        "development": evaluate_split(
            config,
            "development",
            development,
            corpus,
            train,
            model,
            transitions,
            transition_summary,
        ),
    }


def _save_and_validate_submission(config, submission, queries, items):
    config.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.output, index=False, encoding="utf-8")
    saved_submission = pd.read_csv(config.output, dtype=str)
    return validate_submission(saved_submission, queries, items)


def reproduce(config: PipelineConfig, retrain: bool = False):
    """Последовательно воспроизводит frozen v4 и проверяет итоговый CSV."""
    started = time.time()
    train, benchmark_queries, benchmark_items = load_data(config.data_dir)

    development, corpus, ranker_train = _load_or_build_splits(config, train)
    transitions, transition_summary = _load_or_build_geo_history(config, ranker_train, corpus)
    _load_or_build_token_idf(config, corpus)

    training = None
    if retrain:
        training = build_training_matrix(
            config,
            train,
            ranker_train,
            corpus,
            transitions,
            transition_summary,
        )

    model = load_or_train_ranker(config, training, retrain)
    submission, mode = predict_benchmark(
        config,
        model,
        train,
        benchmark_queries,
        benchmark_items,
        transitions,
        transition_summary,
    )

    local_metrics = None
    if retrain:
        local_metrics = _evaluate_local_splits(
            config,
            train,
            development,
            ranker_train,
            corpus,
            model,
            transitions,
            transition_summary,
        )

    validation = _save_and_validate_submission(
        config, submission, benchmark_queries, benchmark_items
    )

    report = {
        "mode": mode,
        "runtime_seconds": time.time() - started,
        "output": str(config.output),
        "validation": validation,
        "features": len(FINAL_FEATURE_NAMES),
        "local_metrics": local_metrics,
    }
    atomic_json(config.cache_dir / "v4" / "last_run.json", report)
    return report
