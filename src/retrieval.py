from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import BallTree

from .utils import choose_device, normalize_text


def save_source(path: Path, rankings: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rankings=rankings.astype(np.int32), scores=scores.astype(np.float32))


def load_source(path: Path, depth: int | None = None, item_index: pd.Index | None = None):
    with np.load(path, allow_pickle=False) as payload:
        if item_index is not None and "item_ids" in payload.files:
            shape = payload["item_ids"].shape
            rankings = item_index.get_indexer(payload["item_ids"].astype(str).ravel()).reshape(
                shape
            )
        else:
            rankings = payload["rankings"]
        scores = payload["scores"]
    if depth is not None:
        rankings, scores = rankings[:, :depth], scores[:, :depth]
    return rankings.astype(np.int32), scores.astype(np.float32)


def sparse_topk(query_matrix, item_matrix, top_k: int, batch_size: int = 8):
    """Ищет top-k по разреженному скалярному произведению без полной матрицы."""
    rankings = np.empty((query_matrix.shape[0], top_k), np.int32)
    values = np.empty((query_matrix.shape[0], top_k), np.float32)
    for start in range(0, query_matrix.shape[0], batch_size):
        end = min(start + batch_size, query_matrix.shape[0])
        similarity = (
            (query_matrix[start:end] @ item_matrix.T).toarray().astype(np.float32, copy=False)
        )
        local = np.argpartition(similarity, similarity.shape[1] - top_k, axis=1)[:, -top_k:]
        local_scores = np.take_along_axis(similarity, local, axis=1)
        order = np.argsort(local_scores, axis=1)[:, ::-1]
        rankings[start:end] = np.take_along_axis(local, order, axis=1)
        values[start:end] = np.take_along_axis(local_scores, order, axis=1)
    return rankings, values


def exact_dense_topk(queries, items, top_k: int, batch_size: int = 16):
    """Ищет точный top-k по нормированным плотным векторам небольшими пакетами."""
    rankings = np.empty((len(queries), top_k), np.int32)
    values = np.empty((len(queries), top_k), np.float32)
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        similarity = np.asarray(queries[start:end], np.float32) @ np.asarray(items, np.float32).T
        local = np.argpartition(similarity, similarity.shape[1] - top_k, axis=1)[:, -top_k:]
        local_scores = np.take_along_axis(similarity, local, axis=1)
        order = np.argsort(local_scores, axis=1)[:, ::-1]
        rankings[start:end] = np.take_along_axis(local, order, axis=1)
        values[start:end] = np.take_along_axis(local_scores, order, axis=1)
    return rankings, values


def restricted_dense(query_embeddings, item_embeddings, query_keys, item_keys, top_k: int):
    """Выполняет плотный поиск только среди items с тем же ключом группы."""
    rankings = np.full((len(query_embeddings), top_k), -1, np.int32)
    values = np.full((len(query_embeddings), top_k), -np.inf, np.float32)
    groups = {key: np.flatnonzero(item_keys == key) for key in np.unique(item_keys)}
    for key in np.unique(query_keys):
        query_rows, item_rows = np.flatnonzero(query_keys == key), groups.get(key)
        if item_rows is None or not len(item_rows):
            continue
        k = min(top_k, len(item_rows))
        block = np.asarray(item_embeddings[item_rows], np.float32)
        for start in range(0, len(query_rows), 32):
            rows = query_rows[start : start + 32]
            similarity = np.asarray(query_embeddings[rows], np.float32) @ block.T
            local = np.argpartition(similarity, similarity.shape[1] - k, axis=1)[:, -k:]
            local_scores = np.take_along_axis(similarity, local, axis=1)
            order = np.argsort(local_scores, axis=1)[:, ::-1]
            rankings[rows, :k] = item_rows[np.take_along_axis(local, order, axis=1)]
            values[rows, :k] = np.take_along_axis(local_scores, order, axis=1)
    return rankings, values


def restricted_sparse(query_matrix, item_matrix, query_keys, item_keys, top_k: int):
    """Выполняет разреженный поиск только среди items с тем же ключом группы."""
    rankings = np.full((query_matrix.shape[0], top_k), -1, np.int32)
    values = np.full((query_matrix.shape[0], top_k), -np.inf, np.float32)
    groups = {key: np.flatnonzero(item_keys == key) for key in np.unique(item_keys)}
    for key in np.unique(query_keys):
        query_rows, item_rows = np.flatnonzero(query_keys == key), groups.get(key)
        if item_rows is None or not len(item_rows):
            continue
        k = min(top_k, len(item_rows))
        for start in range(0, len(query_rows), 16):
            rows = query_rows[start : start + 16]
            similarity = (
                (query_matrix[rows] @ item_matrix[item_rows].T).toarray().astype(np.float32)
            )
            local = np.argpartition(similarity, similarity.shape[1] - k, axis=1)[:, -k:]
            local_scores = np.take_along_axis(similarity, local, axis=1)
            order = np.argsort(local_scores, axis=1)[:, ::-1]
            rankings[rows, :k] = item_rows[np.take_along_axis(local, order, axis=1)]
            values[rows, :k] = np.take_along_axis(local_scores, order, axis=1)
    return rankings, values


def encode_giga(
    texts,
    model_name: str,
    query_prefix: str = "",
    batch_size: int = 64,
    revision: str | None = None,
):
    from sentence_transformers import SentenceTransformer

    device = choose_device()
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
        revision=revision,
    )
    prepared = [query_prefix + str(text) for text in texts]
    return model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)


def tfidf_candidates(frame, items, field: str, top_k: int, local: bool = False):
    documents = normalize_text(items[field])
    vectorizer = TfidfVectorizer(dtype=np.float32, min_df=3, max_features=100000, sublinear_tf=True)
    item_matrix = vectorizer.fit_transform(documents)
    query_matrix = vectorizer.transform(frame.search_query.fillna("").astype(str))
    if local:
        return restricted_sparse(
            query_matrix,
            item_matrix,
            frame.search_location_id.to_numpy(),
            items.item_location_id.to_numpy(),
            top_k,
        )
    return sparse_topk(query_matrix, item_matrix, top_k)


def geo50_dense(frame, items, query_embeddings, item_embeddings, top_k: int = 100):
    """Ищет похожие объявления в радиусе 50 км от центра локации запроса."""
    latitude = pd.to_numeric(items.item_latitude, errors="coerce").to_numpy(float)
    longitude = pd.to_numeric(items.item_longitude, errors="coerce").to_numpy(float)
    valid_coordinates = np.isfinite(latitude) & np.isfinite(longitude)
    valid_positions = np.flatnonzero(valid_coordinates)
    coordinates = np.column_stack([latitude[valid_coordinates], longitude[valid_coordinates]])
    tree = BallTree(np.radians(coordinates), metric="haversine")
    centers = (
        pd.DataFrame(
            {
                "location": items.item_location_id,
                "lat": latitude,
                "lon": longitude,
            }
        )
        .dropna()
        .groupby("location")[["lat", "lon"]]
        .median()
    )
    rankings = np.full((len(frame), top_k), -1, np.int32)
    scores = np.full((len(frame), top_k), -np.inf, np.float32)

    for location in np.unique(frame.search_location_id):
        if location not in centers.index:
            continue
        query_rows = np.flatnonzero(frame.search_location_id.to_numpy() == location)
        center = centers.loc[location, ["lat", "lon"]].to_numpy(float)
        center = np.radians(center).reshape(1, -1)
        nearby_tree_rows = tree.query_radius(center, r=50 / 6371.0088)[0]
        candidate_positions = valid_positions[nearby_tree_rows]
        if not len(candidate_positions):
            continue

        result_count = min(top_k, len(candidate_positions))
        candidate_embeddings = np.asarray(item_embeddings[candidate_positions], np.float32)
        for start in range(0, len(query_rows), 32):
            rows = query_rows[start : start + 32]
            query_batch = np.asarray(query_embeddings[rows], np.float32)
            similarity = query_batch @ candidate_embeddings.T
            local_top = np.argpartition(similarity, similarity.shape[1] - result_count, axis=1)[
                :, -result_count:
            ]
            local_scores = np.take_along_axis(similarity, local_top, axis=1)
            order = np.argsort(local_scores, axis=1)[:, ::-1]
            ordered_local = np.take_along_axis(local_top, order, axis=1)
            rankings[rows, :result_count] = candidate_positions[ordered_local]
            scores[rows, :result_count] = np.take_along_axis(local_scores, order, axis=1)
    return rankings, scores
