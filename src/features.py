from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import BASE_SOURCES
from .utils import normalize_text, safe_text


PREFIX = {
    "Lexical": "lexical",
    "Giga": "giga",
    "Giga_local": "giga_local",
    "Description": "description",
    "Title_params": "title_params",
    "Giga_local_deep": "giga_local_deep",
    "Description_local": "description_local",
    "Giga_geo50": "giga_geo50",
    "Geo_redirect": "geo_redirect",
}

TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.I)
ALNUM_RE = re.compile(r"(?=.*[a-zа-я])(?=.*\d)[a-zа-я0-9]+", re.I)

# Значение пришло из frozen fusion-реализации v3/v4. Отдельный подбор k=30
# в журнале исследований не зафиксирован, поэтому мы не приписываем ему оптимальность.
RRF_K = 30
MISSING_RANK = 201
EARTH_RADIUS_KM = 6371.0088


def base_feature_names(source_names=BASE_SOURCES):
    """Возвращает frozen-порядок 131 базового признака."""
    absolute_names = []
    relative_names = []

    for source in source_names:
        prefix = PREFIX[source]
        absolute_names.extend(
            f"{prefix}_{suffix}"
            for suffix in (
                "present",
                "rank",
                "reciprocal_rank",
                "score",
                "score_over_top1",
                "score_gap_top1",
            )
        )
        relative_names.extend(
            f"{prefix}_{suffix}"
            for suffix in (
                "normalized_rank",
                "score_gap_rank50",
                "query_score_mean",
                "query_score_std",
                "score_z",
                "score_percentile",
            )
        )

    query_and_item_names = [
        "num_global_sources",
        "best_global_rank",
        "global_rrf",
        "same_location",
        "same_category",
        "query_filter_nonempty",
        "item_params_nonempty",
        "filter_tfidf_cosine",
        "filter_token_matches",
        "filter_token_fraction",
        "filter_token_jaccard",
        "rating",
        "rating_missing",
        "log_reviews",
        "log_price",
        "price_missing",
        "phone_hidden",
        "message_forbidden",
    ]
    agreement_names = [
        "global_local_rank_diff",
        "global_local_score_diff",
        "giga_present_both",
        "num_sources_present",
        "best_source_rank",
        "second_best_source_rank",
        "source_rank_variance",
        "dense_agreement",
        "lexical_dense_agreement",
        "local_global_agreement",
    ]
    distance_names = [
        "geo_distance_km",
        "geo_log_distance",
        "geo_missing",
        "within_10km",
        "within_25km",
        "within_50km",
        "within_100km",
    ]
    return absolute_names + query_and_item_names + relative_names + agreement_names + distance_names


FIELD_FEATURE_NAMES = []
for field in ("title", "params", "description"):
    FIELD_FEATURE_NAMES.extend(
        f"{field}_{suffix}"
        for suffix in (
            "query_token_coverage",
            "item_token_coverage",
            "token_jaccard",
            "normalized_exact_query_present",
            "all_query_tokens_present",
            "longest_ordered_coverage",
            "exact_token_matches",
        )
    )

FIELD_FEATURE_NAMES += [
    "exact_query_equals_title",
    "title_is_substring_query",
    "title_prefix_match",
    "title_first_query_token_match",
    "title_last_query_token_match",
    "rarest_query_token_matched_title",
    "rarest_query_token_matched_params",
    "rarest_query_token_matched_description",
    "matched_query_idf_sum",
    "matched_query_idf_fraction",
    "query_has_number",
    "item_has_same_number",
    "number_conflict",
    "query_has_latin_token",
    "latin_token_overlap",
    "exact_alphanumeric_token_match",
    "title_match_stronger_than_description",
    "params_match_stronger_than_description",
    "description_only_match",
    "num_fields_with_match",
    "best_field_coverage",
    "second_best_field_coverage",
]

GEO_FEATURE_NAMES = [
    "geo_redirect_present",
    "geo_redirect_rank",
    "geo_redirect_rr",
    "geo_redirect_score",
    "transition_probability",
    "transition_location_rank",
    "transition_confidence",
    "transition_entropy",
    "candidate_location_is_top1_redirect",
    "candidate_location_is_top3_redirect",
    "candidate_location_is_top5_redirect",
]

FINAL_FEATURE_NAMES = base_feature_names() + GEO_FEATURE_NAMES + FIELD_FEATURE_NAMES


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    latitude_delta = lat2 - lat1
    longitude_delta = lon2 - lon1
    value = (
        np.sin(latitude_delta / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(longitude_delta / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def collect_candidate_positions(source_data, source_names, split, query_index):
    rankings = [source_data[split][name][0][query_index] for name in source_names]
    valid_positions = [ranking[ranking >= 0] for ranking in rankings]
    return np.unique(np.concatenate(valid_positions))


def build_source_features(candidate_positions, ranking, scores):
    """Описывает место кандидата внутри выдачи одного источника."""
    valid = ranking >= 0
    ranking = ranking[valid]
    scores = scores[valid]
    candidate_count = len(candidate_positions)

    present = np.zeros(candidate_count, np.float32)
    ranks = np.full(candidate_count, MISSING_RANK, np.float32)
    raw_scores = np.zeros(candidate_count, np.float32)

    candidate_rows = np.searchsorted(candidate_positions, ranking)
    present[candidate_rows] = 1
    ranks[candidate_rows] = np.arange(1, len(ranking) + 1)
    raw_scores[candidate_rows] = scores

    top1_score = float(scores[0]) if len(scores) else 1.0
    mean_score = float(scores.mean()) if len(scores) else 0.0
    score_std = float(scores.std()) if len(scores) else 0.0
    rank50_score = float(scores[min(49, len(scores) - 1)]) if len(scores) else 0.0
    source_depth = max(len(ranking), 1)

    return {
        "present": present,
        "rank": ranks,
        "reciprocal_rank": np.where(present > 0, 1 / ranks, 0).astype(np.float32),
        "score": raw_scores,
        "score_over_top1": np.where(
            present > 0,
            raw_scores / max(abs(top1_score), 1e-7),
            0,
        ).astype(np.float32),
        "score_gap_top1": np.where(present > 0, top1_score - raw_scores, 0).astype(np.float32),
        "normalized_rank": np.where(present > 0, ranks / source_depth, 1).astype(np.float32),
        "score_gap_rank50": np.where(present > 0, raw_scores - rank50_score, 0).astype(np.float32),
        "query_score_mean": np.full(candidate_count, mean_score, np.float32),
        "query_score_std": np.full(candidate_count, score_std, np.float32),
        "score_z": np.where(
            present > 0,
            (raw_scores - mean_score) / max(score_std, 1e-7),
            0,
        ).astype(np.float32),
        "score_percentile": np.where(
            present > 0,
            1 - (ranks - 1) / max(source_depth - 1, 1),
            0,
        ).astype(np.float32),
    }


def source_agreement_summary(source_parts, source_names):
    present_matrix = np.vstack([source_parts[name]["present"] for name in source_names])
    rank_matrix = np.vstack([source_parts[name]["rank"] for name in source_names])
    masked_ranks = np.where(present_matrix > 0, rank_matrix, np.inf)
    ordered_ranks = np.sort(masked_ranks, axis=0)
    source_count = np.maximum(present_matrix.sum(axis=0), 1)
    mean_rank = np.sum(np.where(present_matrix > 0, rank_matrix, 0), axis=0) / source_count
    rank_variance = (
        np.sum(np.where(present_matrix > 0, (rank_matrix - mean_rank) ** 2, 0), axis=0)
        / source_count
    )
    return present_matrix, {
        "num_sources_present": present_matrix.sum(axis=0),
        "best_source_rank": np.where(np.isfinite(ordered_ranks[0]), ordered_ranks[0], MISSING_RANK),
        "second_best_source_rank": np.where(
            np.isfinite(ordered_ranks[1]), ordered_ranks[1], MISSING_RANK
        ),
        "source_rank_variance": rank_variance,
    }


class CandidateFeatureBuilder:
    """Собирает 131 базовый признак для пары запрос–кандидат."""

    def __init__(self, items, sources, frames, filter_vectorizer, source_names):
        self.items = items
        self.sources = sources
        self.frames = frames
        self.source_names = tuple(source_names)
        self.item_locations = items.item_location_id.to_numpy()
        self.item_categories = items.item_category_id.to_numpy()
        self.item_params = normalize_text(items.item_infm_params_text)
        self.params_array = self.item_params.to_numpy()
        self.filter_items = filter_vectorizer.transform(self.item_params)
        self.filter_queries = {
            split: filter_vectorizer.transform(normalize_text(frame.search_infm_params_text))
            for split, frame in frames.items()
        }

        price = pd.to_numeric(items.item_price, errors="coerce").to_numpy(float)
        self.price_missing = np.isnan(price) | (price < 0)
        self.log_price = np.log1p(np.where(self.price_missing, 0, price)).astype(np.float32)

        rating = pd.to_numeric(items.item_rating, errors="coerce").to_numpy(float)
        self.rating_missing = np.isnan(rating)
        self.rating = np.nan_to_num(rating, nan=0).astype(np.float32)

        reviews = pd.to_numeric(items.item_rating_reviews_count, errors="coerce").to_numpy(float)
        self.log_reviews = np.log1p(np.nan_to_num(reviews, nan=0)).astype(np.float32)
        self.phone_hidden = items.item_is_phone_hidden.to_numpy().astype(np.float32)
        self.message_forbidden = items.item_is_message_forbidden.to_numpy().astype(np.float32)
        self.latitude = pd.to_numeric(items.item_latitude, errors="coerce").to_numpy(float)
        self.longitude = pd.to_numeric(items.item_longitude, errors="coerce").to_numpy(float)
        self.location_centroids = (
            pd.DataFrame(
                {
                    "location": self.item_locations,
                    "lat": self.latitude,
                    "lon": self.longitude,
                }
            )
            .dropna()
            .groupby("location")[["lat", "lon"]]
            .median()
        )

    def _source_features(self, split, query_index, candidate_positions):
        values = {}
        parts = {}
        for source_name in self.source_names:
            ranking = self.sources[split][source_name][0][query_index]
            scores = self.sources[split][source_name][1][query_index]
            source_values = build_source_features(candidate_positions, ranking, scores)
            prefix = PREFIX[source_name]
            for suffix, array in source_values.items():
                values[f"{prefix}_{suffix}"] = array
            parts[source_name] = source_values
        return values, parts

    def _filter_features(self, split, query_index, candidate_positions):
        frame = self.frames[split]
        filter_text = normalize_text(frame.search_infm_params_text.iloc[[query_index]]).iloc[0]
        filter_tokens = set(filter_text.split()) if filter_text else set()
        candidate_count = len(candidate_positions)

        if filter_text:
            item_token_sets = [
                set(self.params_array[position].split()) for position in candidate_positions
            ]
            token_matches = np.array(
                [len(filter_tokens & tokens) for tokens in item_token_sets],
                np.float32,
            )
            cosine = (
                (self.filter_items[candidate_positions] @ self.filter_queries[split][query_index].T)
                .toarray()
                .ravel()
                .astype(np.float32)
            )
            jaccard = np.array(
                [
                    token_matches[row] / max(len(filter_tokens | item_token_sets[row]), 1)
                    for row in range(candidate_count)
                ],
                np.float32,
            )
        else:
            item_token_sets = None
            token_matches = np.zeros(candidate_count, np.float32)
            cosine = np.zeros(candidate_count, np.float32)
            jaccard = np.zeros(candidate_count, np.float32)

        return {
            "query_filter_nonempty": np.full(candidate_count, bool(filter_text), np.float32),
            "item_params_nonempty": np.array(
                [bool(self.params_array[position]) for position in candidate_positions],
                np.float32,
            ),
            "filter_tfidf_cosine": cosine,
            "filter_token_matches": token_matches,
            "filter_token_fraction": token_matches / max(len(filter_tokens), 1),
            "filter_token_jaccard": jaccard,
        }

    def _metadata_features(self, candidate_positions):
        return {
            "rating": self.rating[candidate_positions],
            "rating_missing": self.rating_missing[candidate_positions].astype(np.float32),
            "log_reviews": self.log_reviews[candidate_positions],
            "log_price": self.log_price[candidate_positions],
            "price_missing": self.price_missing[candidate_positions].astype(np.float32),
            "phone_hidden": self.phone_hidden[candidate_positions],
            "message_forbidden": self.message_forbidden[candidate_positions],
        }

    def _agreement_features(self, source_parts):
        lexical = source_parts["Lexical"]
        giga = source_parts["Giga"]
        local = source_parts["Giga_local"]
        both_giga = (giga["present"] > 0) & (local["present"] > 0)

        present_matrix, summary = source_agreement_summary(source_parts, self.source_names)
        values = {
            "num_global_sources": lexical["present"] + giga["present"],
            "best_global_rank": np.minimum(lexical["rank"], giga["rank"]),
            "global_rrf": (
                np.where(
                    lexical["present"] > 0,
                    1 / (RRF_K + lexical["rank"]),
                    0,
                )
                + np.where(
                    giga["present"] > 0,
                    1 / (RRF_K + giga["rank"]),
                    0,
                )
            ).astype(np.float32),
            "global_local_rank_diff": np.where(both_giga, giga["rank"] - local["rank"], 0).astype(
                np.float32
            ),
            "global_local_score_diff": np.where(
                both_giga, local["score"] - giga["score"], 0
            ).astype(np.float32),
            "giga_present_both": both_giga.astype(np.float32),
            "dense_agreement": both_giga.astype(np.float32),
            "lexical_dense_agreement": ((lexical["present"] > 0) & (giga["present"] > 0)).astype(
                np.float32
            ),
            "local_global_agreement": both_giga.astype(np.float32),
            **summary,
        }
        return values, present_matrix

    def _distance_features(self, query_location, candidate_positions):
        candidate_count = len(candidate_positions)
        distance = np.full(candidate_count, np.nan)
        if query_location in self.location_centroids.index:
            has_coordinates = np.isfinite(self.latitude[candidate_positions]) & np.isfinite(
                self.longitude[candidate_positions]
            )
            center = self.location_centroids.loc[query_location]
            distance[has_coordinates] = haversine(
                center["lat"],
                center["lon"],
                self.latitude[candidate_positions][has_coordinates],
                self.longitude[candidate_positions][has_coordinates],
            )

        missing = ~np.isfinite(distance)
        clean_distance = np.where(missing, 0, distance).astype(np.float32)
        values = {
            "geo_distance_km": clean_distance,
            "geo_log_distance": np.log1p(clean_distance),
            "geo_missing": missing.astype(np.float32),
        }
        for radius in (10, 25, 50, 100):
            values[f"within_{radius}km"] = ((distance <= radius) & ~missing).astype(np.float32)
        return values

    def _query_context_features(
        self,
        split,
        query_index,
        candidate_positions,
    ):
        query_frame = self.frames[split]
        query_location = query_frame.search_location_id.iloc[query_index]
        query_category = query_frame.search_category.iloc[query_index]
        values = {
            "same_location": (self.item_locations[candidate_positions] == query_location).astype(
                np.float32
            ),
            "same_category": (self.item_categories[candidate_positions] == query_category).astype(
                np.float32
            ),
        }
        values.update(self._filter_features(split, query_index, candidate_positions))
        values.update(self._metadata_features(candidate_positions))
        values.update(self._distance_features(query_location, candidate_positions))
        return values

    @staticmethod
    def _ordered_base_matrix(values, source_parts):
        feature_names = base_feature_names(BASE_SOURCES)
        matrix = np.column_stack([values[name] for name in feature_names]).astype(np.float32)

        # Geo_redirect участвует в общем пуле кандидатов. Эти четыре агрегата
        # намеренно считаются только по восьми источникам frozen v3: так устроен v4.
        _, base_summary = source_agreement_summary(source_parts, BASE_SOURCES)
        for name, array in base_summary.items():
            matrix[:, feature_names.index(name)] = array
        return np.nan_to_num(matrix, nan=0, posinf=0, neginf=0)

    def build(self, split: str, query_index: int):
        """Объединяет кандидатов и собирает базовые семейства признаков."""
        candidate_positions = collect_candidate_positions(
            self.sources, self.source_names, split, query_index
        )
        values, source_parts = self._source_features(split, query_index, candidate_positions)

        agreement_values, present_matrix = self._agreement_features(source_parts)
        values.update(agreement_values)
        values.update(
            self._query_context_features(
                split,
                query_index,
                candidate_positions,
            )
        )
        matrix = self._ordered_base_matrix(values, source_parts)
        return candidate_positions, matrix, values, present_matrix


def _tokens(value):
    return tuple(TOKEN_RE.findall(safe_text(value)))


def consecutive_query_token_presence(query_tokens, field_tokens):
    """Считает максимальную цепочку токенов запроса, встречающихся в поле.

    Историческое публичное имя признака — longest_ordered_coverage. Оно сохранено
    для совместимости v4. Фактически позиции слов внутри поля не проверяются:
    учитывается только присутствие последовательных токенов самого запроса.
    """
    if not query_tokens or not field_tokens:
        return 0.0

    field_set = set(field_tokens)
    best = 0
    current = 0
    for token in query_tokens:
        if token in field_set:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best / len(query_tokens)


def build_token_idf(items: pd.DataFrame):
    counts = Counter()
    item_count = len(items)
    for row in items.itertuples(index=False):
        text = " ".join(
            str(getattr(row, field, ""))
            for field in (
                "item_title_raw",
                "item_infm_params_text",
                "item_description_raw",
            )
        )
        counts.update(set(_tokens(text)))
    return {token: math.log((item_count + 1) / (count + 1)) + 1 for token, count in counts.items()}


class FieldInteractionBuilder:
    """Сравнивает запрос отдельно с title, params и description."""

    def __init__(self, items: pd.DataFrame, idf: dict[str, float]):
        self.title = items.item_title_raw.fillna("").astype(str).to_numpy()
        self.params = items.item_infm_params_text.fillna("").astype(str).to_numpy()
        self.description = (
            items.item_description_raw.fillna("").astype(str).str.slice(0, 2000).to_numpy()
        )
        self.idf = idf

    @lru_cache(maxsize=120000)
    def _item(self, position):
        raw_fields = (
            self.title[position],
            self.params[position],
            self.description[position],
        )
        clean_fields = tuple(safe_text(value) for value in raw_fields)
        field_tokens = tuple(_tokens(value) for value in clean_fields)
        token_sets = tuple(set(tokens) for tokens in field_tokens)
        return clean_fields, field_tokens, token_sets

    def build(self, query_row, candidate_positions):
        query = safe_text(query_row.search_query)
        query_tokens = _tokens(query)
        query_set = set(query_tokens)
        query_idf = {token: self.idf.get(token, 0.0) for token in query_set}
        rarest_token = max(query_idf, key=query_idf.get) if query_idf else None
        total_query_idf = sum(query_idf.values()) or 1.0
        query_numbers = {token for token in query_set if token.isdigit()}
        query_latin = {token for token in query_set if re.search(r"[a-z]", token)}
        query_alnum = {token for token in query_set if ALNUM_RE.fullmatch(token)}

        result = np.zeros((len(candidate_positions), len(FIELD_FEATURE_NAMES)), np.float32)
        for row_index, position in enumerate(candidate_positions):
            clean_fields, field_tokens, token_sets = self._item(int(position))
            coverages = []
            offset = 0

            for clean_field, tokens, token_set in zip(clean_fields, field_tokens, token_sets):
                matched_tokens = query_set & token_set
                query_coverage = len(matched_tokens) / max(len(query_set), 1)
                coverages.append(query_coverage)
                result[row_index, offset : offset + 7] = [
                    query_coverage,
                    len(matched_tokens) / max(len(token_set), 1),
                    len(matched_tokens) / max(len(query_set | token_set), 1),
                    float(bool(query) and query in clean_field),
                    float(bool(query_set) and query_set <= token_set),
                    consecutive_query_token_presence(query_tokens, tokens),
                    len(matched_tokens),
                ]
                offset += 7

            title_set, params_set, description_set = token_sets
            all_item_tokens = title_set | params_set | description_set
            item_numbers = {token for token in all_item_tokens if token.isdigit()}
            item_latin = {token for token in all_item_tokens if re.search(r"[a-z]", token)}
            item_alnum = {token for token in all_item_tokens if ALNUM_RE.fullmatch(token)}
            matched_anywhere = query_set & all_item_tokens
            sorted_coverages = sorted(coverages)

            result[row_index, 21:] = [
                float(bool(query) and query == clean_fields[0]),
                float(bool(clean_fields[0]) and clean_fields[0] in query),
                float(bool(query) and clean_fields[0].startswith(query)),
                float(bool(query_tokens) and query_tokens[0] in title_set),
                float(bool(query_tokens) and query_tokens[-1] in title_set),
                float(rarest_token in title_set) if rarest_token else 0,
                float(rarest_token in params_set) if rarest_token else 0,
                float(rarest_token in description_set) if rarest_token else 0,
                sum(query_idf[token] for token in matched_anywhere),
                sum(query_idf[token] for token in matched_anywhere) / total_query_idf,
                float(bool(query_numbers)),
                float(bool(query_numbers & item_numbers)),
                float(
                    bool(query_numbers)
                    and bool(item_numbers)
                    and not bool(query_numbers & item_numbers)
                ),
                float(bool(query_latin)),
                float(bool(query_latin & item_latin)),
                float(bool(query_alnum & item_alnum)),
                float(coverages[0] > coverages[2]),
                float(coverages[1] > coverages[2]),
                float(coverages[2] > 0 and coverages[0] == 0 and coverages[1] == 0),
                sum(coverage > 0 for coverage in coverages),
                max(coverages),
                sorted_coverages[-2],
            ]
        return result


def geo_redirect_features(
    geo_cache,
    query_index,
    candidate_positions,
    values,
    transition_summary,
    search_location,
):
    """Добавляет 11 признаков поведенческого перехода между локациями."""
    result = np.zeros((len(candidate_positions), len(GEO_FEATURE_NAMES)), np.float32)
    for column, key in enumerate(("present", "rank", "reciprocal_rank", "score")):
        result[:, column] = values[f"geo_redirect_{key}"]

    ranking = geo_cache["rankings"][query_index]
    valid = ranking >= 0
    candidate_rows = np.searchsorted(candidate_positions, ranking[valid])
    result[candidate_rows, 4] = geo_cache["destination_probability"][query_index, valid]
    result[candidate_rows, 5] = geo_cache["redirect_rank"][query_index, valid]

    summary = transition_summary.get(search_location, {"top1_probability": 0.0, "entropy": 0.0})
    result[:, 6] = summary["top1_probability"]
    result[:, 7] = summary["entropy"]
    result[:, 8] = (result[:, 5] == 1) & (result[:, 0] > 0)
    result[:, 9] = (result[:, 5] <= 3) & (result[:, 5] > 0)
    result[:, 10] = (result[:, 5] <= 5) & (result[:, 5] > 0)
    return result


def build_final_matrix(
    base_builder,
    field_builder,
    geo_cache,
    transition_summary,
    split,
    query_index,
):
    """Соединяет 131 базовый, 11 geo и 43 текстовых признака."""
    candidate_positions, base, values, _ = base_builder.build(split, query_index)
    search_location = base_builder.frames[split].search_location_id.iloc[query_index]
    geo = geo_redirect_features(
        geo_cache,
        query_index,
        candidate_positions,
        values,
        transition_summary,
        search_location,
    )
    query_row = base_builder.frames[split].iloc[query_index]
    fields = field_builder.build(query_row, candidate_positions)
    matrix = np.column_stack([base, geo, fields]).astype(np.float32, copy=False)
    assert matrix.shape[1] == 185
    return candidate_positions, matrix
