from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from .utils import normalize_text


FILES = {
    "train": "train.parquet",
    "benchmark_queries": "benchmark_queries.parquet",
    "benchmark_items": "benchmark_items.parquet",
}

REQUIRED = {
    "train": {
        "search_query",
        "search_location_id",
        "search_infm_params_text",
        "item_id",
        "item_title_raw",
    },
    "benchmark_queries": {
        "query_id",
        "search_query",
        "search_location_id",
        "search_infm_params_text",
    },
    "benchmark_items": {
        "item_id",
        "item_title_raw",
        "item_location_id",
        "item_infm_params_text",
        "item_description_raw",
    },
}

SEARCH_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

# Эти значения зафиксированы в протоколе v4 после ранних экспериментов.
VALIDATION_SIZE = 6000
DEVELOPMENT_SIZE = 5000
HOLDOUT_SIZE = 1000
COLD_COUNT = 3758
TEXT_WARM_COUNT = 1978
EXACT_WARM_COUNT = 264
TARGET_FILTER_RATE = 0.3695
RANKER_TRAIN_SIZE = 15000
INTERNAL_SIZE = 5000
LOCAL_CORPUS_SIZE = 189212


def validate_data_dir(data_dir: Path) -> dict[str, Path]:
    paths = {name: data_dir / filename for name, filename in FILES.items()}
    missing_paths = [str(path) for path in paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Не найдены исходные файлы:\n" + "\n".join(missing_paths))

    for name, path in paths.items():
        actual_columns = set(pq.ParquetFile(path).schema.names)
        missing_columns = REQUIRED[name] - actual_columns
        if missing_columns:
            message = f"В {path.name} отсутствуют колонки: {sorted(missing_columns)}"
            raise ValueError(message)

    return paths


def load_data(data_dir: Path):
    paths = validate_data_dir(data_dir)
    train = pd.read_parquet(paths["train"])
    benchmark_queries = pd.read_parquet(paths["benchmark_queries"])
    benchmark_items = pd.read_parquet(paths["benchmark_items"])
    return train, benchmark_queries, benchmark_items


def unique_items(train: pd.DataFrame) -> pd.DataFrame:
    item_columns = [
        column for column in train.columns if column == "item_id" or column.startswith("item_")
    ]
    return train[item_columns].drop_duplicates("item_id").reset_index(drop=True)


def unique_associations(raw: pd.DataFrame) -> pd.DataFrame:
    """Оставляет одну строку на видимый контекст поиска и выбранный item."""
    pairs = raw[SEARCH_COLUMNS + ["item_id"]].copy()
    pairs["_normalized_query"] = normalize_text(pairs.search_query)
    pairs["_normalized_params"] = normalize_text(pairs.search_infm_params_text)

    normalized_location = pairs.search_location_id.astype("string").fillna("<MISSING>")
    normalized_delivery = pairs.search_is_delivery_search.astype("string").fillna("<MISSING>")
    normalized_category = pairs.search_category.astype("string").fillna("<MISSING>")
    pairs["_signature"] = list(
        zip(
            pairs._normalized_query,
            normalized_location,
            normalized_delivery,
            pairs._normalized_params,
            normalized_category,
        )
    )

    duplicate_key = ["_signature", "item_id"]
    return pairs.drop_duplicates(duplicate_key).reset_index(drop=True)


def _sample_units(
    pool: pd.DataFrame,
    sample_size: int,
    unit_column: str,
    filter_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Выбирает уникальные запросы и сохраняет заданную долю строк с фильтрами."""
    pool = pool.assign(_filter_present=pool._normalized_params.ne(""))
    target_with_filter = round(sample_size * filter_rate)
    target_without_filter = sample_size - target_with_filter

    selected_parts = []
    used_units = set()
    targets = (
        (True, target_with_filter, 1),
        (False, target_without_filter, 2),
    )

    for filter_present, target_count, seed_offset in targets:
        available = pool[
            pool._filter_present.eq(filter_present) & ~pool[unit_column].isin(used_units)
        ]
        shuffled = available.sample(frac=1, random_state=seed + seed_offset)
        selected = shuffled.drop_duplicates(unit_column).head(target_count)
        selected_parts.append(selected)
        used_units.update(selected[unit_column])

    result = pd.concat(selected_parts, ignore_index=True)
    if len(result) < sample_size:
        remaining = pool[~pool[unit_column].isin(used_units)]
        shuffled = remaining.sample(frac=1, random_state=seed + 3)
        extra_count = sample_size - len(result)
        extra = shuffled.drop_duplicates(unit_column).head(extra_count)
        result = pd.concat([result, extra], ignore_index=True)

    return result.sample(frac=1, random_state=seed + 4).reset_index(drop=True)


def build_validation_and_corpus(raw: pd.DataFrame, seed: int = 42):
    """Строит single-positive validation и фиксированный локальный корпус v4."""
    pairs = unique_associations(raw)

    cold = _sample_units(pairs, COLD_COUNT, "_normalized_query", TARGET_FILTER_RATE, seed)
    cold["segment"] = "cold"

    signatures_per_text = pairs.groupby("_normalized_query")["_signature"].nunique()
    text_is_repeated = pairs._normalized_query.map(signatures_per_text).ge(2)
    text_not_used_in_cold = ~pairs._normalized_query.isin(cold._normalized_query)
    warm_pool = pairs[text_is_repeated & text_not_used_in_cold]
    text_warm = _sample_units(
        warm_pool,
        TEXT_WARM_COUNT,
        "_normalized_query",
        TARGET_FILTER_RATE,
        seed + 10,
    )
    text_warm["segment"] = "text_warm_only"

    items_per_signature = pairs.groupby("_signature")["item_id"].nunique()
    signature_has_several_items = pairs._signature.map(items_per_signature).ge(2)
    signature_not_used_in_text_warm = ~pairs._signature.isin(text_warm._signature)
    exact_pool = pairs[
        signature_has_several_items & text_not_used_in_cold & signature_not_used_in_text_warm
    ]
    exact_warm = _sample_units(
        exact_pool,
        EXACT_WARM_COUNT,
        "_signature",
        TARGET_FILTER_RATE,
        seed + 20,
    )
    exact_warm["segment"] = "exact_signature_warm"

    validation = pd.concat([cold, text_warm, exact_warm], ignore_index=True)
    validation = validation.sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(validation) != VALIDATION_SIZE:
        raise RuntimeError("Размер validation не совпал с frozen-протоколом v4")

    filter_flag = validation._normalized_params.ne("").astype(str)
    strata = validation.segment + "__" + filter_flag
    development, holdout = train_test_split(
        validation,
        train_size=DEVELOPMENT_SIZE,
        random_state=seed,
        stratify=strata,
    )
    if len(holdout) != HOLDOUT_SIZE:
        raise RuntimeError("Размер holdout не совпал с frozen-протоколом v4")

    all_items = unique_items(raw)
    required_items = all_items[all_items.item_id.isin(validation.item_id)]
    remaining_items = all_items[~all_items.item_id.isin(required_items.item_id)]
    filler_count = LOCAL_CORPUS_SIZE - len(required_items)
    filler_items = remaining_items.sample(n=filler_count, random_state=seed)
    corpus = pd.concat([required_items, filler_items])
    corpus = corpus.sample(frac=1, random_state=seed).reset_index(drop=True)

    return development.reset_index(drop=True), holdout.reset_index(drop=True), corpus


def build_ranker_train(
    raw: pd.DataFrame,
    development: pd.DataFrame,
    corpus: pd.DataFrame,
    n: int = RANKER_TRAIN_SIZE,
    seed: int = 42,
) -> pd.DataFrame:
    """Выбирает train-запросы, не повторяющие development, для обучения ранжирования."""
    pairs = unique_associations(raw)
    corpus_ids = set(corpus.item_id.astype(str))
    development_texts = set(normalize_text(development.search_query))
    development_signatures = set(development._signature)

    item_is_in_corpus = pairs.item_id.astype(str).isin(corpus_ids)
    signature_is_new = ~pairs._signature.isin(development_signatures)
    eligible = pairs[item_is_in_corpus & signature_is_new]

    text_is_new = ~eligible._normalized_query.isin(development_texts)
    without_development_text = eligible[text_is_new]
    source = without_development_text if len(without_development_text) >= n else eligible
    return source.sample(n=n, random_state=seed).reset_index(drop=True)


def build_internal(
    raw: pd.DataFrame,
    development: pd.DataFrame,
    ranker_train: pd.DataFrame,
    corpus: pd.DataFrame,
    n: int = INTERNAL_SIZE,
    seed: int = 42,
) -> pd.DataFrame:
    """Строит internal без текстов из обучения ранжирования и development."""
    pairs = unique_associations(raw)
    development_texts = set(normalize_text(development.search_query))
    ranker_train_texts = set(normalize_text(ranker_train.search_query))
    forbidden_texts = development_texts | ranker_train_texts
    corpus_ids = set(corpus.item_id.astype(str))

    item_is_in_corpus = pairs.item_id.astype(str).isin(corpus_ids)
    query_is_new = ~pairs._normalized_query.isin(forbidden_texts)
    eligible = pairs[item_is_in_corpus & query_is_new]
    return eligible.sample(n=n, random_state=seed).reset_index(drop=True)
