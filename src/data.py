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
    "train": {"search_query", "search_location_id", "search_infm_params_text", "item_id", "item_title_raw"},
    "benchmark_queries": {"query_id", "search_query", "search_location_id", "search_infm_params_text"},
    "benchmark_items": {"item_id", "item_title_raw", "item_location_id", "item_infm_params_text", "item_description_raw"},
}


def validate_data_dir(data_dir: Path) -> dict[str, Path]:
    paths = {name: data_dir / filename for name, filename in FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Не найдены исходные файлы:\n" + "\n".join(missing))
    for name, path in paths.items():
        columns = set(pq.ParquetFile(path).schema.names)
        absent = REQUIRED[name] - columns
        if absent:
            raise ValueError(f"В {path.name} отсутствуют колонки: {sorted(absent)}")
    return paths


def load_data(data_dir: Path):
    paths = validate_data_dir(data_dir)
    return (
        pd.read_parquet(paths["train"]),
        pd.read_parquet(paths["benchmark_queries"]),
        pd.read_parquet(paths["benchmark_items"]),
    )


def unique_items(train: pd.DataFrame) -> pd.DataFrame:
    item_columns = [c for c in train.columns if c == "item_id" or c.startswith("item_")]
    return train[item_columns].drop_duplicates("item_id").reset_index(drop=True)


SEARCH_COLUMNS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]


def unique_associations(raw: pd.DataFrame) -> pd.DataFrame:
    pairs = raw[SEARCH_COLUMNS + ["item_id"]].copy()
    pairs["_normalized_query"] = normalize_text(pairs.search_query)
    pairs["_normalized_params"] = normalize_text(pairs.search_infm_params_text)
    pairs["_signature"] = list(zip(
        pairs._normalized_query,
        pairs.search_location_id.astype("string").fillna("<MISSING>"),
        pairs.search_is_delivery_search.astype("string").fillna("<MISSING>"),
        pairs._normalized_params,
        pairs.search_category.astype("string").fillna("<MISSING>"),
    ))
    return pairs.drop_duplicates(["_signature", "item_id"]).reset_index(drop=True)


def _sample_units(pool, n, unit, filter_rate, seed):
    pool = pool.assign(_filter_present=pool._normalized_params.ne(""))
    target = round(n*filter_rate); parts=[]; used=set()
    for flag,count,offset in ((True,target,1),(False,n-target,2)):
        part=(pool[pool._filter_present.eq(flag)&~pool[unit].isin(used)]
              .sample(frac=1,random_state=seed+offset).drop_duplicates(unit).head(count))
        parts.append(part);used.update(part[unit])
    selected=pd.concat(parts,ignore_index=True)
    if len(selected)<n:
        extra=(pool[~pool[unit].isin(used)].sample(frac=1,random_state=seed+3)
               .drop_duplicates(unit).head(n-len(selected)))
        selected=pd.concat([selected,extra],ignore_index=True)
    return selected.sample(frac=1,random_state=seed+4).reset_index(drop=True)


def build_validation_and_corpus(raw: pd.DataFrame, seed: int = 42):
    """Точный single-positive протокол, использованный в v4."""
    pairs=unique_associations(raw)
    cold=_sample_units(pairs,3758,"_normalized_query",.3695,seed);cold["segment"]="cold"
    warm_pool=pairs[pairs._normalized_query.map(pairs.groupby("_normalized_query")["_signature"].nunique()).ge(2)&~pairs._normalized_query.isin(cold._normalized_query)]
    warm=_sample_units(warm_pool,1978,"_normalized_query",.3695,seed+10);warm["segment"]="text_warm_only"
    counts=pairs.groupby("_signature")["item_id"].nunique()
    exact_pool=pairs[pairs._signature.map(counts).ge(2)&~pairs._normalized_query.isin(cold._normalized_query)&~pairs._signature.isin(warm._signature)]
    exact=_sample_units(exact_pool,264,"_signature",.3695,seed+20);exact["segment"]="exact_signature_warm"
    validation=pd.concat([cold,warm,exact],ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)
    strata=validation.segment+"__"+validation._normalized_params.ne("").astype(str)
    development,holdout=train_test_split(validation,train_size=5000,random_state=seed,stratify=strata)
    items=unique_items(raw);required=items[items.item_id.isin(validation.item_id)]
    filler=items[~items.item_id.isin(required.item_id)].sample(n=189212-len(required),random_state=seed)
    corpus=pd.concat([required,filler]).sample(frac=1,random_state=seed).reset_index(drop=True)
    return development.reset_index(drop=True),holdout.reset_index(drop=True),corpus


def build_ranker_train(raw, development, corpus, n: int = 15000, seed: int = 42):
    pairs=unique_associations(raw);corpus_ids=set(corpus.item_id.astype(str));dev_text=set(normalize_text(development.search_query));dev_signatures=set(development._signature)
    eligible=pairs[pairs.item_id.astype(str).isin(corpus_ids)&~pairs._signature.isin(dev_signatures)]
    without_text=eligible[~eligible._normalized_query.isin(dev_text)]
    source=without_text if len(without_text)>=n else eligible
    return source.sample(n=n,random_state=seed).reset_index(drop=True)


def build_internal(raw, development, ranker_train, corpus, n: int = 5000, seed: int = 42):
    pairs=unique_associations(raw);forbidden=set(normalize_text(development.search_query))|set(normalize_text(ranker_train.search_query))
    eligible=pairs[pairs.item_id.astype(str).isin(set(corpus.item_id.astype(str)))&~pairs._normalized_query.isin(forbidden)]
    return eligible.sample(n=n,random_state=seed).reset_index(drop=True)
