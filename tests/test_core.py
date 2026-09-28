import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.features import (
    FIELD_FEATURE_NAMES,
    FINAL_FEATURE_NAMES,
    FieldInteractionBuilder,
)
from src.geo import build_transition_table
from src.ranking import retrieval100_selection, validate_submission
from src.retrieval import sparse_topk
from src.utils import normalize_text


ROOT = Path(__file__).resolve().parents[1]


def test_normalize_text():
    values = pd.Series(["  Ёжик,  МАСТЕР! ", None])
    assert normalize_text(values).tolist() == ["ежик мастер", ""]


def test_sparse_topk():
    query_matrix = sparse.csr_matrix([[1.0, 0.0]], dtype=np.float32)
    item_matrix = sparse.csr_matrix([[0.1, 0.0], [0.9, 0.0], [0.4, 0.0]], dtype=np.float32)
    rankings, scores = sparse_topk(query_matrix, item_matrix, top_k=2)
    assert rankings.tolist() == [[1, 2]]
    assert np.allclose(scores, [[0.9, 0.4]])


def test_geo_transition_probabilities_sum_to_one():
    train = pd.DataFrame({"search_location_id": [1, 1, 1], "item_id": ["a", "a", "b"]})
    items = pd.DataFrame({"item_id": ["a", "b"], "item_location_id": [10, 20]})
    transitions, _ = build_transition_table(train, items)
    probability_sum = transitions.groupby("search_location_id").probability.sum()
    assert np.allclose(probability_sum.to_numpy(), 1.0)


def test_feature_count_and_order_match_frozen_schema():
    schema = json.loads((ROOT / "artifacts" / "final_v4_feature_schema.json").read_text())
    assert len(FINAL_FEATURE_NAMES) == 185
    assert FINAL_FEATURE_NAMES == schema["ordered_features"]


def test_field_interaction_on_small_example():
    items = pd.DataFrame(
        {
            "item_title_raw": ["Ремонт аккумулятора шуруповерта"],
            "item_infm_params_text": [""],
            "item_description_raw": ["Меняем элементы питания"],
        }
    )
    builder = FieldInteractionBuilder(items, {"ремонт": 1.0, "шуруповерта": 3.0})
    query = pd.Series({"search_query": "ремонт шуруповерта"})
    features = builder.build(query, np.array([0]))
    title_coverage = FIELD_FEATURE_NAMES.index("title_query_token_coverage")
    rare_match = FIELD_FEATURE_NAMES.index("rarest_query_token_matched_title")
    assert features[0, title_coverage] == 1.0
    assert features[0, rare_match] == 1.0


def test_retrieval100_selection_keeps_positive_and_source_negatives():
    labels = np.array([0, 1, 0, 0], dtype=np.int8)
    scores = np.array([0.9, 0.1, 0.8, 0.7])
    present = np.array([[1, 1, 0, 0], [0, 1, 1, 0]])
    selected = retrieval100_selection(labels, scores, present)
    assert selected[0] == 1
    assert set(selected) == {0, 1, 2, 3}


def test_validate_submission_on_valid_small_content():
    query_ids = [str(index) for index in range(2452)]
    item_ids = [str(index) for index in range(50)]
    answer = " ".join(item_ids)
    submission = pd.DataFrame({"query_id": query_ids, "answer": answer})
    queries = pd.DataFrame({"query_id": query_ids})
    items = pd.DataFrame({"item_id": item_ids})
    result = validate_submission(submission, queries, items)
    assert result["all_exactly_50"]
    assert result["all_unique"]


def test_golden_parity_when_local_cache_is_available():
    required = ROOT / "cache" / "reranker_frozen" / "fixed_item_corpus_ordered.pkl"
    if not required.exists():
        return
    from tests.parity_v4 import compare_with_reference

    assert compare_with_reference()["passed"]
