from __future__ import annotations

import numpy as np
from catboost import CatBoostRanker, Pool

from .utils import remap_groups


TRAINING_GROUP_SIZE = 101
SOURCE_NEGATIVES_LIMIT = 20


def retrieval100_selection(labels, baseline_scores, present_matrix):
    """Собирает одну обучающую группу из реальных retrieval-кандидатов.

    Положительный объект добавляется обязательно. Затем берутся отрицательные
    примеры из разных источников, а свободные места заполняются сильными
    кандидатами baseline. Это обычная выборка из выдачи, похожая на inference,
    а не отдельный hard-negative mining по ошибкам уже обученной модели.
    """
    positive_rows = np.flatnonzero(labels == 1)
    if not len(positive_rows):
        return None

    selected_rows = [int(positive_rows[0])]
    seen_rows = set(selected_rows)

    for source_index in range(present_matrix.shape[0]):
        source_rows = np.flatnonzero(present_matrix[source_index] > 0)
        for row_index in source_rows[:SOURCE_NEGATIVES_LIMIT]:
            row_index = int(row_index)
            is_new_negative = row_index not in seen_rows and labels[row_index] == 0
            if is_new_negative:
                selected_rows.append(row_index)
                seen_rows.add(row_index)
            if len(selected_rows) >= TRAINING_GROUP_SIZE:
                return np.asarray(selected_rows, np.int32)

    baseline_order = np.argsort(-baseline_scores, kind="stable")
    for row_index in baseline_order:
        row_index = int(row_index)
        is_new_negative = row_index not in seen_rows and labels[row_index] == 0
        if is_new_negative:
            selected_rows.append(row_index)
            seen_rows.add(row_index)
        if len(selected_rows) >= TRAINING_GROUP_SIZE:
            break

    return np.asarray(selected_rows, np.int32)


def train_yetirank(features, labels, groups, params):
    """Обучает один frozen CatBoost YetiRank на группах поисковых запросов."""
    model = CatBoostRanker(
        **params,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    pool = Pool(features, labels, group_id=remap_groups(groups))
    model.fit(pool)
    return model


def positive_rank(candidate_positions, scores, positive_position):
    matching_rows = np.flatnonzero(candidate_positions == positive_position)
    if not len(matching_rows):
        return 10000

    ordered_rows = np.lexsort((candidate_positions, -scores))
    positive_row = matching_rows[0]
    return int(np.flatnonzero(ordered_rows == positive_row)[0] + 1)


def validate_submission(frame, queries, items):
    """Проверяет формат answer_v4.csv до ручной отправки."""
    answer_tokens = frame.answer.str.split(" ")
    valid_item_ids = set(items.item_id.astype(str))
    expected_query_ids = set(queries.query_id.astype(str))

    result = {
        "rows": int(len(frame)),
        "columns_exact": frame.columns.tolist() == ["query_id", "answer"],
        "query_ids_unique": bool(frame.query_id.is_unique),
        "query_ids_complete": set(frame.query_id) == expected_query_ids,
        "all_exactly_50": bool(answer_tokens.map(len).eq(50).all()),
        "all_unique": bool(answer_tokens.map(lambda values: len(values) == len(set(values))).all()),
        "all_items_valid": bool(
            answer_tokens.map(
                lambda values: all(item_id in valid_item_ids for item_id in values)
            ).all()
        ),
    }

    checks_without_row_count = [value for key, value in result.items() if key != "rows"]
    if result["rows"] != 2452 or not all(checks_without_row_count):
        raise ValueError(result)

    return result
