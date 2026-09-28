from __future__ import annotations

import numpy as np
import pandas as pd


LAPLACE_ALPHA = 1.0
MAX_REDIRECT_LOCATIONS = 5


def build_transition_table(
    train: pd.DataFrame,
    items: pd.DataFrame,
    top_locations: int = MAX_REDIRECT_LOCATIONS,
):
    """Считает, куда пользователи из каждой локации реально выбирали объявления.

    Получается простая цепочка: локация поиска -> выбранные в train объявления ->
    их локации -> P(item_location | search_location). В итоговую таблицу попадают
    пять самых частых направлений для каждой исходной локации.
    """
    item_location_by_id = items.set_index(items.item_id.astype(str)).item_location_id
    history = pd.DataFrame(
        {
            "search_location_id": train.search_location_id,
            "item_location_id": train.item_id.astype(str).map(item_location_by_id),
        }
    ).dropna()

    counts = (
        history.groupby(["search_location_id", "item_location_id"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("search_location_id")["count"].transform("sum")
    destination_count = counts.groupby("search_location_id")["item_location_id"].transform("count")

    # +1 не подбирался как гиперпараметр. Сглаживание немного уменьшает влияние
    # малых частот и делает распределение устойчивее для редких направлений.
    numerator = counts["count"] + LAPLACE_ALPHA
    denominator = totals + LAPLACE_ALPHA * destination_count
    counts["probability"] = numerator / denominator

    counts = counts.sort_values(
        ["search_location_id", "probability", "item_location_id"],
        ascending=[True, False, True],
    )
    counts["redirect_rank"] = counts.groupby("search_location_id").cumcount() + 1

    entropy_terms = -counts.probability * np.log(counts.probability)
    entropy = (
        counts.assign(entropy_term=entropy_terms).groupby("search_location_id").entropy_term.sum()
    )
    summary = (
        counts.groupby("search_location_id")
        .agg(
            observations=("count", "sum"),
            top1_probability=("probability", "max"),
            destinations=("item_location_id", "count"),
        )
        .join(entropy.rename("entropy"))
        .reset_index()
    )

    top_transitions = counts[counts.redirect_rank <= top_locations].copy()
    return top_transitions, summary


def behavioral_geo_candidates(
    frame: pd.DataFrame,
    items: pd.DataFrame,
    query_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    transitions: pd.DataFrame,
    top_k: int = 100,
):
    """Ищет похожие объявления в типичных локациях назначения из train."""
    item_locations = items.item_location_id.to_numpy()
    item_positions_by_location = {
        location: np.flatnonzero(item_locations == location)
        for location in np.unique(item_locations)
    }
    transitions_by_source = {
        location: part for location, part in transitions.groupby("search_location_id")
    }

    query_count = len(frame)
    rankings = np.full((query_count, top_k), -1, np.int32)
    scores = np.full((query_count, top_k), -np.inf, np.float32)
    probabilities = np.zeros((query_count, top_k), np.float32)
    redirect_ranks = np.zeros((query_count, top_k), np.int8)
    target_locations = np.full((query_count, top_k), -1, np.int64)

    for query_index, source_location in enumerate(frame.search_location_id.to_numpy()):
        possible_destinations = transitions_by_source.get(source_location)
        if possible_destinations is None:
            continue

        available_destinations = []
        for row in possible_destinations.itertuples(index=False):
            positions = item_positions_by_location.get(row.item_location_id)
            if positions is not None and len(positions):
                available_destinations.append((row, positions))

        if not available_destinations:
            continue

        candidate_positions = np.unique(
            np.concatenate([positions for _, positions in available_destinations])
        )
        query_embedding = np.asarray(query_embeddings[query_index], np.float32)
        candidate_embeddings = np.asarray(item_embeddings[candidate_positions], np.float32)
        similarity = query_embedding @ candidate_embeddings.T

        result_count = min(top_k, len(candidate_positions))
        local_top = np.argpartition(similarity, len(similarity) - result_count)[-result_count:]
        local_order = local_top[np.argsort(similarity[local_top])[::-1]]
        chosen_positions = candidate_positions[local_order]

        probability_by_location = {
            row.item_location_id: row.probability for row, _ in available_destinations
        }
        rank_by_location = {
            row.item_location_id: row.redirect_rank for row, _ in available_destinations
        }
        chosen_locations = item_locations[chosen_positions]

        rankings[query_index, :result_count] = chosen_positions
        scores[query_index, :result_count] = similarity[local_order]
        probabilities[query_index, :result_count] = [
            probability_by_location[location] for location in chosen_locations
        ]
        redirect_ranks[query_index, :result_count] = [
            rank_by_location[location] for location in chosen_locations
        ]
        target_locations[query_index, :result_count] = chosen_locations

    return {
        "rankings": rankings,
        "scores": scores,
        "destination_probability": probabilities,
        "redirect_rank": redirect_ranks,
        "source_location": frame.search_location_id.to_numpy(),
        "target_location": target_locations,
    }
