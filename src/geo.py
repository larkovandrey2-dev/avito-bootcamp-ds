from __future__ import annotations

import numpy as np
import pandas as pd


def build_transition_table(train: pd.DataFrame, items: pd.DataFrame, top_locations: int = 5):
    location = items.set_index(items.item_id.astype(str)).item_location_id
    history = pd.DataFrame({
        "search_location_id": train.search_location_id,
        "item_location_id": train.item_id.astype(str).map(location),
    }).dropna()
    counts = history.groupby(["search_location_id", "item_location_id"]).size().rename("count").reset_index()
    totals = counts.groupby("search_location_id")["count"].transform("sum")
    destinations = counts.groupby("search_location_id")["item_location_id"].transform("count")
    counts["probability"] = (counts["count"] + 1.0) / (totals + destinations)
    counts = counts.sort_values(
        ["search_location_id", "probability", "item_location_id"],
        ascending=[True, False, True],
    )
    counts["redirect_rank"] = counts.groupby("search_location_id").cumcount() + 1
    entropy = counts.assign(term=-counts.probability*np.log(counts.probability)).groupby("search_location_id").term.sum()
    summary = counts.groupby("search_location_id").agg(
        observations=("count", "sum"), top1_probability=("probability", "max"),
        destinations=("item_location_id", "count"),
    ).join(entropy.rename("entropy")).reset_index()
    return counts[counts.redirect_rank <= top_locations].copy(), summary


def behavioral_geo_candidates(frame, items, query_embeddings, item_embeddings, transitions, top_k: int = 100):
    item_locations = items.item_location_id.to_numpy()
    groups = {key: np.flatnonzero(item_locations == key) for key in np.unique(item_locations)}
    transitions_by_source = {key: part for key, part in transitions.groupby("search_location_id")}
    rankings = np.full((len(frame), top_k), -1, np.int32)
    scores = np.full((len(frame), top_k), -np.inf, np.float32)
    probabilities = np.zeros((len(frame), top_k), np.float32)
    redirect_rank = np.zeros((len(frame), top_k), np.int8)
    target_location = np.full((len(frame), top_k), -1, np.int64)
    for q, source in enumerate(frame.search_location_id.to_numpy()):
        part = transitions_by_source.get(source)
        if part is None:
            continue
        available = [(row, groups.get(row.item_location_id)) for row in part.itertuples(index=False)]
        available = [(row, positions) for row, positions in available if positions is not None and len(positions)]
        if not available:
            continue
        candidates = np.unique(np.concatenate([positions for _, positions in available]))
        similarity = np.asarray(query_embeddings[q], np.float32) @ np.asarray(item_embeddings[candidates], np.float32).T
        k = min(top_k, len(candidates)); local = np.argpartition(similarity, len(similarity)-k)[-k:]
        order = local[np.argsort(similarity[local])[::-1]]; chosen = candidates[order]
        probability_by_location = {row.item_location_id:row.probability for row,_ in available}
        rank_by_location = {row.item_location_id:row.redirect_rank for row,_ in available}
        chosen_locations = item_locations[chosen]
        rankings[q,:k], scores[q,:k] = chosen, similarity[order]
        probabilities[q,:k] = [probability_by_location[x] for x in chosen_locations]
        redirect_rank[q,:k] = [rank_by_location[x] for x in chosen_locations]
        target_location[q,:k] = chosen_locations
    return {
        "rankings":rankings, "scores":scores, "destination_probability":probabilities,
        "redirect_rank":redirect_rank, "source_location":frame.search_location_id.to_numpy(),
        "target_location":target_location,
    }
