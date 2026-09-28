from __future__ import annotations

import numpy as np
from catboost import CatBoostRanker, Pool

from .utils import remap_groups


def retrieval100_selection(labels, baseline_scores, present_matrix):
    positive=np.flatnonzero(labels==1)
    if not len(positive):return None
    selected=[int(positive[0])];seen=set(selected)
    for source in range(present_matrix.shape[0]):
        for row in np.flatnonzero(present_matrix[source]>0)[:20]:
            row=int(row)
            if row not in seen and labels[row]==0:selected.append(row);seen.add(row)
            if len(selected)>=101:return np.asarray(selected,np.int32)
    for row in np.argsort(-baseline_scores,kind="stable"):
        row=int(row)
        if row not in seen and labels[row]==0:selected.append(row);seen.add(row)
        if len(selected)>=101:break
    return np.asarray(selected,np.int32)


def train_yetirank(features, labels, groups, params):
    model=CatBoostRanker(**params,verbose=False,allow_writing_files=False,thread_count=-1)
    model.fit(Pool(features,labels,group_id=remap_groups(groups)))
    return model


def positive_rank(candidates,scores,positive):
    match=np.flatnonzero(candidates==positive)
    if not len(match):return 10000
    order=np.lexsort((candidates,-scores))
    return int(np.flatnonzero(order==match[0])[0]+1)


def validate_submission(frame,queries,items):
    tokens=frame.answer.str.split(" ");valid=set(items.item_id.astype(str))
    result={
        "rows":int(len(frame)),"columns_exact":frame.columns.tolist()==["query_id","answer"],
        "query_ids_unique":bool(frame.query_id.is_unique),"query_ids_complete":set(frame.query_id)==set(queries.query_id.astype(str)),
        "all_exactly_50":bool(tokens.map(len).eq(50).all()),"all_unique":bool(tokens.map(lambda x:len(x)==len(set(x))).all()),
        "all_items_valid":bool(tokens.map(lambda x:all(v in valid for v in x)).all()),
    }
    if result["rows"]!=2452 or not all(v for k,v in result.items() if k!="rows"):raise ValueError(result)
    return result
