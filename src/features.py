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
    "Lexical":"lexical", "Giga":"giga", "Giga_local":"giga_local",
    "Description":"description", "Title_params":"title_params",
    "Giga_local_deep":"giga_local_deep", "Description_local":"description_local",
    "Giga_geo50":"giga_geo50", "Geo_redirect":"geo_redirect",
}
TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.I)
ALNUM_RE = re.compile(r"(?=.*[a-zа-я])(?=.*\d)[a-zа-я0-9]+", re.I)


def base_feature_names(source_names=BASE_SOURCES):
    base, relative = [], []
    for source in source_names:
        prefix = PREFIX[source]
        base += [f"{prefix}_{suffix}" for suffix in (
            "present", "rank", "reciprocal_rank", "score", "score_over_top1", "score_gap_top1")]
        relative += [f"{prefix}_{suffix}" for suffix in (
            "normalized_rank", "score_gap_rank50", "query_score_mean", "query_score_std",
            "score_z", "score_percentile")]
    base += [
        "num_global_sources", "best_global_rank", "global_rrf", "same_location", "same_category",
        "query_filter_nonempty", "item_params_nonempty", "filter_tfidf_cosine", "filter_token_matches",
        "filter_token_fraction", "filter_token_jaccard", "rating", "rating_missing", "log_reviews",
        "log_price", "price_missing", "phone_hidden", "message_forbidden",
    ]
    cross = [
        "global_local_rank_diff", "global_local_score_diff", "giga_present_both",
        "num_sources_present", "best_source_rank", "second_best_source_rank", "source_rank_variance",
        "dense_agreement", "lexical_dense_agreement", "local_global_agreement",
    ]
    geo = ["geo_distance_km", "geo_log_distance", "geo_missing", "within_10km", "within_25km", "within_50km", "within_100km"]
    return base + relative + cross + geo


FIELD_FEATURE_NAMES = []
for field in ("title", "params", "description"):
    FIELD_FEATURE_NAMES += [f"{field}_{suffix}" for suffix in (
        "query_token_coverage", "item_token_coverage", "token_jaccard",
        "normalized_exact_query_present", "all_query_tokens_present",
        "longest_ordered_coverage", "exact_token_matches",
    )]
FIELD_FEATURE_NAMES += [
    "exact_query_equals_title", "title_is_substring_query", "title_prefix_match",
    "title_first_query_token_match", "title_last_query_token_match",
    "rarest_query_token_matched_title", "rarest_query_token_matched_params",
    "rarest_query_token_matched_description", "matched_query_idf_sum", "matched_query_idf_fraction",
    "query_has_number", "item_has_same_number", "number_conflict", "query_has_latin_token",
    "latin_token_overlap", "exact_alphanumeric_token_match", "title_match_stronger_than_description",
    "params_match_stronger_than_description", "description_only_match", "num_fields_with_match",
    "best_field_coverage", "second_best_field_coverage",
]

GEO_FEATURE_NAMES = [
    "geo_redirect_present", "geo_redirect_rank", "geo_redirect_rr", "geo_redirect_score",
    "transition_probability", "transition_location_rank", "transition_confidence",
    "transition_entropy", "candidate_location_is_top1_redirect",
    "candidate_location_is_top3_redirect", "candidate_location_is_top5_redirect",
]
FINAL_FEATURE_NAMES = base_feature_names() + GEO_FEATURE_NAMES + FIELD_FEATURE_NAMES


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))


class CandidateFeatureBuilder:
    """Строит 131 базовый признак и объединяет кандидатов всех источников."""

    def __init__(self, items, sources, frames, filter_vectorizer, source_names):
        self.items, self.sources, self.frames = items, sources, frames
        self.source_names = tuple(source_names)
        self.item_locations = items.item_location_id.to_numpy()
        self.item_categories = items.item_category_id.to_numpy()
        self.item_params = normalize_text(items.item_infm_params_text)
        self.params_array = self.item_params.to_numpy()
        self.filter_items = filter_vectorizer.transform(self.item_params)
        self.filter_queries = {split:filter_vectorizer.transform(normalize_text(frame.search_infm_params_text)) for split,frame in frames.items()}
        price = pd.to_numeric(items.item_price,errors="coerce").to_numpy(float)
        self.price_missing = np.isnan(price) | (price<0); self.log_price=np.log1p(np.where(self.price_missing,0,price)).astype(np.float32)
        rating = pd.to_numeric(items.item_rating,errors="coerce").to_numpy(float)
        self.rating_missing=np.isnan(rating);self.rating=np.nan_to_num(rating,nan=0).astype(np.float32)
        reviews=pd.to_numeric(items.item_rating_reviews_count,errors="coerce").to_numpy(float)
        self.log_reviews=np.log1p(np.nan_to_num(reviews,nan=0)).astype(np.float32)
        self.phone_hidden=items.item_is_phone_hidden.to_numpy().astype(np.float32)
        self.message_forbidden=items.item_is_message_forbidden.to_numpy().astype(np.float32)
        self.lat=pd.to_numeric(items.item_latitude,errors="coerce").to_numpy(float)
        self.lon=pd.to_numeric(items.item_longitude,errors="coerce").to_numpy(float)
        self.centroids=pd.DataFrame({"location":self.item_locations,"lat":self.lat,"lon":self.lon}).dropna().groupby("location")[["lat","lon"]].median()

    def build(self, split: str, q: int):
        frame=self.frames[split]
        arrays=[self.sources[split][name][0][q] for name in self.source_names]
        candidates=np.unique(np.concatenate([x[x>=0] for x in arrays]));n=len(candidates)
        values,parts={},{}
        for source in self.source_names:
            prefix=PREFIX[source];ranking,scores=self.sources[split][source][0][q],self.sources[split][source][1][q]
            valid=ranking>=0;ranking,scores=ranking[valid],scores[valid]
            present=np.zeros(n,np.float32);ranks=np.full(n,201,np.float32);raw=np.zeros(n,np.float32)
            indices=np.searchsorted(candidates,ranking);present[indices]=1;ranks[indices]=np.arange(1,len(ranking)+1);raw[indices]=scores
            top1=float(scores[0]) if len(scores) else 1.;mean=float(scores.mean()) if len(scores) else 0.;std=float(scores.std()) if len(scores) else 0.
            rank50=float(scores[min(49,len(scores)-1)]) if len(scores) else 0.;depth=max(len(ranking),1)
            block={
                "present":present,"rank":ranks,"reciprocal_rank":np.where(present>0,1/ranks,0).astype(np.float32),"score":raw,
                "score_over_top1":np.where(present>0,raw/max(abs(top1),1e-7),0).astype(np.float32),
                "score_gap_top1":np.where(present>0,top1-raw,0).astype(np.float32),
                "normalized_rank":np.where(present>0,ranks/depth,1).astype(np.float32),
                "score_gap_rank50":np.where(present>0,raw-rank50,0).astype(np.float32),
                "query_score_mean":np.full(n,mean,np.float32),"query_score_std":np.full(n,std,np.float32),
                "score_z":np.where(present>0,(raw-mean)/max(std,1e-7),0).astype(np.float32),
                "score_percentile":np.where(present>0,1-(ranks-1)/max(depth-1,1),0).astype(np.float32),
            }
            for suffix,array in block.items():values[f"{prefix}_{suffix}"]=array
            parts[source]=block
        lexical,giga,local=parts["Lexical"],parts["Giga"],parts["Giga_local"]
        values["num_global_sources"]=lexical["present"]+giga["present"]
        values["best_global_rank"]=np.minimum(lexical["rank"],giga["rank"])
        values["global_rrf"]=(np.where(lexical["present"]>0,1/(30+lexical["rank"]),0)+np.where(giga["present"]>0,1/(30+giga["rank"]),0)).astype(np.float32)
        values["same_location"]=(self.item_locations[candidates]==frame.search_location_id.iloc[q]).astype(np.float32)
        values["same_category"]=(self.item_categories[candidates]==frame.search_category.iloc[q]).astype(np.float32)
        query_text=normalize_text(frame.search_infm_params_text.iloc[[q]]).iloc[0];query_tokens=set(query_text.split()) if query_text else set()
        item_tokens=[set(self.params_array[p].split()) for p in candidates] if query_text else None
        matches=np.array([len(query_tokens&t) for t in item_tokens],np.float32) if query_text else np.zeros(n,np.float32)
        cosine=(self.filter_items[candidates]@self.filter_queries[split][q].T).toarray().ravel().astype(np.float32) if query_text else np.zeros(n,np.float32)
        values["query_filter_nonempty"]=np.full(n,bool(query_text),np.float32);values["item_params_nonempty"]=np.array([bool(self.params_array[p]) for p in candidates],np.float32)
        values["filter_tfidf_cosine"]=cosine;values["filter_token_matches"]=matches;values["filter_token_fraction"]=matches/max(len(query_tokens),1)
        values["filter_token_jaccard"]=np.array([matches[i]/max(len(query_tokens|item_tokens[i]),1) for i in range(n)],np.float32) if query_text else np.zeros(n,np.float32)
        values["rating"],values["rating_missing"]=self.rating[candidates],self.rating_missing[candidates].astype(np.float32)
        values["log_reviews"],values["log_price"]=self.log_reviews[candidates],self.log_price[candidates]
        values["price_missing"]=self.price_missing[candidates].astype(np.float32);values["phone_hidden"],values["message_forbidden"]=self.phone_hidden[candidates],self.message_forbidden[candidates]
        both=(giga["present"]>0)&(local["present"]>0)
        values["global_local_rank_diff"]=np.where(both,giga["rank"]-local["rank"],0).astype(np.float32)
        values["global_local_score_diff"]=np.where(both,local["score"]-giga["score"],0).astype(np.float32);values["giga_present_both"]=both.astype(np.float32)
        pm=np.vstack([parts[name]["present"] for name in self.source_names]);rm=np.vstack([parts[name]["rank"] for name in self.source_names])
        ordered=np.sort(np.where(pm>0,rm,np.inf),axis=0);mean_rank=np.sum(np.where(pm>0,rm,0),axis=0)/np.maximum(pm.sum(0),1)
        values["num_sources_present"]=pm.sum(0);values["best_source_rank"]=np.where(np.isfinite(ordered[0]),ordered[0],201);values["second_best_source_rank"]=np.where(np.isfinite(ordered[1]),ordered[1],201)
        values["source_rank_variance"]=np.sum(np.where(pm>0,(rm-mean_rank)**2,0),axis=0)/np.maximum(pm.sum(0),1)
        values["dense_agreement"]=both.astype(np.float32);values["lexical_dense_agreement"]=((lexical["present"]>0)&(giga["present"]>0)).astype(np.float32);values["local_global_agreement"]=both.astype(np.float32)
        query_location=frame.search_location_id.iloc[q];distance=np.full(n,np.nan)
        if query_location in self.centroids.index:
            valid=np.isfinite(self.lat[candidates])&np.isfinite(self.lon[candidates])
            distance[valid]=haversine(self.centroids.loc[query_location,"lat"],self.centroids.loc[query_location,"lon"],self.lat[candidates][valid],self.lon[candidates][valid])
        missing=~np.isfinite(distance);clean=np.where(missing,0,distance).astype(np.float32)
        values["geo_distance_km"]=clean;values["geo_log_distance"]=np.log1p(clean);values["geo_missing"]=missing.astype(np.float32)
        for radius in (10,25,50,100):values[f"within_{radius}km"]=((distance<=radius)&~missing).astype(np.float32)
        base_names=base_feature_names(BASE_SOURCES)
        matrix=np.column_stack([values[name] for name in base_names]).astype(np.float32)
        # Geo_redirect входит в объединение, но четыре агрегата сохраняют семантику frozen v3.
        base_pm=np.vstack([parts[name]["present"] for name in BASE_SOURCES]);base_rm=np.vstack([parts[name]["rank"] for name in BASE_SOURCES])
        base_ordered=np.sort(np.where(base_pm>0,base_rm,np.inf),axis=0);base_mean=np.sum(np.where(base_pm>0,base_rm,0),axis=0)/np.maximum(base_pm.sum(0),1)
        replacements={"num_sources_present":base_pm.sum(0),"best_source_rank":np.where(np.isfinite(base_ordered[0]),base_ordered[0],201),"second_best_source_rank":np.where(np.isfinite(base_ordered[1]),base_ordered[1],201),"source_rank_variance":np.sum(np.where(base_pm>0,(base_rm-base_mean)**2,0),axis=0)/np.maximum(base_pm.sum(0),1)}
        for name,array in replacements.items():matrix[:,base_names.index(name)]=array
        return candidates,np.nan_to_num(matrix,nan=0,posinf=0,neginf=0),values,pm


def _tokens(value):
    return tuple(TOKEN_RE.findall(safe_text(value)))


def _ordered_coverage(query, field):
    if not query or not field:return 0.
    field_set=set(field);best=current=0
    for token in query:
        if token in field_set:current+=1;best=max(best,current)
        else:current=0
    return best/len(query)


def build_token_idf(items: pd.DataFrame):
    counts=Counter();n=len(items)
    for row in items.itertuples(index=False):
        text=" ".join(str(getattr(row,name,"")) for name in ("item_title_raw","item_infm_params_text","item_description_raw"))
        counts.update(set(_tokens(text)))
    return {token:math.log((n+1)/(count+1))+1 for token,count in counts.items()}


class FieldInteractionBuilder:
    def __init__(self, items: pd.DataFrame, idf: dict[str,float]):
        self.title=items.item_title_raw.fillna("").astype(str).to_numpy();self.params=items.item_infm_params_text.fillna("").astype(str).to_numpy();self.desc=items.item_description_raw.fillna("").astype(str).str.slice(0,2000).to_numpy();self.idf=idf

    @lru_cache(maxsize=120000)
    def _item(self,position):
        raw=(self.title[position],self.params[position],self.desc[position]);clean=tuple(safe_text(x) for x in raw);tok=tuple(_tokens(x) for x in clean);sets=tuple(set(x) for x in tok)
        return clean,tok,sets

    def build(self,query_row,positions):
        query=safe_text(query_row.search_query);qt=_tokens(query);qs=set(qt);qidf={x:self.idf.get(x,0.) for x in qs};rarest=max(qidf,key=qidf.get) if qidf else None;total=sum(qidf.values()) or 1.
        qnum={x for x in qs if x.isdigit()};qlatin={x for x in qs if re.search(r"[a-z]",x)};qalnum={x for x in qs if ALNUM_RE.fullmatch(x)}
        result=np.zeros((len(positions),len(FIELD_FEATURE_NAMES)),np.float32)
        for i,pos in enumerate(positions):
            clean,tok,sets=self._item(int(pos));coverages=[];offset=0
            for clean_f,tok_f,set_f in zip(clean,tok,sets):
                matched=qs&set_f;coverage=len(matched)/max(len(qs),1);coverages.append(coverage)
                result[i,offset:offset+7]=[coverage,len(matched)/max(len(set_f),1),len(matched)/max(len(qs|set_f),1),float(bool(query) and query in clean_f),float(bool(qs) and qs<=set_f),_ordered_coverage(qt,tok_f),len(matched)];offset+=7
            title_set,params_set,desc_set=sets;union=title_set|params_set|desc_set;nums={x for x in union if x.isdigit()};latin={x for x in union if re.search(r"[a-z]",x)};alnum={x for x in union if ALNUM_RE.fullmatch(x)};matched=qs&union
            result[i,21:]=[float(bool(query) and query==clean[0]),float(bool(clean[0]) and clean[0] in query),float(bool(query) and clean[0].startswith(query)),float(bool(qt) and qt[0] in title_set),float(bool(qt) and qt[-1] in title_set),float(rarest in title_set) if rarest else 0,float(rarest in params_set) if rarest else 0,float(rarest in desc_set) if rarest else 0,sum(qidf[x] for x in matched),sum(qidf[x] for x in matched)/total,float(bool(qnum)),float(bool(qnum&nums)),float(bool(qnum) and bool(nums) and not bool(qnum&nums)),float(bool(qlatin)),float(bool(qlatin&latin)),float(bool(qalnum&alnum)),float(coverages[0]>coverages[2]),float(coverages[1]>coverages[2]),float(coverages[2]>0 and coverages[0]==0 and coverages[1]==0),sum(x>0 for x in coverages),max(coverages),sorted(coverages)[-2]]
        return result


def geo_redirect_features(geo_cache, q, candidates, values, transition_summary, search_location):
    out=np.zeros((len(candidates),len(GEO_FEATURE_NAMES)),np.float32)
    for column,key in enumerate(("present","rank","reciprocal_rank","score")):
        out[:,column]=values[f"geo_redirect_{key}"]
    ranking=geo_cache["rankings"][q];valid=ranking>=0;indices=np.searchsorted(candidates,ranking[valid])
    out[indices,4]=geo_cache["destination_probability"][q,valid];out[indices,5]=geo_cache["redirect_rank"][q,valid]
    summary=transition_summary.get(search_location,{"top1_probability":0.,"entropy":0.});out[:,6]=summary["top1_probability"];out[:,7]=summary["entropy"]
    out[:,8]=(out[:,5]==1)&(out[:,0]>0);out[:,9]=(out[:,5]<=3)&(out[:,5]>0);out[:,10]=(out[:,5]<=5)&(out[:,5]>0)
    return out


def build_final_matrix(base_builder,field_builder,geo_cache,transition_summary,split,q):
    candidates,base,values,_=base_builder.build(split,q)
    geo=geo_redirect_features(geo_cache,q,candidates,values,transition_summary,base_builder.frames[split].search_location_id.iloc[q])
    fields=field_builder.build(base_builder.frames[split].iloc[q],candidates)
    matrix=np.column_stack([base,geo,fields]).astype(np.float32,copy=False)
    assert matrix.shape[1]==185
    return candidates,matrix
