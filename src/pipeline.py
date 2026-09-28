from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ALL_SOURCES, BASE_SOURCES, PipelineConfig
from .data import build_internal, build_ranker_train, build_validation_and_corpus, load_data, validate_data_dir
from .features import (
    CandidateFeatureBuilder, FieldInteractionBuilder, FINAL_FEATURE_NAMES,
    build_final_matrix, build_token_idf,
)
from .geo import behavioral_geo_candidates, build_transition_table
from .ranking import retrieval100_selection, train_yetirank, validate_submission
from .retrieval import (
    encode_giga, exact_dense_topk, geo50_dense, load_source, restricted_dense,
    restricted_sparse, save_source, sparse_topk,
)
from .utils import atomic_json, normalize_text


def _source_paths(folder: Path):
    return {name:folder/f"{name.lower()}_top.npz" for name in BASE_SOURCES}


def _load_npz_dict(path: Path):
    with np.load(path,allow_pickle=False) as payload:return {k:payload[k] for k in payload.files}


def _transition_summary_map(summary: pd.DataFrame):
    return {row.search_location_id:{"top1_probability":row.top1_probability,"entropy":row.entropy} for row in summary.itertuples(index=False)}


def build_candidate_cache(config: PipelineConfig, scope: str, frame: pd.DataFrame, items: pd.DataFrame,
                          train: pd.DataFrame, transition_table: pd.DataFrame):
    """Строит девять источников кандидатов для одного корпуса."""
    folder=config.cache_dir/"v4"/scope;folder.mkdir(parents=True,exist_ok=True);paths=_source_paths(folder)
    item_embeddings_path=folder/"item_embeddings.npy";query_embeddings_path=folder/"query_embeddings.npy"
    if item_embeddings_path.exists():item_embeddings=np.load(item_embeddings_path,mmap_mode="r")
    else:
        item_embeddings=encode_giga(
            items.item_title_raw.fillna(""), config.embedding_model,
            revision=config.embedding_revision,
        )
        np.save(item_embeddings_path,item_embeddings)
    if query_embeddings_path.exists():query_embeddings=np.load(query_embeddings_path,mmap_mode="r")
    else:
        query_embeddings=encode_giga(
            frame.search_query.fillna(""), config.embedding_model, config.query_prefix,
            revision=config.embedding_revision,
        )
        np.save(query_embeddings_path,query_embeddings)
    locations=items.item_location_id.to_numpy();query_locations=frame.search_location_id.to_numpy()
    if not paths["Lexical"].exists():
        vectorizer=TfidfVectorizer(dtype=np.float32,min_df=3,max_features=60000,ngram_range=(1,2),sublinear_tf=True)
        im=vectorizer.fit_transform(normalize_text(items.item_title_raw));qm=vectorizer.transform(frame.search_query.fillna(""));save_source(paths["Lexical"],*sparse_topk(qm,im,200))
    if not paths["Description"].exists() or not paths["Description_local"].exists():
        vectorizer=TfidfVectorizer(dtype=np.float32,min_df=3,max_features=100000,sublinear_tf=True)
        im=vectorizer.fit_transform(normalize_text(items.item_description_raw).str.slice(0,2000));qm=vectorizer.transform(frame.search_query.fillna(""))
        save_source(paths["Description"],*sparse_topk(qm,im,200));save_source(paths["Description_local"],*restricted_sparse(qm,im,query_locations,locations,100))
    if not paths["Title_params"].exists():
        docs=items.item_title_raw.fillna("").astype(str).str.cat(normalize_text(items.item_infm_params_text).str.slice(0,1200),sep=" ")
        vectorizer=TfidfVectorizer(dtype=np.float32,min_df=2,max_features=100000,sublinear_tf=True);im=vectorizer.fit_transform(docs);qm=vectorizer.transform(frame.search_query.fillna(""));save_source(paths["Title_params"],*sparse_topk(qm,im,200))
    if not paths["Giga"].exists():save_source(paths["Giga"],*exact_dense_topk(query_embeddings,item_embeddings,200))
    if not paths["Giga_local_deep"].exists():
        deep=restricted_dense(query_embeddings,item_embeddings,query_locations,locations,200)
        save_source(paths["Giga_local_deep"],*deep)
    else:
        deep=load_source(paths["Giga_local_deep"])
    if not paths["Giga_local"].exists():
        save_source(paths["Giga_local"],deep[0][:,:100],deep[1][:,:100])
    if not paths["Giga_geo50"].exists():save_source(paths["Giga_geo50"],*geo50_dense(frame,items,query_embeddings,item_embeddings,100))
    geo_path=folder/"geo_redirect_top100.npz"
    if not geo_path.exists():
        geo=behavioral_geo_candidates(frame,items,query_embeddings,item_embeddings,transition_table,100);np.savez_compressed(geo_path,**geo)
    sources={scope:{name:load_source(path) for name,path in paths.items()}}
    geo=_load_npz_dict(geo_path);sources[scope]["Geo_redirect"]=(geo["rankings"],geo["scores"])
    return sources,geo


def _feature_builders(items,frame,sources,scope,idf):
    vectorizer=TfidfVectorizer(dtype=np.float32,min_df=3,max_features=60000,ngram_range=(1,2),sublinear_tf=True)
    vectorizer.fit(normalize_text(items.item_infm_params_text))
    return (
        CandidateFeatureBuilder(items,sources,{scope:frame},vectorizer,ALL_SOURCES),
        FieldInteractionBuilder(items,idf),
    )


def build_training_matrix(config,train,train_queries,corpus,transition_table,transition_summary):
    output=config.cache_dir/"v4"/"training_matrix.npz"
    if output.exists():return _load_npz_dict(output)
    sources,geo=build_candidate_cache(config,"train",train_queries,corpus,train,transition_table)
    idf_path=config.cache_dir/"v4"/"token_idf.joblib"
    if idf_path.exists():idf=joblib.load(idf_path)
    else:idf=build_token_idf(corpus);joblib.dump(idf,idf_path,compress=3)
    base,fields=_feature_builders(corpus,train_queries,sources,"train",idf)
    summary=_transition_summary_map(transition_summary);item_index=pd.Index(corpus.item_id.astype(str))
    sampler=joblib.load(config.artifacts_dir/"retrieval100_sampler.joblib")
    blocks,labels,groups=[],[],[]
    for q in range(len(train_queries)):
        candidates,matrix=build_final_matrix(base,fields,geo,summary,"train",q)
        positive=item_index.get_indexer([str(train_queries.item_id.iloc[q])])[0];y=(candidates==positive).astype(np.int8)
        if not y.any():continue
        _,_,values,present=base.build("train",q)
        legacy=np.column_stack([values[name] for name in sampler["features"]]).astype(np.float32)
        score=sampler["model"].predict_proba(sampler["scaler"].transform(legacy))[:,1]
        chosen=retrieval100_selection(y,score,present)
        blocks.append(matrix[chosen]);labels.append(y[chosen]);groups.append(np.full(len(chosen),q,np.int32))
    result={"X":np.concatenate(blocks),"y":np.concatenate(labels),"g":np.concatenate(groups)}
    np.savez_compressed(output,**result);return result


def _model(config,training=None,retrain=False):
    path=config.artifacts_dir/"final_v4_e4_yetirank.cbm"
    if path.exists() and not retrain:
        model=CatBoostRanker();model.load_model(path);return model
    if training is None:raise ValueError("Для переобучения нужна training matrix")
    model=train_yetirank(training["X"],training["y"],training["g"],config.catboost_params)
    model.save_model(path);return model


def _consolidate_cached_chunks(config,queries,items):
    folder=config.cache_dir/"final_v4_e4"/"benchmark_chunks"
    files=sorted(folder.glob("q_*.npz"))
    if not files:return None
    query_index=[];positions=[]
    for path in files:
        with np.load(path,allow_pickle=False) as part:
            query_index.append(part["query_index"]);positions.append(part["top_positions"])
    query_index=np.concatenate(query_index);positions=np.concatenate(positions)
    if not np.array_equal(query_index,np.arange(len(queries))):return None
    item_ids=items.item_id.astype(str).to_numpy()
    return pd.DataFrame({"query_id":queries.query_id.astype(str),"answer":[" ".join(item_ids[row]) for row in positions]})


def predict_benchmark(config,model,train,queries,items,transition_table,transition_summary):
    cached=_consolidate_cached_chunks(config,queries,items)
    if cached is not None:return cached,"cached benchmark chunks"
    sources,geo=build_candidate_cache(config,"benchmark",queries,items,train,transition_table)
    idf_path=config.cache_dir/"v4"/"token_idf.joblib"
    if idf_path.exists():idf=joblib.load(idf_path)
    else:
        # IDF обучается только по train-корпусу; вызов без кэша происходит после подготовки train.
        raise FileNotFoundError("Не найден token_idf.joblib: сначала подготовьте training matrix")
    base,fields=_feature_builders(items,queries,sources,"benchmark",idf);summary=_transition_summary_map(transition_summary);item_ids=items.item_id.astype(str).to_numpy();answers=[]
    for q in range(len(queries)):
        candidates,matrix=build_final_matrix(base,fields,geo,summary,"benchmark",q);scores=model.predict(matrix);order=np.lexsort((candidates,-scores))[:config.top_k];answers.append(" ".join(item_ids[candidates[order]]))
    return pd.DataFrame({"query_id":queries.query_id.astype(str),"answer":answers}),"fresh benchmark inference"


def evaluate_split(config,scope,frame,items,train,model,transitions,summary):
    """Считает single-positive Hit@50 и coverage без чтения готового отчёта."""
    sources,geo=build_candidate_cache(config,scope,frame,items,train,transitions)
    idf_path=config.cache_dir/"v4"/"token_idf.joblib"
    idf=joblib.load(idf_path) if idf_path.exists() else build_token_idf(items)
    if not idf_path.exists():joblib.dump(idf,idf_path,compress=3)
    base,fields=_feature_builders(items,frame,sources,scope,idf);summary_map=_transition_summary_map(summary)
    item_index=pd.Index(items.item_id.astype(str));positive=item_index.get_indexer(frame.item_id.astype(str));hits=np.zeros(len(frame),bool);coverage=np.zeros(len(frame),bool)
    for q in range(len(frame)):
        candidates,matrix=build_final_matrix(base,fields,geo,summary_map,scope,q);coverage[q]=(candidates==positive[q]).any();scores=model.predict(matrix);order=np.lexsort((candidates,-scores))[:50];hits[q]=(candidates[order]==positive[q]).any()
    return {"hit_at_50":float(hits.mean()),"candidate_coverage":float(coverage.mean()),"queries":len(frame)}


def check_environment(config):
    paths=validate_data_dir(config.data_dir);checks={"data":{k:str(v) for k,v in paths.items()},"model":str(config.artifacts_dir/"final_v4_e4_yetirank.cbm"),"model_exists":(config.artifacts_dir/"final_v4_e4_yetirank.cbm").exists(),"feature_count":len(FINAL_FEATURE_NAMES),"cached_benchmark_chunks":len(list((config.cache_dir/"final_v4_e4"/"benchmark_chunks").glob("q_*.npz")))}
    return checks


def reproduce(config: PipelineConfig,retrain: bool=False):
    started=time.time();train,queries,items=load_data(config.data_dir)
    transition_path=config.cache_dir/"v4"/"transition.parquet";summary_path=config.cache_dir/"v4"/"transition_summary.parquet"
    split_dir=config.cache_dir/"v4"/"splits";split_dir.mkdir(parents=True,exist_ok=True)
    dev_path,corpus_path,train_path=split_dir/"development.pkl",split_dir/"corpus.pkl",split_dir/"ranker_train.pkl"
    development=corpus=ranker_train=None
    if retrain or not (transition_path.exists() and summary_path.exists()):
        if all(p.exists() for p in (dev_path,corpus_path,train_path)):
            development=pd.read_pickle(dev_path);corpus=pd.read_pickle(corpus_path);ranker_train=pd.read_pickle(train_path)
        else:
            development,_,corpus=build_validation_and_corpus(train,config.seed)
            ranker_train=build_ranker_train(train,development,corpus,seed=config.seed)
            development.to_pickle(dev_path);corpus.to_pickle(corpus_path);ranker_train.to_pickle(train_path)
    if transition_path.exists() and summary_path.exists():
        transitions=pd.read_parquet(transition_path);summary=pd.read_parquet(summary_path)
    else:
        transitions,summary=build_transition_table(ranker_train,corpus)
        transition_path.parent.mkdir(parents=True,exist_ok=True)
        transitions.to_parquet(transition_path,index=False);summary.to_parquet(summary_path,index=False)
        atomic_json(config.cache_dir/"v4"/"transition_manifest.json",{
            "history":"ranker_train only","rows":len(ranker_train),"seed":config.seed,
        })
    idf_path=config.cache_dir/"v4"/"token_idf.joblib"
    if not idf_path.exists():
        if corpus is None:
            if all(p.exists() for p in (dev_path,corpus_path,train_path)):
                development=pd.read_pickle(dev_path);corpus=pd.read_pickle(corpus_path);ranker_train=pd.read_pickle(train_path)
            else:
                development,_,corpus=build_validation_and_corpus(train,config.seed)
                ranker_train=build_ranker_train(train,development,corpus,seed=config.seed)
                development.to_pickle(dev_path);corpus.to_pickle(corpus_path);ranker_train.to_pickle(train_path)
        joblib.dump(build_token_idf(corpus),idf_path,compress=3)
    training=None
    local_metrics=None
    if retrain:
        training=build_training_matrix(config,train,ranker_train,corpus,transitions,summary)
    model=_model(config,training,retrain);submission,mode=predict_benchmark(config,model,train,queries,items,transitions,summary)
    if retrain:
        internal=build_internal(train,development,ranker_train,corpus,seed=config.seed)
        local_metrics={
            "internal":evaluate_split(config,"internal",internal,corpus,train,model,transitions,summary),
            "development":evaluate_split(config,"development",development,corpus,train,model,transitions,summary),
        }
    config.output.parent.mkdir(parents=True,exist_ok=True);submission.to_csv(config.output,index=False,encoding="utf-8")
    reloaded=pd.read_csv(config.output,dtype=str);validation=validate_submission(reloaded,queries,items)
    report={"mode":mode,"runtime_seconds":time.time()-started,"output":str(config.output),"validation":validation,"features":len(FINAL_FEATURE_NAMES),"local_metrics":local_metrics}
    atomic_json(config.cache_dir/"v4"/"last_run.json",report);return report
