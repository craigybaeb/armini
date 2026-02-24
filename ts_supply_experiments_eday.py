#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supply forecasting experiments (aggregate series) with E-day style handling.

Adds:
- --e_day_mode {off,last_diff,diff_then_sum}
  * last_diff: hourly = target.resample('H').last().diff().clip(lower=0)
  * diff_then_sum: native-frequency within-day diffs grouped by local day, then hourly sum
- Optional timestamp synthesis for files without a time column:
  --synthesize_time_start "2023-01-01 00:00:00" --synthesize_time_freq "30S" --reset_tz Europe/London
"""

import os, json, argparse, warnings, csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np, pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import tensorflow as tf
    from tensorflow.keras import Model, Input, Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, Conv1D, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except Exception:
    HAS_TF = False

@dataclass
class Config:
    csv_path: str
    out_dir: str = "outputs_supply_eday"
    time_col: str = None
    target_col: str = None
    on_bad_lines: str = "skip"   # 'skip'|'warn'|'error'
    resample_rule: str = "1H"
    test_fraction_time: float = 0.2
    blackout_ranges: Optional[List[Tuple[str, str]]] = (("2024-05-02","2024-05-10"),)
    max_interp_minutes: int = 60
    lags: Tuple[int,...] = (1,2,3,6,12,24)
    roll_windows: Tuple[int,...] = (3,6,12,24)
    gbm_grid: Dict = None
    knn_grid: Dict = None
    rnn_grid: Dict = None
    rnn_epochs: int = 40
    rnn_patience: int = 6
    tcn_grid: Dict = None
    tcn_epochs: int = 30
    tcn_patience: int = 5
    transformer_grid: Dict = None
    transformer_epochs: int = 30
    transformer_patience: int = 5
    # E-day specifics
    e_day_mode: str = "last_diff"    # 'off'|'last_diff'|'diff_then_sum'
    reset_tz: str = "Europe/London"
    synth_start: Optional[str] = None
    synth_freq: Optional[str] = None

    def __post_init__(self):
        if self.gbm_grid is None: self.gbm_grid={"learning_rate":[0.05,0.1],"max_iter":[300,600],"max_leaf_nodes":[31,63]}
        if self.knn_grid is None: self.knn_grid={"n_neighbors":[5,15,30],"weights":["uniform","distance"]}
        if self.rnn_grid is None: self.rnn_grid={"seq_len":[12],"units1":[64],"units2":[32],"dropout":[0.2],"batch_size":[256]}
        if self.tcn_grid is None: self.tcn_grid={"seq_len":[12],"filters":[32,64],"kernel_size":[3,5],"dilations":[[1,2,4,8]],"dropout":[0.2],"batch_size":[256]}
        if self.transformer_grid is None: self.transformer_grid={"seq_len":[12],"d_model":[64],"num_heads":[2,4],"ff_units":[128],"dropout":[0.1],"batch_size":[256]}

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def smape(y,yhat,eps=1e-6): d=np.abs(y)+np.abs(yhat); d=np.where(d<eps,eps,d); return 100.0*np.mean(2.0*np.abs(yhat-y)/d)
def mape(y,yhat,eps=1e-6): d=np.where(np.abs(y)<eps,eps,np.abs(y)); return 100.0*np.mean(np.abs((y-yhat)/d))
def time_split_cutoff(df,col,frac): return df[col].quantile(1.0-frac)
from pandas.tseries.frequencies import to_offset
def steps_for_period(rule,period): off=to_offset(rule); delta=getattr(off,'delta',None) or pd.Timedelta('1H'); return max(1,int(pd.Timedelta(period)/delta))

def read_csv_robust(path: str, on_bad_lines: str = "skip") -> pd.DataFrame:
    try:
        return pd.read_csv(path, engine="c", low_memory=False)
    except Exception:
        pass
    try:
        return pd.read_csv(
            path,
            engine="python",
            sep=None,
            quotechar='"',
            escapechar="\\",
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines=on_bad_lines,
        )
    except Exception:
        return pd.read_csv(
            path,
            engine="python",
            sep=",",
            quotechar='"',
            escapechar="\\",
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines=on_bad_lines,
        )

CAND_TIME=["datetime","date_time","timestamp","time"]
CAND_TARGETS=["e-day/daily power generation","grid side total power","drid side total power","grid total power","load total power","total power"]

def detect_time_and_target(df,cfg):
    tcol=cfg.time_col
    if tcol is None and cfg.synth_start is None:
        for c in df.columns:
            cl=c.strip().lower()
            if any(ct in cl for ct in CAND_TIME): tcol=c; break
    ycol=cfg.target_col
    if ycol is None:
        cols_l={c.lower():c for c in df.columns}
        for cand in CAND_TARGETS:
            if cand in cols_l: ycol=cols_l[cand]; break
        if ycol is None:
            for c in df.columns:
                cl=c.strip().lower()
                if "total" in cl and "power" in cl: ycol=c; break
    return tcol,ycol

def load_supply(csv_path,cfg):
    df=read_csv_robust(csv_path, cfg.on_bad_lines)
    tcol,ycol=detect_time_and_target(df,cfg)
    # Timestamp handling
    if tcol is None and cfg.synth_start:
        # Synthesize a timestamp column
        n=len(df)
        ts = pd.date_range(start=cfg.synth_start, periods=n, freq=(cfg.synth_freq or "30S"), tz=cfg.reset_tz)
        df.insert(0, "time_stamp", ts.tz_convert("UTC"))
    else:
        df[tcol]=pd.to_datetime(df[tcol],errors='coerce',utc=True)
        df=df.rename(columns={tcol:"time_stamp"})
    # Target handling
    if ycol is not None and ycol in df.columns:
        df=df.rename(columns={ycol:"target"})
    elif ycol is not None:
        df["target"]=pd.to_numeric(df[ycol], errors="coerce")  # safety
    # Keep numeric features as potential exogenous (optional)
    num_cols=df.select_dtypes(include=[np.number]).columns.tolist()
    keep=["time_stamp"]+list(dict.fromkeys(["target"]+[c for c in num_cols if c!="target"]))
    df=df[keep].dropna(subset=["time_stamp"]).sort_values("time_stamp").reset_index(drop=True)
    return df,ycol

def hourly_from_eday_last_diff(df: pd.DataFrame, reset_tz: str="Europe/London") -> pd.DataFrame:
    g=df.copy().set_index("time_stamp").sort_index()
    # Work in local time to align the hourly boundary
    g = g.tz_convert(reset_tz)
    hourly_last = g["target"].resample("1H").last()
    hourly_inc = hourly_last.diff().clip(lower=0)
    out = pd.DataFrame({"target": hourly_inc}).dropna()
    out.index = out.index.tz_convert("UTC")
    out = out.reset_index().rename(columns={"index":"time_stamp", "time_stamp":"time_stamp"})
    return out

def convert_daily_resets_to_flow(df: pd.DataFrame, target_col: str = "target", reset_tz: str = "Europe/London") -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy()
    ts_local = out["time_stamp"].dt.tz_convert(reset_tz)
    day_key = ts_local.dt.floor("D")
    delta = out[target_col].groupby(day_key).diff().fillna(out[target_col]).clip(lower=0)
    out[target_col] = delta
    return out

def resample_hourly_sum_target(df, rule="1H", max_interp_minutes=60):
    g=df.set_index("time_stamp").sort_index()
    agg={c:"mean" for c in g.columns}
    if "target" in agg: agg["target"]="sum"
    g=g.resample(rule).agg(agg)
    if "target" in g.columns:
        g["target"]=g["target"].interpolate(method="time",limit=max_interp_minutes,limit_direction="both")
    for c in g.columns:
        if c!="target":
            g[c]=g[c].ffill(limit=max_interp_minutes).bfill(limit=max_interp_minutes)
    return g.reset_index().dropna(subset=["target"])

def remove_blackouts(df, blackout_ranges):
    if not blackout_ranges: return df
    out=df.copy()
    if out["time_stamp"].dt.tz is None: out["time_stamp"]=pd.to_datetime(out["time_stamp"], utc=True)
    tz=out["time_stamp"].dt.tz
    for s,e in blackout_ranges:
        s=pd.to_datetime(s,utc=True).tz_convert(tz); e=pd.to_datetime(e,utc=True).tz_convert(tz)
        mask=(out["time_stamp"]>=s)&(out["time_stamp"]<e); out=out.loc[~mask]
    return out.reset_index(drop=True)

def add_time_features(df):
    df=df.copy(); df["hour"]=df["time_stamp"].dt.hour; df["dayofweek"]=df["time_stamp"].dt.dayofweek; df["is_weekend"]=df["dayofweek"].isin([5,6]).astype(int); return df

def add_lag_roll(df,lags,rolls):
    g=df.copy()
    for L in lags: g[f"target_lag_{L}"]=g["target"].shift(L)
    for W in rolls:
        g[f"target_roll_mean_{W}"]=g["target"].rolling(W,min_periods=max(1,W//2)).mean()
        g[f"target_roll_std_{W}"]=g["target"].rolling(W,min_periods=max(1,W//2)).std()
    g=add_time_features(g); return g.dropna().reset_index(drop=True)

def build_train_test(df,frac): c=time_split_cutoff(df,"time_stamp",frac); return df[df["time_stamp"]<=c].copy(), df[df["time_stamp"]>c].copy(), c

def forecast_naive_last_agg(test_df,train_df):
    all_df=pd.concat([train_df,test_df],ignore_index=True).sort_values("time_stamp"); all_df["yhat"]=all_df["target"].shift(1)
    return all_df[all_df["time_stamp"].isin(test_df["time_stamp"])].set_index("time_stamp")["yhat"].sort_index()

def forecast_naive_seasonal_agg(test_df,train_df,season):
    all_df=pd.concat([train_df,test_df],ignore_index=True).sort_values("time_stamp"); all_df["yhat"]=all_df["target"].shift(season)
    return all_df[all_df["time_stamp"].isin(test_df["time_stamp"])].set_index("time_stamp")["yhat"].sort_index()

def select_exogenous(df,max_cols=12):
    cand=[c for c in df.columns if c not in ["time_stamp","target"] and not c.startswith("target_")]
    nun=df[cand].nunique() if len(cand)>0 else pd.Series(dtype=int)
    cand=[c for c in cand if nun.get(c,0)>10]
    if len(cand)==0: return []
    corr=df[cand+["target"]].corr(numeric_only=True)["target"].drop("target").abs().sort_values(ascending=False)
    return list(corr.head(max_cols).index)

def train_eval_gbm(train_df,test_df,cfg):
    timef=["hour","dayofweek","is_weekend"]
    lag=[c for c in train_df.columns if c.startswith("target_lag_") or c.startswith("target_roll_")]
    exog=select_exogenous(train_df, max_cols=12)
    feats=(exog+timef+lag) if exog else (timef+lag)
    Xtr=train_df[feats].astype(np.float32); Xte=test_df[feats].astype(np.float32); ytr=train_df["target"].values
    cut=train_df["time_stamp"].quantile(0.9); tr=train_df["time_stamp"]<=cut; va=train_df["time_stamp"]>cut
    best=None; br=np.inf; bm=None
    for lr in cfg.gbm_grid["learning_rate"]:
        for it in cfg.gbm_grid["max_iter"]:
            for mln in cfg.gbm_grid["max_leaf_nodes"]:
                m=HistGradientBoostingRegressor(learning_rate=lr,max_iter=it,max_leaf_nodes=mln,early_stopping=True,validation_fraction=0.1,random_state=42)
                m.fit(Xtr[tr],ytr[tr]); rmse=np.sqrt(mean_squared_error(ytr[va], m.predict(Xtr[va])))
                if rmse<br: br,best,bm=rmse, {"learning_rate":lr,"max_iter":it,"max_leaf_nodes":mln}, m
    bm.fit(Xtr,ytr); yhat=bm.predict(Xte)
    return pd.Series(yhat,index=test_df["time_stamp"]), {"best_params":best,"n_features":Xtr.shape[1],"model":bm,"feature_cols":feats,"exog":exog}

def train_eval_knn(train_df,test_df,cfg):
    timef=["hour","dayofweek","is_weekend"]; lag=[c for c in train_df.columns if c.startswith("target_lag_") or c.startswith("target_roll_")]; exog=select_exogenous(train_df, max_cols=12)
    feats=(exog+timef+lag) if exog else (timef+lag); Xtr=train_df[feats].astype(np.float32); Xte=test_df[feats].astype(np.float32); ytr=train_df["target"].values
    sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    cut=train_df["time_stamp"].quantile(0.9); tr=train_df["time_stamp"]<=cut; va=train_df["time_stamp"]>cut
    best=None; br=np.inf; bm=None
    for k in cfg.knn_grid["n_neighbors"]:
        for w in cfg.knn_grid["weights"]:
            m=KNeighborsRegressor(n_neighbors=k,weights=w); m.fit(Xtr_s[tr],ytr[tr]); rmse=np.sqrt(mean_squared_error(ytr[va],m.predict(Xtr_s[va])))
            if rmse<br: br,best,bm=rmse,{"n_neighbors":k,"weights":w},m
    bm.fit(Xtr_s,ytr); yhat=bm.predict(Xte_s)
    return pd.Series(yhat,index=test_df["time_stamp"]), {"best_params":best,"scaler":sc,"model":bm,"feature_cols":feats,"exog":exog}

def build_tf_datasets_agg(train_df,feats,L,sc,val_cut):
    def _gen(split):
        g=train_df.sort_values("time_stamp")
        if len(g)<=L: return
        X=sc.transform(g[feats].astype(np.float32)); y=g["target"].astype(np.float32).values; ts=g["time_stamp"]
        for i in range(len(g)-L):
            end_ts=ts.iloc[i+L]; is_tr=(end_ts<=val_cut)
            if (split=="train" and is_tr) or (split=="val" and not is_tr):
                last=np.array([y[i+L-1]],dtype=np.float32)
                yield {"seq_in":X[i:i+L,:],"last_in":last}, float(y[i+L])
    sig=({"seq_in":tf.TensorSpec(shape=(L,len(feats)),dtype=tf.float32),"last_in":tf.TensorSpec(shape=(1,),dtype=tf.float32)}, tf.TensorSpec(shape=(),dtype=tf.float32))
    return tf.data.Dataset.from_generator(lambda:_gen("train"),output_signature=sig), tf.data.Dataset.from_generator(lambda:_gen("val"),output_signature=sig)

def build_sequences_for_predictions_agg(train_df,test_df,feats,L,sc):
    base=pd.concat([train_df,test_df],ignore_index=True).sort_values("time_stamp"); X=sc.transform(base[feats].astype(np.float32)); y=base["target"].astype(np.float32).values
    test_ts=set(test_df["time_stamp"])
    for i in range(len(base)-L):
        t=i+L; ts=base.iloc[t]["time_stamp"]
        if ts in test_ts: yield ts, X[i:t,:], np.array([y[t-1]],dtype=np.float32)

def build_rnn_model_agg(L,n_features,u1,u2,drop):
    inp=Input(shape=(L,n_features),name="seq_in"); x=GRU(u1,return_sequences=True)(inp); x=Dropout(drop)(x); x=GRU(u2)(x); x=Dropout(drop)(x)
    last=Input(shape=(1,),dtype="float32",name="last_in"); delta=Dense(1)(x); out=tf.keras.layers.Add()([last,delta])
    m=Model(inputs=[inp,last],outputs=out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,clipnorm=1.0),loss="mse"); return m

def train_eval_rnn_agg(train_df,test_df,cfg,out_dir):
    timef=["hour","dayofweek","is_weekend"]; feats=["target"]+timef; sc=StandardScaler().fit(train_df[feats].astype(np.float32)); cut=train_df["time_stamp"].quantile(0.9)
    best_m=None; best_c=None; br=np.inf
    for L in cfg.rnn_grid["seq_len"]:
        dtr,dva=build_tf_datasets_agg(train_df,feats,L,sc,cut)
        for u1 in cfg.rnn_grid["units1"]:
            for u2 in cfg.rnn_grid["units2"]:
                for dr in cfg.rnn_grid["dropout"]:
                    for bs in cfg.rnn_grid["batch_size"]:
                        m=build_rnn_model_agg(L,len(feats),u1,u2,dr)
                        cbs=[EarlyStopping(monitor="val_loss",patience=cfg.rnn_patience,restore_best_weights=True,verbose=0), ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=max(2,cfg.rnn_patience//2),min_lr=1e-5,verbose=0)]
                        m.fit(dtr.batch(bs).prefetch(tf.data.AUTOTUNE),validation_data=dva.batch(bs).prefetch(tf.data.AUTOTUNE),epochs=cfg.rnn_epochs,verbose=0,callbacks=cbs)
                        rmse=float(np.sqrt(m.evaluate(dva.batch(bs),verbose=0)))
                        if rmse<br: br,best_c,best_m=rmse,{"seq_len":L,"units1":u1,"units2":u2,"dropout":dr,"batch_size":bs},m
    idx=[]; vals=[]
    for ts, X_seq, last_in in build_sequences_for_predictions_agg(train_df,test_df,feats,best_c["seq_len"],sc):
        y=float(best_m.predict([X_seq.reshape(1,*X_seq.shape), last_in.reshape(1,1)], verbose=0).ravel()[0]); idx.append(ts); vals.append(y)
    return pd.Series(vals,index=pd.Index(idx,name="time_stamp")).sort_index(), {"best_params":best_c,"feature_cols":feats,"scaler":sc,"model":best_m}

def build_tcn_model_agg(L,n_features,filters,ks,dils,drop):
    inp=Input(shape=(L,n_features),name="seq_in"); x=inp
    for d in dils:
        res=x; x=Conv1D(filters=filters,kernel_size=ks,padding="causal",dilation_rate=d,activation="relu")(x); x=Dropout(drop)(x)
        x=Conv1D(filters=filters,kernel_size=ks,padding="causal",dilation_rate=d,activation="relu")(x)
        if res.shape[-1]!=x.shape[-1]: res=Conv1D(filters=filters,kernel_size=1,padding="same")(res)
        x=x+res
    x=GlobalAveragePooling1D()(x)
    last=Input(shape=(1,),dtype="float32",name="last_in"); delta=Dense(1)(x); out=tf.keras.layers.Add()([last,delta])
    m=Model(inputs=[inp,last],outputs=out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,clipnorm=1.0),loss="mse"); return m

def train_eval_tcn_agg(train_df,test_df,cfg,out_dir):
    timef=["hour","dayofweek","is_weekend"]; feats=["target"]+timef; sc=StandardScaler().fit(train_df[feats].astype(np.float32)); cut=train_df["time_stamp"].quantile(0.9)
    best_m=None; best_c=None; br=np.inf
    for L in cfg.tcn_grid["seq_len"]:
        dtr,dva=build_tf_datasets_agg(train_df,feats,L,sc,cut)
        for f in cfg.tcn_grid["filters"]:
            for ks in cfg.tcn_grid["kernel_size"]:
                for dils in cfg.tcn_grid["dilations"]:
                    for dr in cfg.tcn_grid["dropout"]:
                        for bs in cfg.tcn_grid["batch_size"]:
                            m=build_tcn_model_agg(L,len(feats),f,ks,dils,dr)
                            cbs=[EarlyStopping(monitor="val_loss",patience=cfg.tcn_patience,restore_best_weights=True,verbose=0), ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=max(2,cfg.tcn_patience//2),min_lr=1e-5,verbose=0)]
                            m.fit(dtr.batch(bs).prefetch(tf.data.AUTOTUNE),validation_data=dva.batch(bs).prefetch(tf.data.AUTOTUNE),epochs=cfg.tcn_epochs,verbose=0,callbacks=cbs)
                            rmse=float(np.sqrt(m.evaluate(dva.batch(bs),verbose=0)))
                            if rmse<br: br,best_c,best_m=rmse,{"seq_len":L,"filters":f,"kernel_size":ks,"dilations":dils,"dropout":dr,"batch_size":bs},m
    idx=[]; vals=[]
    for ts, X_seq, last_in in build_sequences_for_predictions_agg(train_df,test_df,feats,best_c["seq_len"],sc):
        y=float(best_m.predict([X_seq.reshape(1,*X_seq.shape), last_in.reshape(1,1)], verbose=0).ravel()[0]); idx.append(ts); vals.append(y)
    return pd.Series(vals,index=pd.Index(idx,name="time_stamp")).sort_index(), {"best_params":best_c,"feature_cols":feats,"scaler":sc,"model":best_m}

def build_transformer_model_agg(L,n_features,d_model,heads,ffu,drop):
    inp=Input(shape=(L,n_features),name="seq_in"); x=Dense(d_model)(inp)
    pos=np.arange(L)[:,None]; i=np.arange(d_model)[None,:]; rate=1/np.power(10000,(2*(i//2))/np.float32(d_model)); ang=pos*rate
    pe=np.zeros((L,d_model)); pe[:,0::2]=np.sin(ang[:,0::2]); pe[:,1::2]=np.cos(ang[:,1::2]); pe=tf.convert_to_tensor(pe,dtype=tf.float32)
    x=x+pe; att=MultiHeadAttention(num_heads=heads,key_dim=max(1,d_model//max(1,heads)))(x,x); x=LayerNormalization()(x+att)
    ff=Sequential([Dense(ffu,activation="relu"),Dropout(drop),Dense(d_model)]); x=LayerNormalization()(x+ff(x)); x=GlobalAveragePooling1D()(x)
    last=Input(shape=(1,),dtype="float32",name="last_in"); delta=Dense(1)(x); out=tf.keras.layers.Add()([last,delta])
    m=Model(inputs=[inp,last],outputs=out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,clipnorm=1.0),loss="mse"); return m

def train_eval_transformer_agg(train_df,test_df,cfg,out_dir):
    timef=["hour","dayofweek","is_weekend"]; feats=["target"]+timef; sc=StandardScaler().fit(train_df[feats].astype(np.float32)); cut=train_df["time_stamp"].quantile(0.9)
    best_m=None; best_c=None; br=np.inf
    for L in cfg.transformer_grid["seq_len"]:
        dtr,dva=build_tf_datasets_agg(train_df,feats,L,sc,cut)
        for dm in cfg.transformer_grid["d_model"]:
            for h in cfg.transformer_grid["num_heads"]:
                for ff in cfg.transformer_grid["ff_units"]:
                    for dr in cfg.transformer_grid["dropout"]:
                        for bs in cfg.transformer_grid["batch_size"]:
                            m=build_transformer_model_agg(L,len(feats),dm,h,ff,dr)
                            cbs=[EarlyStopping(monitor="val_loss",patience=cfg.transformer_patience,restore_best_weights=True,verbose=0), ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=max(2,cfg.transformer_patience//2),min_lr=1e-5,verbose=0)]
                            m.fit(dtr.batch(bs).prefetch(tf.data.AUTOTUNE),validation_data=dva.batch(bs).prefetch(tf.data.AUTOTUNE),epochs=cfg.transformer_epochs,verbose=0,callbacks=cbs)
                            rmse=float(np.sqrt(m.evaluate(dva.batch(bs),verbose=0)))
                            if rmse<br: br,best_c,best_m=rmse,{"seq_len":L,"d_model":dm,"num_heads":h,"ff_units":ff,"dropout":dr,"batch_size":bs},m
    idx=[]; vals=[]
    for ts, X_seq, last_in in build_sequences_for_predictions_agg(train_df,test_df,feats,best_c["seq_len"],sc):
        y=float(best_m.predict([X_seq.reshape(1,*X_seq.shape), last_in.reshape(1,1)], verbose=0).ravel()[0]); idx.append(ts); vals.append(y)
    return pd.Series(vals,index=pd.Index(idx,name="time_stamp")).sort_index(), {"best_params":best_c,"feature_cols":feats,"scaler":sc,"model":best_m}

def compute_metrics(y_true,y_pred):
    aligned=pd.concat([y_true.rename("y"),y_pred.rename("yhat")],axis=1).dropna()
    if len(aligned)==0: return {"MAE":np.nan,"MSE":np.nan,"RMSE":np.nan,"MAPE%":np.nan,"sMAPE%":np.nan}
    y=aligned["y"].values; yhat=aligned["yhat"].values; e=yhat-y
    return {"MAE":float(np.mean(np.abs(e))),"MSE":float(np.mean(e**2)),"RMSE":float(np.sqrt(np.mean(e**2))),"MAPE%":float(mape(y,yhat)),"sMAPE%":float(smape(y,yhat))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv_path",required=True)
    ap.add_argument("--out_dir",default="outputs_supply_eday")
    ap.add_argument("--time_col",default=None)
    ap.add_argument("--target_col",default=None)
    ap.add_argument("--on_bad_lines",default="skip",choices=["skip","warn","error"])
    ap.add_argument("--resample_rule",default="1H")
    ap.add_argument("--e_day_mode",default="last_diff",choices=["off","last_diff","diff_then_sum"], help="How to convert E-day cumulative values to hourly increments.")
    ap.add_argument("--reset_tz",default="Europe/London",help="Timezone for daily/hourly boundaries")
    ap.add_argument("--synthesize_time_start",dest="synth_start",default=None,help="If provided, synthesize a timestamp starting here (e.g., '2023-01-01 00:00:00')")
    ap.add_argument("--synthesize_time_freq",dest="synth_freq",default=None,help="Frequency for synthesized timestamps (e.g., '30S')")
    args=ap.parse_args()

    cfg=Config(csv_path=args.csv_path,out_dir=args.out_dir,resample_rule=args.resample_rule,
               time_col=args.time_col,target_col=args.target_col,on_bad_lines=args.on_bad_lines,
               e_day_mode=args.e_day_mode, reset_tz=args.reset_tz, synth_start=args.synth_start, synth_freq=args.synth_freq)

    ensure_dir(cfg.out_dir); ensure_dir(os.path.join(cfg.out_dir,"plots")); ensure_dir(os.path.join(cfg.out_dir,"models")); ensure_dir(os.path.join(cfg.out_dir,"predictions"))
    print("[1/7] Load"); raw, detected_target = load_supply(cfg.csv_path,cfg); print("Detected target:", detected_target or "(none)")
    # Ensure tz-aware UTC
    raw["time_stamp"] = pd.to_datetime(raw["time_stamp"], utc=True)

    print("[2/7] E-day conversion mode:", cfg.e_day_mode)
    if cfg.e_day_mode == "last_diff":
        rs = hourly_from_eday_last_diff(raw, reset_tz=cfg.reset_tz)
    elif cfg.e_day_mode == "diff_then_sum":
        raw = convert_daily_resets_to_flow(raw, target_col="target", reset_tz=cfg.reset_tz)
        rs = resample_hourly_sum_target(raw, cfg.resample_rule, cfg.max_interp_minutes)
    else:
        # off -> assume target already per-interval; just hourly sum
        rs = resample_hourly_sum_target(raw, cfg.resample_rule, cfg.max_interp_minutes)

    print("[3/7] Remove outage window"); rs = remove_blackouts(rs, cfg.blackout_ranges)
    print("[4/7] Time features + FE"); rs = add_time_features(rs); fe = add_lag_roll(rs, cfg.lags, cfg.roll_windows)
    print("[5/7] Split"); train_df, test_df, cutoff = build_train_test(fe, cfg.test_fraction_time); print("Cutoff:", cutoff)
    y_true = test_df.set_index("time_stamp")["target"].sort_index()
    print("[6/7] Baselines"); steps_daily = steps_for_period(cfg.resample_rule,"1D"); naive_last=forecast_naive_last_agg(test_df,train_df); naive_seas=forecast_naive_seasonal_agg(test_df,train_df,steps_daily)
    print("[7/7] Models")
    preds={}; details={}
    print(" - GBM"); y_gbm,info=train_eval_gbm(train_df,test_df,cfg); preds["GBM"]=y_gbm; details["GBM"]=info
    try:
        import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); joblib.dump(info["model"], d/"gbm_model.joblib")
        open(d/"gbm_meta.json","w").write(json.dumps({"best_params":info["best_params"],"n_features":info["n_features"],"feature_cols":info["feature_cols"],"exog":info["exog"]},indent=2))
    except Exception as e: print("[WARN] save GBM",e)
    print(" - KNN"); y_knn,info=train_eval_knn(train_df,test_df,cfg); preds["KNN"]=y_knn; details["KNN"]=info
    try:
        import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); joblib.dump(info["model"], d/"knn_model.joblib"); joblib.dump(info["scaler"], d/"knn_scaler.joblib")
        open(d/"knn_meta.json","w").write(json.dumps({"best_params":info["best_params"],"feature_cols":info["feature_cols"],"exog":info["exog"]},indent=2))
    except Exception as e: print("[WARN] save KNN",e)
    if HAS_TF:
        print(" - GRU"); y_gru,info=train_eval_rnn_agg(train_df,test_df,cfg,cfg.out_dir); preds["GRU"]=y_gru; details["GRU"]=info
        try:
            import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); info["model"].save(d/"gru_best.keras"); joblib.dump(info["scaler"], d/"gru_scaler.joblib")
            meta={k:v for k,v in info.items() if k not in ("model","scaler")}; meta["scaler_path"]=str(d/"gru_scaler.joblib"); open(d/"gru_meta.json","w").write(json.dumps(meta,indent=2))
        except Exception as e: print("[WARN] save GRU:",e)
        print(" - TCN"); y_tcn,info=train_eval_tcn_agg(train_df,test_df,cfg,cfg.out_dir); preds["TCN"]=y_tcn; details["TCN"]=info
        try:
            import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); info["model"].save(d/"tcn_best.keras"); joblib.dump(info["scaler"], d/"tcn_scaler.joblib")
            meta={k:v for k,v in info.items() if k not in ("model","scaler")}; meta["scaler_path"]=str(d/"tcn_scaler.joblib"); open(d/"tcn_meta.json","w").write(json.dumps(meta,indent=2))
        except Exception as e: print("[WARN] save TCN:",e)
        print(" - Transformer"); y_trf,info=train_eval_transformer_agg(train_df,test_df,cfg,cfg.out_dir); preds["Transformer"]=y_trf; details["Transformer"]=info
        try:
            import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); info["model"].save(d/"transformer_best.keras"); joblib.dump(info["scaler"], d/"transformer_scaler.joblib")
            meta={k:v for k,v in info.items() if k not in ("model","scaler")}; meta["scaler_path"]=str(d/"transformer_scaler.joblib"); open(d/"transformer_meta.json","w").write(json.dumps(meta,indent=2))
        except Exception as e: print("[WARN] save Transformer:",e)
    else:
        print("[WARN] TensorFlow not available; skipping GRU/TCN/Transformer.")

    # Evaluate & Save
    rows=[{"model":k,**compute_metrics(y_true,v)} for k,v in preds.items()]
    rows.append({"model":"NaiveLast",**compute_metrics(y_true,naive_last)}); rows.append({"model":"NaiveSeasonalDaily",**compute_metrics(y_true,naive_seas)})
    res=pd.DataFrame(rows).sort_values("RMSE"); res.to_csv(os.path.join(cfg.out_dir,"metrics_summary.csv"),index=False); print(res)
    comb=pd.DataFrame({"time_stamp":y_true.index,"y":y_true.values})
    for k,v in preds.items(): comb=comb.merge(v.rename(f"yhat_{k}").reset_index(), on="time_stamp", how="left")
    comb.to_csv(os.path.join(cfg.out_dir,"predictions","all_predictions_supply_eday.csv"),index=False)

    # Plots
    fig=go.Figure()
    for m in ["MAE","RMSE","sMAPE%"]: fig.add_trace(go.Bar(x=res["model"],y=res[m],name=m))
    fig.update_layout(barmode="group",title="Supply (E-day) — Model Comparison",xaxis_title="Model",yaxis_title="Score",width=1200,height=500)
    fig.write_html(os.path.join(cfg.out_dir,"plots","supply_eday_model_comparison.html"))
    try: fig.write_image(os.path.join(cfg.out_dir,"plots","supply_eday_model_comparison.png"),width=1200,height=500,scale=2)
    except Exception as e: print("[WARN] kaleido:",e)

    fig2=go.Figure(); fig2.add_trace(go.Scatter(x=comb["time_stamp"],y=comb["y"],mode="lines",name="Actual"))
    for k in preds.keys(): fig2.add_trace(go.Scatter(x=comb["time_stamp"],y=comb[f"yhat_{k}"],mode="lines",name=k))
    fig2.update_layout(title="Supply (E-day) — Actual vs Predicted (Test)",xaxis_title="Time",yaxis_title="Target",width=1200,height=500)
    fig2.write_html(os.path.join(cfg.out_dir,"plots","supply_eday_actual_vs_pred.html"))
    try: fig2.write_image(os.path.join(cfg.out_dir,"plots","supply_eday_actual_vs_pred.png"),width=1200,height=500,scale=2)
    except Exception as e: print("[WARN] kaleido:",e)

    comb_err = comb.copy()
    for k in preds.keys(): comb_err[f"resid_{k}"]=comb_err[f"yhat_{k}"]-comb_err["y"]
    fig3=go.Figure()
    for k in preds.keys(): fig3.add_trace(go.Scatter(x=comb_err["time_stamp"],y=comb_err[f"resid_{k}"],mode="lines",name=k))
    fig3.add_hline(y=0, line_dash="dash")
    fig3.update_layout(title="Supply (E-day) — Residuals Over Time (yhat - y)",xaxis_title="Time",yaxis_title="Residual",width=1200,height=500)
    fig3.write_html(os.path.join(cfg.out_dir,"plots","supply_eday_residuals_over_time.html"))
    try: fig3.write_image(os.path.join(cfg.out_dir,"plots","supply_eday_residuals_over_time.png"),width=1200,height=500,scale=2)
    except Exception as e: print("[WARN] kaleido:",e)

    open(os.path.join(cfg.out_dir,"experiment_manifest.json"),"w").write(json.dumps({
        "cutoff_time": str(cutoff),
        "detected_target": detected_target,
        "config": asdict(cfg),
        "models": list(preds.keys())
    }, indent=2))
    print("Done. Outputs in:", cfg.out_dir)

if __name__=="__main__":
    main()
