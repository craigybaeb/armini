#!/usr/bin/env python3
import os, json, argparse, warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
try:
    import tensorflow as tf
    from tensorflow.keras import Model, Input, Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding, Concatenate, Flatten
    from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except Exception:
    HAS_TF = False

@dataclass
class Config:
    csv_path: str
    out_dir: str = "outputs_global"
    resample_rule: str = "1H"
    test_fraction_time: float = 0.2
    blackout_ranges: Optional[List[Tuple[str, str]]] = (("2024-05-02","2024-05-10"),)
    max_interp_minutes: int = 60
    aggregate_total: bool = False
    lags: Tuple[int,...] = (1,2,3,6,12)
    roll_windows: Tuple[int,...] = (3,6,12)
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
    def __post_init__(self):
        if self.gbm_grid is None: self.gbm_grid={"learning_rate":[0.05,0.1],"max_iter":[300,600],"max_leaf_nodes":[31,63]}
        if self.knn_grid is None: self.knn_grid={"n_neighbors":[5,15,30],"weights":["uniform","distance"]}
        if self.rnn_grid is None: self.rnn_grid={"seq_len":[12],"units1":[64],"units2":[32],"dropout":[0.2],"batch_size":[256]}
        if self.tcn_grid is None: self.tcn_grid={"seq_len":[12],"filters":[32,64],"kernel_size":[3,5],"dilations":[[1,2,4,8]],"dropout":[0.2],"batch_size":[256]}
        if self.transformer_grid is None: self.transformer_grid={"seq_len":[12],"d_model":[64],"num_heads":[2,4],"ff_units":[128],"dropout":[0.1],"batch_size":[256]}
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def smape(y, yhat, eps=1e-6): d=np.abs(y)+np.abs(yhat); d=np.where(d<eps,eps,d); return 100.0*np.mean(2.0*np.abs(yhat-y)/d)
def mape(y, yhat, eps=1e-6): d=np.where(np.abs(y)<eps,eps,np.abs(y)); return 100.0*np.mean(np.abs((y-yhat)/d))
def time_split_cutoff(df, col, frac): return df[col].quantile(1.0-frac)

def load_and_preprocess(csv_path:str)->pd.DataFrame:
    cols=['id','time_stamp','device_id','power','voltage','apparent_power','factor','reactive_power','current','space_cluster_id','space_id']
    df=pd.read_csv(csv_path, header=None, names=cols, dtype=str)
    df['time_stamp']=pd.to_datetime(df['time_stamp'], errors='coerce', utc=True)
    for c in ['power','voltage','apparent_power','factor','reactive_power','current']: df[c]=pd.to_numeric(df[c], errors='coerce')
    df['space_cluster_id']=df['space_cluster_id'].replace('$twin.tags.space_cluster_id', np.nan)
    df['space_id']=df['space_id'].replace('$twin.tags.space_id', np.nan)
    df=df.dropna(subset=['time_stamp','device_id','power']).copy()
    df=df.sort_values(['device_id','time_stamp']).drop_duplicates(['device_id','time_stamp'], keep='last')
    return df.reset_index(drop=True)
def resample_per_device(df, rule="1H", max_interp_minutes=60):
    parts=[]; exog=['voltage','apparent_power','current','factor','reactive_power']
    for dev,g in df.groupby('device_id'):
        g=g.set_index('time_stamp').sort_index(); use=['power']+[c for c in exog if c in g.columns]
        g=g[use].resample(rule).mean(); g['power']=g['power'].interpolate(method='time',limit=max_interp_minutes,limit_direction='both')
        for c in exog:
            if c in g: g[c]=g[c].ffill(limit=max_interp_minutes).bfill(limit=max_interp_minutes)
        g['device_id']=dev; parts.append(g.reset_index())
    out=pd.concat(parts, ignore_index=True); out=out.dropna(subset=['power']).copy()
    return out.sort_values(['device_id','time_stamp']).reset_index(drop=True)
def remove_blackouts(df, blackout_ranges):
    if not blackout_ranges: return df
    out=df.copy()
    if out['time_stamp'].dt.tz is None: out['time_stamp']=pd.to_datetime(out['time_stamp'], utc=True)
    tz=out['time_stamp'].dt.tz
    for s,e in blackout_ranges:
        s=pd.to_datetime(s,utc=True).tz_convert(tz); e=pd.to_datetime(e,utc=True).tz_convert(tz)
        mask=(out['time_stamp']>=s)&(out['time_stamp']<e); out=out.loc[~mask]
    return out.reset_index(drop=True)
def add_time_features(df):
    df=df.copy(); df['hour']=df['time_stamp'].dt.hour; df['dayofweek']=df['time_stamp'].dt.dayofweek; df['is_weekend']=df['dayofweek'].isin([5,6]).astype(int); return df

def add_lag_roll_time_features(df,lags,rolls):
    parts=[]
    for dev,g in df.groupby('device_id'):
        g=g.copy()
        for L in lags:
            g[f'power_lag_{L}']=g['power'].shift(L)
            for c in ['voltage','apparent_power','current','factor','reactive_power']:
                if c in g: g[f'{c}_lag_{L}']=g[c].shift(L)
        for W in rolls:
            g[f'power_roll_mean_{W}']=g['power'].rolling(W, min_periods=max(1,W//2)).mean()
            g[f'power_roll_std_{W}']=g['power'].rolling(W, min_periods=max(1,W//2)).std()
        parts.append(g)
    out=pd.concat(parts, ignore_index=True); out=add_time_features(out)
    return out.dropna().reset_index(drop=True)
def build_train_test(df, frac): 
    c=time_split_cutoff(df,'time_stamp',frac); 
    return df[df['time_stamp']<=c].copy(), df[df['time_stamp']>c].copy(), c
from pandas.tseries.frequencies import to_offset
def steps_for_period(rule, period):
    off=to_offset(rule); delta=getattr(off,'delta',None) or pd.Timedelta('1H')
    return max(1,int(pd.Timedelta(period)/delta))

def forecast_naive_last(test_df, train_df):
    preds=[]; all_df=pd.concat([train_df,test_df],ignore_index=True)
    for dev,g in all_df.groupby('device_id'):
        g=g.sort_values('time_stamp'); g['yhat']=g['power'].shift(1)
        g=g[g['time_stamp'].isin(test_df['time_stamp'])]; preds.append(g[['time_stamp','device_id','yhat']])
    return pd.concat(preds).set_index(['time_stamp','device_id']).sort_index()['yhat']
def forecast_naive_seasonal(test_df, train_df, season):
    preds=[]; all_df=pd.concat([train_df,test_df],ignore_index=True)
    for dev,g in all_df.groupby('device_id'):
        g=g.sort_values('time_stamp'); g['yhat']=g['power'].shift(season)
        g=g[g['time_stamp'].isin(test_df['time_stamp'])]; preds.append(g[['time_stamp','device_id','yhat']])
    return pd.concat(preds).set_index(['time_stamp','device_id']).sort_index()['yhat']

def train_eval_gbm(train_df,test_df,cfg):
    base=['voltage','apparent_power','current','factor','reactive_power']
    lag=[c for c in train_df.columns if c.startswith('power_lag_') or any(c.startswith(f'{v}_lag_') for v in base)]
    roll=[c for c in train_df.columns if c.startswith('power_roll_')]
    timef=['hour','dayofweek','is_weekend']
    dtr=pd.get_dummies(train_df['device_id'],prefix='dev')
    dte=pd.get_dummies(test_df['device_id'],prefix='dev').reindex(columns=dtr.columns, fill_value=0)
    feats=base+lag+roll+timef
    Xtr=pd.concat([train_df[feats],dtr],axis=1).astype(np.float32); Xte=pd.concat([test_df[feats],dte],axis=1).astype(np.float32)
    ytr=train_df['power'].values
    cut=train_df['time_stamp'].quantile(0.9); idx_tr=train_df['time_stamp']<=cut; idx_va=train_df['time_stamp']>cut
    best=None; br=np.inf; bm=None
    for lr in cfg.gbm_grid['learning_rate']:
        for it in cfg.gbm_grid['max_iter']:
            for mln in cfg.gbm_grid['max_leaf_nodes']:
                m=HistGradientBoostingRegressor(learning_rate=lr,max_iter=it,max_leaf_nodes=mln,early_stopping=True,validation_fraction=0.1,random_state=42)
                m.fit(Xtr[idx_tr],ytr[idx_tr]); rmse=mean_squared_error(ytr[idx_va],m.predict(Xtr[idx_va]),squared=False)
                if rmse<br: br, best, bm = rmse, {"learning_rate":lr,"max_iter":it,"max_leaf_nodes":mln}, m
    bm.fit(Xtr,ytr); yhat=bm.predict(Xte)
    return pd.Series(yhat,index=test_df.set_index(['time_stamp','device_id']).index), {"best_params":best,"n_features":Xtr.shape[1],"model":bm}

def train_eval_knn(train_df,test_df,cfg):
    base=['voltage','apparent_power','current','factor','reactive_power']
    lag=[c for c in train_df.columns if c.startswith('power_lag_') or any(c.startswith(f'{v}_lag_') for v in base)]
    roll=[c for c in train_df.columns if c.startswith('power_roll_')]
    timef=['hour','dayofweek','is_weekend']
    dtr=pd.get_dummies(train_df['device_id'],prefix='dev')
    dte=pd.get_dummies(test_df['device_id'],prefix='dev').reindex(columns=dtr.columns, fill_value=0)
    feats=base+lag+roll+timef
    Xtr=pd.concat([train_df[feats],dtr],axis=1).astype(np.float32); Xte=pd.concat([test_df[feats],dte],axis=1).astype(np.float32)
    ytr=train_df['power'].values
    sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    cut=train_df['time_stamp'].quantile(0.9); idx_tr=train_df['time_stamp']<=cut; idx_va=train_df['time_stamp']>cut
    best=None; br=np.inf; bm=None
    for k in cfg.knn_grid['n_neighbors']:
        for w in cfg.knn_grid['weights']:
            m=KNeighborsRegressor(n_neighbors=k,weights=w); m.fit(Xtr_s[idx_tr],ytr[idx_tr]); rmse=mean_squared_error(ytr[idx_va],m.predict(Xtr_s[idx_va]),squared=False)
            if rmse<br: br, best, bm = rmse, {"n_neighbors":k,"weights":w}, m
    bm.fit(Xtr_s,ytr); yhat=bm.predict(Xte_s)
    return pd.Series(yhat,index=test_df.set_index(['time_stamp','device_id']).index), {"best_params":best,"scaler":sc,"model":bm}

def build_tf_datasets(train_df, feature_cols, seq_len, scaler, val_cutoff_ts):
    devices=sorted(train_df['device_id'].unique().tolist()); d2i={d:i for i,d in enumerate(devices)}
    def _gen(split):
        for dev in devices:
            g=train_df[train_df['device_id']==dev].sort_values('time_stamp')
            if len(g)<=seq_len: continue
            Xall=scaler.transform(g[feature_cols].astype(np.float32)); yall=g['power'].astype(np.float32); ts=g['time_stamp']; di=np.array([d2i[dev]],dtype=np.int32)
            for i in range(len(g)-seq_len):
                end_ts=ts.iloc[i+seq_len]; is_tr=(end_ts<=val_cutoff_ts)
                if (split=="train" and is_tr) or (split=="val" and not is_tr):
                    last_in=np.array([yall.iloc[i+seq_len-1]],dtype=np.float32)
                    yield {"seq_in":Xall[i:i+seq_len,:],"dev_in":di,"last_in":last_in}, float(yall.iloc[i+seq_len])
    sig=({"seq_in":tf.TensorSpec(shape=(seq_len,len(feature_cols)),dtype=tf.float32),"dev_in":tf.TensorSpec(shape=(1,),dtype=tf.int32),"last_in":tf.TensorSpec(shape=(1,),dtype=tf.float32)}, tf.TensorSpec(shape=(),dtype=tf.float32))
    return tf.data.Dataset.from_generator(lambda:_gen("train"),output_signature=sig), tf.data.Dataset.from_generator(lambda:_gen("val"),output_signature=sig), devices
def build_sequences_for_predictions(train_df,test_df,feature_cols,seq_len,scaler,devices_to_predict,train_devices):
    d2i={d:i for i,d in enumerate(sorted(train_devices))}; unk=len(train_devices)
    for dev in sorted(devices_to_predict):
        gtr=train_df[train_df['device_id']==dev].sort_values('time_stamp'); gte=test_df[test_df['device_id']==dev].sort_values('time_stamp')
        if gte.empty: continue
        base=pd.concat([gtr,gte],ignore_index=True); X=scaler.transform(base[feature_cols].astype(np.float32)); y=base['power'].astype(np.float32).values
        test_ts=set(gte['time_stamp'])
        for i in range(len(base)-seq_len):
            t=i+seq_len; ts=base.iloc[t]['time_stamp']
            if ts in test_ts:
                yield ts, dev, X[i:t,:], np.array([[d2i.get(dev,unk)]],dtype=np.int32), np.array([y[t-1]],dtype=np.float32)

def build_rnn_model(seq_len,n_features,n_devices,u1,u2,drop):
    inp_seq=Input(shape=(seq_len,n_features),name="seq_in")
    x=GRU(u1,return_sequences=True)(inp_seq); x=Dropout(drop)(x); x=GRU(u2)(x); x=Dropout(drop)(x)
    inp_dev=Input(shape=(1,),dtype='int32',name="dev_in"); emb=Embedding(input_dim=n_devices,output_dim=max(4,int(np.ceil(np.log2(n_devices+1)))),name="dev_emb")(inp_dev); emb=Flatten()(emb)
    inp_last=Input(shape=(1,),dtype='float32',name="last_in")
    h=Concatenate()([x,emb]); delta=Dense(1)(h); out=tf.keras.layers.Add()([inp_last,delta])
    m=Model(inputs=[inp_seq,inp_dev,inp_last],outputs=out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,clipnorm=1.0),loss='mse'); return m
def train_eval_rnn(train_df,test_df,cfg,out_dir):
    base=['voltage','apparent_power','current','factor','reactive_power']; timef=['hour','dayofweek','is_weekend']
    feat=['power']+[c for c in base+timef if c in train_df.columns]
    sc=StandardScaler().fit(train_df[feat].astype(np.float32)); best_m=None; best_c=None; br=np.inf; best_devs=sorted(train_df['device_id'].unique().tolist())
    cut=train_df['time_stamp'].quantile(0.9)
    for L in cfg.rnn_grid['seq_len']:
        dtr,dva,devs=build_tf_datasets(train_df,feat,L,sc,cut)
        for u1 in cfg.rnn_grid['units1']:
            for u2 in cfg.rnn_grid['units2']:
                for dr in cfg.rnn_grid['dropout']:
                    for bs in cfg.rnn_grid['batch_size']:
                        m=build_rnn_model(L,len(feat),len(devs)+1,u1,u2,dr)
                        cbs=[EarlyStopping(monitor='val_loss',patience=cfg.rnn_patience,restore_best_weights=True,verbose=0), ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=max(2,cfg.rnn_patience//2),min_lr=1e-5,verbose=0)]
                        m.fit(dtr.batch(bs).prefetch(tf.data.AUTOTUNE), validation_data=dva.batch(bs).prefetch(tf.data.AUTOTUNE), epochs=cfg.rnn_epochs, verbose=0, callbacks=cbs)
                        rmse=float(np.sqrt(m.evaluate(dva.batch(bs),verbose=0)))
                        if rmse<br: br, best_c, best_m, best_devs = rmse, {"seq_len":L,"units1":u1,"units2":u2,"dropout":dr,"batch_size":bs}, m, devs
    devices_all=sorted(pd.concat([train_df['device_id'],test_df['device_id']]).unique().tolist())
    idx=[]; vals=[]
    for ts,dev,X_seq,dev_idx,last_in in build_sequences_for_predictions(train_df,test_df,feat,best_c['seq_len'],sc,devices_all,best_devs):
        y=float(best_m.predict([X_seq.reshape(1,*X_seq.shape),dev_idx,last_in.reshape(1,1)],verbose=0).ravel()[0]); idx.append((ts,dev)); vals.append(y)
    yhat=pd.Series(vals,index=pd.MultiIndex.from_tuples(idx,names=['time_stamp','device_id'])).sort_index()
    return yhat, {"best_params":best_c,"feature_cols":feat,"scaler":sc,"model":best_m}

def build_tcn_model(L,n_features,n_devices,filters,ks,dils,drop):
    inp=Input(shape=(L,n_features),name="seq_in"); x=inp
    for d in dils:
        res=x
        x=Conv1D(filters=filters,kernel_size=ks,padding='causal',dilation_rate=d,activation='relu')(x); x=Dropout(drop)(x)
        x=Conv1D(filters=filters,kernel_size=ks,padding='causal',dilation_rate=d,activation='relu')(x)
        if res.shape[-1]!=x.shape[-1]: res=Conv1D(filters=filters,kernel_size=1,padding='same')(res)
        x=x+res
    x=GlobalAveragePooling1D()(x)
    inp_dev=Input(shape=(1,),dtype='int32',name="dev_in"); emb=Embedding(input_dim=n_devices,output_dim=max(4,int(np.ceil(np.log2(n_devices+1)))),name="dev_emb")(inp_dev); emb=Flatten()(emb)
    last=Input(shape=(1,),dtype='float32',name="last_in"); h=Concatenate()([x,emb]); delta=Dense(1)(h); out=tf.keras.layers.Add()([last,delta])
    m=Model(inputs=[inp,inp_dev,last],outputs=out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,clipnorm=1.0),loss='mse'); return m
def train_eval_tcn(train_df,test_df,cfg,out_dir):
    base=['voltage','apparent_power','current','factor','reactive_power']; timef=['hour','dayofweek','is_weekend']
    feat=['power']+[c for c in base+timef if c in train_df.columns]
    sc=StandardScaler().fit(train_df[feat].astype(np.float32)); cut=train_df['time_stamp'].quantile(0.9)
    best_m=None; best_c=None; br=np.inf; best_devs=sorted(train_df['device_id'].unique().tolist())
    def build_ds(L): 
        return build_tf_datasets(train_df,feat,L,sc,cut)
    for L in cfg.tcn_grid['seq_len']:
        dtr,dva,devs=build_ds(L)
        for f in cfg.tcn_grid['filters']:
            for ks in cfg.tcn_grid['kernel_size']:
                for dils in cfg.tcn_grid['dilations']:
                    for dr in cfg.tcn_grid['dropout']:
                        for bs in cfg.tcn_grid['batch_size']:
                            m=build_tcn_model(L,len(feat),len(devs)+1,f,ks,dils,dr)
                            cbs=[EarlyStopping(monitor='val_loss',patience=cfg.tcn_patience,restore_best_weights=True,verbose=0), ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=max(2,cfg.tcn_patience//2),min_lr=1e-5,verbose=0)]
                            m.fit(dtr.batch(bs).prefetch(tf.data.AUTOTUNE),validation_data=dva.batch(bs).prefetch(tf.data.AUTOTUNE),epochs=cfg.tcn_epochs,verbose=0,callbacks=cbs)
                            rmse=float(np.sqrt(m.evaluate(dva.batch(bs),verbose=0)))
                            if rmse<br: br,best_c,best_m,best_devs=rmse,{"seq_len":L,"filters":f,"kernel_size":ks,"dilations":dils,"dropout":dr,"batch_size":bs},m,devs
    devices_all=sorted(pd.concat([train_df['device_id'],test_df['device_id']]).unique().tolist())
    idx=[]; vals=[]
    for ts,dev,X_seq,dev_idx,last_in in build_sequences_for_predictions(train_df,test_df,feat,best_c['seq_len'],sc,devices_all,best_devs):
        y=float(best_m.predict([X_seq.reshape(1,*X_seq.shape),dev_idx,last_in.reshape(1,1)],verbose=0).ravel()[0]); idx.append((ts,dev)); vals.append(y)
    yhat=pd.Series(vals,index=pd.MultiIndex.from_tuples(idx,names=['time_stamp','device_id'])).sort_index()
    return yhat, {"best_params":best_c,"feature_cols":feat,"scaler":sc,"model":best_m}

def positional_encoding(L,d_model):
    pos=np.arange(L)[:,None]; i=np.arange(d_model)[None,:]; rate=1/np.power(10000,(2*(i//2))/np.float32(d_model)); ang=pos*rate
    pe=np.zeros((L,d_model)); pe[:,0::2]=np.sin(ang[:,0::2]); pe[:,1::2]=np.cos(ang[:,1::2]); return tf.convert_to_tensor(pe,dtype=tf.float32)
def build_transformer_model(L,n_features,n_devices,d_model,heads,ffu,drop):
    inp=Input(shape=(L,n_features),name="seq_in"); x=Dense(d_model)(inp); x=x+positional_encoding(L,d_model)
    att=MultiHeadAttention(num_heads=heads,key_dim=max(1,d_model//max(1,heads)))(x,x); x=LayerNormalization()(x+att)
    ff=Sequential([Dense(ffu,activation='relu'),Dropout(drop),Dense(d_model)]); x=LayerNormalization()(x+ff(x)); x=GlobalAveragePooling1D()(x)
    inp_dev=Input(shape=(1,),dtype='int32',name="dev_in"); emb=Embedding(input_dim=n_devices,output_dim=max(4,int(np.ceil(np.log2(n_devices+1)))),name="dev_emb")(inp_dev); emb=Flatten()(emb)
    last=Input(shape=(1,),dtype='float32',name="last_in"); h=Concatenate()([x,emb]); delta=Dense(1)(h); out=tf.keras.layers.Add()([last,delta])
    m=Model(inputs=[inp,inp_dev,last],outputs=out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,clipnorm=1.0),loss='mse'); return m
def train_eval_transformer(train_df,test_df,cfg,out_dir):
    base=['voltage','apparent_power','current','factor','reactive_power']; timef=['hour','dayofweek','is_weekend']
    feat=['power']+[c for c in base+timef if c in train_df.columns]
    sc=StandardScaler().fit(train_df[feat].astype(np.float32)); cut=train_df['time_stamp'].quantile(0.9)
    best_m=None; best_c=None; br=np.inf; best_devs=sorted(train_df['device_id'].unique().tolist())
    for L in cfg.transformer_grid['seq_len']:
        dtr,dva,devs=build_tf_datasets(train_df,feat,L,sc,cut)
        for dm in cfg.transformer_grid['d_model']:
            for h in cfg.transformer_grid['num_heads']:
                for ff in cfg.transformer_grid['ff_units']:
                    for dr in cfg.transformer_grid['dropout']:
                        for bs in cfg.transformer_grid['batch_size']:
                            m=build_transformer_model(L,len(feat),len(devs)+1,dm,h,ff,dr)
                            cbs=[EarlyStopping(monitor='val_loss',patience=cfg.transformer_patience,restore_best_weights=True,verbose=0), ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=max(2,cfg.transformer_patience//2),min_lr=1e-5,verbose=0)]
                            m.fit(dtr.batch(bs).prefetch(tf.data.AUTOTUNE),validation_data=dva.batch(bs).prefetch(tf.data.AUTOTUNE),epochs=cfg.transformer_epochs,verbose=0,callbacks=cbs)
                            rmse=float(np.sqrt(m.evaluate(dva.batch(bs),verbose=0)))
                            if rmse<br: br,best_c,best_m,best_devs=rmse,{"seq_len":L,"d_model":dm,"num_heads":h,"ff_units":ff,"dropout":dr,"batch_size":bs},m,devs
    devices_all=sorted(pd.concat([train_df['device_id'],test_df['device_id']]).unique().tolist())
    idx=[]; vals=[]
    for ts,dev,X_seq,dev_idx,last_in in build_sequences_for_predictions(train_df,test_df,feat,best_c['seq_len'],sc,devices_all,best_devs):
        y=float(best_m.predict([X_seq.reshape(1,*X_seq.shape),dev_idx,last_in.reshape(1,1)],verbose=0).ravel()[0]); idx.append((ts,dev)); vals.append(y)
    yhat=pd.Series(vals,index=pd.MultiIndex.from_tuples(idx,names=['time_stamp','device_id'])).sort_index()
    return yhat, {"best_params":best_c,"feature_cols":feat,"scaler":sc,"model":best_m}

def aggregate_total_series(rs):
    agg=(rs.groupby('time_stamp',as_index=False).agg(total_power=('power','sum'),active_devices=('device_id','nunique'),mean_voltage=('voltage','mean'),sum_apparent_power=('apparent_power','sum'),sum_reactive_power=('reactive_power','sum'),sum_current=('current','sum'),mean_factor=('factor','mean')).sort_values('time_stamp'))
    for c in ['mean_voltage','sum_apparent_power','sum_reactive_power','sum_current','mean_factor']:
        if c in agg: agg[c]=agg[c].ffill().bfill()
    return agg
def add_lag_roll_total(agg,lags,rolls):
    df=agg.copy()
    for L in lags: df[f'total_lag_{L}']=df['total_power'].shift(L)
    for W in rolls:
        df[f'total_roll_mean_{W}']=df['total_power'].rolling(W,min_periods=max(1,W//2)).mean()
        df[f'total_roll_std_{W}']=df['total_power'].rolling(W,min_periods=max(1,W//2)).std()
    df=add_time_features(df); return df.dropna().reset_index(drop=True)
def forecast_naive_last_agg(test_df,train_df):
    all_df=pd.concat([train_df,test_df],ignore_index=True).sort_values('time_stamp'); all_df['yhat']=all_df['total_power'].shift(1)
    return all_df[all_df['time_stamp'].isin(test_df['time_stamp'])].set_index('time_stamp')['yhat'].sort_index()
def forecast_naive_seasonal_agg(test_df,train_df,season):
    all_df=pd.concat([train_df,test_df],ignore_index=True).sort_values('time_stamp'); all_df['yhat']=all_df['total_power'].shift(season)
    return all_df[all_df['time_stamp'].isin(test_df['time_stamp'])].set_index('time_stamp')['yhat'].sort_index()

def train_eval_gbm_agg(train_df,test_df,cfg):
    timef=['hour','dayofweek','is_weekend']; lag=[c for c in train_df.columns if c.startswith('total_lag_') or c.startswith('total_roll_')]
    exog=[c for c in ['active_devices','mean_voltage','sum_apparent_power','sum_reactive_power','sum_current','mean_factor'] if c in train_df.columns]
    feats=exog+timef+lag; Xtr=train_df[feats].astype(np.float32); Xte=test_df[feats].astype(np.float32); ytr=train_df['total_power'].values
    cut=train_df['time_stamp'].quantile(0.9); idx_tr=train_df['time_stamp']<=cut; idx_va=train_df['time_stamp']>cut
    best=None; br=np.inf; bm=None
    for lr in cfg.gbm_grid['learning_rate']:
        for it in cfg.gbm_grid['max_iter']:
            for mln in cfg.gbm_grid['max_leaf_nodes']:
                m=HistGradientBoostingRegressor(learning_rate=lr,max_iter=it,max_leaf_nodes=mln,early_stopping=True,validation_fraction=0.1,random_state=42)
                m.fit(Xtr[idx_tr],ytr[idx_tr]); rmse=mean_squared_error(ytr[idx_va],m.predict(Xtr[idx_va]),squared=False)
                if rmse<br: br,best,bm=rmse,{"learning_rate":lr,"max_iter":it,"max_leaf_nodes":mln},m
    bm.fit(Xtr,ytr); yhat=bm.predict(Xte); 
    return pd.Series(yhat,index=test_df['time_stamp']), {"best_params":best,"n_features":Xtr.shape[1],"model":bm,"feature_cols":feats}
def train_eval_knn_agg(train_df,test_df,cfg):
    timef=['hour','dayofweek','is_weekend']; lag=[c for c in train_df.columns if c.startswith('total_lag_') or c.startswith('total_roll_')]
    exog=[c for c in ['active_devices','mean_voltage','sum_apparent_power','sum_reactive_power','sum_current','mean_factor'] if c in train_df.columns]
    feats=exog+timef+lag; Xtr=train_df[feats].astype(np.float32); Xte=test_df[feats].astype(np.float32); ytr=train_df['total_power'].values
    sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    cut=train_df['time_stamp'].quantile(0.9); idx_tr=train_df['time_stamp']<=cut; idx_va=train_df['time_stamp']>cut
    best=None; br=np.inf; bm=None
    for k in cfg.knn_grid['n_neighbors']:
        for w in cfg.knn_grid['weights']:
            m=KNeighborsRegressor(n_neighbors=k,weights=w); m.fit(Xtr_s[idx_tr],ytr[idx_tr]); rmse=mean_squared_error(ytr[idx_va],m.predict(Xtr_s[idx_va]),squared=False)
            if rmse<br: br,best,bm=rmse,{"n_neighbors":k,"weights":w},m
    bm.fit(Xtr_s,ytr); yhat=bm.predict(Xte_s); 
    return pd.Series(yhat,index=test_df['time_stamp']), {"best_params":best,"scaler":sc,"model":bm,"feature_cols":feats}

def build_tf_datasets_agg(train_df,feats,L,sc,val_cut):
    def _gen(split):
        g=train_df.sort_values('time_stamp')
        if len(g)<=L: return
        X=sc.transform(g[feats].astype(np.float32)); y=g['total_power'].astype(np.float32).values; ts=g['time_stamp']
        for i in range(len(g)-L):
            end_ts=ts.iloc[i+L]; is_tr=(end_ts<=val_cut)
            if (split=="train" and is_tr) or (split=="val" and not is_tr):
                last=np.array([y[i+L-1]],dtype=np.float32)
                yield {"seq_in":X[i:i+L,:],"last_in":last}, float(y[i+L])
    sig=({"seq_in":tf.TensorSpec(shape=(L,len(feats)),dtype=tf.float32),"last_in":tf.TensorSpec(shape=(1,),dtype=tf.float32)}, tf.TensorSpec(shape=(),dtype=tf.float32))
    return tf.data.Dataset.from_generator(lambda:_gen("train"),output_signature=sig), tf.data.Dataset.from_generator(lambda:_gen("val"),output_signature=sig)
def build_sequences_for_predictions_agg(train_df,test_df,feats,L,sc):
    base=pd.concat([train_df,test_df],ignore_index=True).sort_values('time_stamp'); X=sc.transform(base[feats].astype(np.float32)); y=base['total_power'].astype(np.float32).values
    test_ts=set(test_df['time_stamp'])
    for i in range(len(base)-L):
        t=i+L; ts=base.iloc[t]['time_stamp']
        if ts in test_ts:
            yield ts, X[i:t,:], np.array([y[t-1]],dtype=np.float32)

def build_rnn_model_agg(L,n_features,u1,u2,drop):
    inp=Input(shape=(L,n_features),name="seq_in"); x=GRU(u1,return_sequences=True)(inp); x=Dropout(drop)(x); x=GRU(u2)(x); x=Dropout(drop)(x)
    last=Input(shape=(1,),dtype='float32',name="last_in"); delta=Dense(1)(x); out=tf.keras.layers.Add()([last,delta])
    m=Model(inputs=[inp,last],outputs=out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,clipnorm=1.0),loss='mse'); return m
def train_eval_rnn_agg(train_df,test_df,cfg,out_dir):
    timef=['hour','dayofweek','is_weekend']; exog=[c for c in ['active_devices','mean_voltage','sum_apparent_power','sum_reactive_power','sum_current','mean_factor'] if c in train_df.columns]
    if not set(timef).issubset(train_df.columns): train_df=add_time_features(train_df)
    if not set(timef).issubset(test_df.columns): test_df=add_time_features(test_df)
    feats=['total_power']+timef+exog; sc=StandardScaler().fit(train_df[feats].astype(np.float32)); cut=train_df['time_stamp'].quantile(0.9)
    best_m=None; best_c=None; br=np.inf
    for L in cfg.rnn_grid['seq_len']:
        dtr,dva=build_tf_datasets_agg(train_df,feats,L,sc,cut)
        for u1 in cfg.rnn_grid['units1']:
            for u2 in cfg.rnn_grid['units2']:
                for dr in cfg.rnn_grid['dropout']:
                    for bs in cfg.rnn_grid['batch_size']:
                        m=build_rnn_model_agg(L,len(feats),u1,u2,dr)
                        cbs=[EarlyStopping(monitor='val_loss',patience=cfg.rnn_patience,restore_best_weights=True,verbose=0), ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=max(2,cfg.rnn_patience//2),min_lr=1e-5,verbose=0)]
                        m.fit(dtr.batch(bs).prefetch(tf.data.AUTOTUNE),validation_data=dva.batch(bs).prefetch(tf.data.AUTOTUNE),epochs=cfg.rnn_epochs,verbose=0,callbacks=cbs)
                        rmse=float(np.sqrt(m.evaluate(dva.batch(bs),verbose=0)))
                        if rmse<br: br,best_c,best_m=rmse,{"seq_len":L,"units1":u1,"units2":u2,"dropout":dr,"batch_size":bs},m
    idx=[]; vals=[]
    for ts, X_seq, last_in in build_sequences_for_predictions_agg(train_df,test_df,feats,best_c['seq_len'],sc):
        y=float(best_m.predict([X_seq.reshape(1,*X_seq.shape), last_in.reshape(1,1)], verbose=0).ravel()[0]); idx.append(ts); vals.append(y)
    return pd.Series(vals,index=pd.Index(idx,name='time_stamp')).sort_index(), {"best_params":best_c,"feature_cols":feats,"scaler":sc,"model":best_m}

def build_tcn_model_agg(L,n_features,filters,ks,dils,drop):
    inp=Input(shape=(L,n_features),name="seq_in"); x=inp
    for d in dils:
        res=x; x=Conv1D(filters=filters,kernel_size=ks,padding='causal',dilation_rate=d,activation='relu')(x); x=Dropout(drop)(x)
        x=Conv1D(filters=filters,kernel_size=ks,padding='causal',dilation_rate=d,activation='relu')(x)
        if res.shape[-1]!=x.shape[-1]: res=Conv1D(filters=filters,kernel_size=1,padding='same')(res)
        x=x+res
    x=GlobalAveragePooling1D()(x)
    last=Input(shape=(1,),dtype='float32',name="last_in"); delta=Dense(1)(x); out=tf.keras.layers.Add()([last,delta])
    m=Model(inputs=[inp,last],outputs=out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,clipnorm=1.0),loss='mse'); return m
def train_eval_tcn_agg(train_df,test_df,cfg,out_dir):
    timef=['hour','dayofweek','is_weekend']; exog=[c for c in ['active_devices','mean_voltage','sum_apparent_power','sum_reactive_power','sum_current','mean_factor'] if c in train_df.columns]
    if not set(timef).issubset(train_df.columns): train_df=add_time_features(train_df)
    if not set(timef).issubset(test_df.columns): test_df=add_time_features(test_df)
    feats=['total_power']+timef+exog; sc=StandardScaler().fit(train_df[feats].astype(np.float32)); cut=train_df['time_stamp'].quantile(0.9)
    best_m=None; best_c=None; br=np.inf
    for L in cfg.tcn_grid['seq_len']:
        dtr,dva=build_tf_datasets_agg(train_df,feats,L,sc,cut)
        for f in cfg.tcn_grid['filters']:
            for ks in cfg.tcn_grid['kernel_size']:
                for dils in cfg.tcn_grid['dilations']:
                    for dr in cfg.tcn_grid['dropout']:
                        for bs in cfg.tcn_grid['batch_size']:
                            m=build_tcn_model_agg(L,len(feats),f,ks,dils,dr)
                            cbs=[EarlyStopping(monitor='val_loss',patience=cfg.tcn_patience,restore_best_weights=True,verbose=0), ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=max(2,cfg.tcn_patience//2),min_lr=1e-5,verbose=0)]
                            m.fit(dtr.batch(bs).prefetch(tf.data.AUTOTUNE),validation_data=dva.batch(bs).prefetch(tf.data.AUTOTUNE),epochs=cfg.tcn_epochs,verbose=0,callbacks=cbs)
                            rmse=float(np.sqrt(m.evaluate(dva.batch(bs),verbose=0)))
                            if rmse<br: br,best_c,best_m=rmse,{"seq_len":L,"filters":f,"kernel_size":ks,"dilations":dils,"dropout":dr,"batch_size":bs},m
    idx=[]; vals=[]
    for ts, X_seq, last_in in build_sequences_for_predictions_agg(train_df,test_df,feats,best_c['seq_len'],sc):
        y=float(best_m.predict([X_seq.reshape(1,*X_seq.shape), last_in.reshape(1,1)], verbose=0).ravel()[0]); idx.append(ts); vals.append(y)
    return pd.Series(vals,index=pd.Index(idx,name='time_stamp')).sort_index(), {"best_params":best_c,"feature_cols":feats,"scaler":sc,"model":best_m}

def build_transformer_model_agg(L,n_features,d_model,heads,ffu,drop):
    inp=Input(shape=(L,n_features),name="seq_in"); x=Dense(d_model)(inp)
    pos=np.arange(L)[:,None]; i=np.arange(d_model)[None,:]; rate=1/np.power(10000,(2*(i//2))/np.float32(d_model)); ang=pos*rate
    pe=np.zeros((L,d_model)); pe[:,0::2]=np.sin(ang[:,0::2]); pe[:,1::2]=np.cos(ang[:,1::2]); pe=tf.convert_to_tensor(pe,dtype=tf.float32)
    x=x+pe; att=MultiHeadAttention(num_heads=heads,key_dim=max(1,d_model//max(1,heads)))(x,x); x=LayerNormalization()(x+att)
    ff=Sequential([Dense(ffu,activation='relu'),Dropout(drop),Dense(d_model)]); x=LayerNormalization()(x+ff(x)); x=GlobalAveragePooling1D()(x)
    last=Input(shape=(1,),dtype='float32',name="last_in"); delta=Dense(1)(x); out=tf.keras.layers.Add()([last,delta])
    m=Model(inputs=[inp,last],outputs=out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,clipnorm=1.0),loss='mse'); return m
def train_eval_transformer_agg(train_df,test_df,cfg,out_dir):
    timef=['hour','dayofweek','is_weekend']; exog=[c for c in ['active_devices','mean_voltage','sum_apparent_power','sum_reactive_power','sum_current','mean_factor'] if c in train_df.columns]
    if not set(timef).issubset(train_df.columns): train_df=add_time_features(train_df)
    if not set(timef).issubset(test_df.columns): test_df=add_time_features(test_df)
    feats=['total_power']+timef+exog; sc=StandardScaler().fit(train_df[feats].astype(np.float32)); cut=train_df['time_stamp'].quantile(0.9)
    best_m=None; best_c=None; br=np.inf
    for L in cfg.transformer_grid['seq_len']:
        dtr,dva=build_tf_datasets_agg(train_df,feats,L,sc,cut)
        for dm in cfg.transformer_grid['d_model']:
            for h in cfg.transformer_grid['num_heads']:
                for ff in cfg.transformer_grid['ff_units']:
                    for dr in cfg.transformer_grid['dropout']:
                        for bs in cfg.transformer_grid['batch_size']:
                            m=build_transformer_model_agg(L,len(feats),dm,h,ff,dr)
                            cbs=[EarlyStopping(monitor='val_loss',patience=cfg.transformer_patience,restore_best_weights=True,verbose=0), ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=max(2,cfg.transformer_patience//2),min_lr=1e-5,verbose=0)]
                            m.fit(dtr.batch(bs).prefetch(tf.data.AUTOTUNE),validation_data=dva.batch(bs).prefetch(tf.data.AUTOTUNE),epochs=cfg.transformer_epochs,verbose=0,callbacks=cbs)
                            rmse=float(np.sqrt(m.evaluate(dva.batch(bs),verbose=0)))
                            if rmse<br: br,best_c,best_m=rmse,{"seq_len":L,"d_model":dm,"num_heads":h,"ff_units":ff,"dropout":dr,"batch_size":bs},m
    idx=[]; vals=[]
    for ts, X_seq, last_in in build_sequences_for_predictions_agg(train_df,test_df,feats,best_c['seq_len'],sc):
        y=float(best_m.predict([X_seq.reshape(1,*X_seq.shape), last_in.reshape(1,1)], verbose=0).ravel()[0]); idx.append(ts); vals.append(y)
    return pd.Series(vals,index=pd.Index(idx,name='time_stamp')).sort_index(), {"best_params":best_c,"feature_cols":feats,"scaler":sc,"model":best_m}

def compute_metrics(y_true,y_pred):
    aligned=pd.concat([y_true.rename('y'),y_pred.rename('yhat')],axis=1).dropna()
    if len(aligned)==0: return {"MAE":np.nan,"MSE":np.nan,"RMSE":np.nan,"MAPE%":np.nan,"sMAPE%":np.nan}
    y=aligned['y'].values; yhat=aligned['yhat'].values
    return {"MAE":float(mean_absolute_error(y,yhat)),"MSE":float(mean_squared_error(y,yhat)),"RMSE":float(mean_squared_error(y,yhat,squared=False)),"MAPE%":float(mape(y,yhat)),"sMAPE%":float(smape(y,yhat))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv_path",required=True); ap.add_argument("--out_dir",default="outputs_global")
    ap.add_argument("--resample_rule",default="1H"); ap.add_argument("--aggregate_total",action="store_true")
    args=ap.parse_args(); cfg=Config(csv_path=args.csv_path,out_dir=args.out_dir,resample_rule=args.resample_rule,aggregate_total=args.aggregate_total)
    ensure_dir(cfg.out_dir); ensure_dir(os.path.join(cfg.out_dir,"plots")); ensure_dir(os.path.join(cfg.out_dir,"models")); ensure_dir(os.path.join(cfg.out_dir,"predictions"))
    print("[1/7] Loading & preprocessing..."); raw=load_and_preprocess(cfg.csv_path); rs=resample_per_device(raw,cfg.resample_rule,cfg.max_interp_minutes); rs=remove_blackouts(rs,cfg.blackout_ranges); rs=add_time_features(rs)
    if cfg.aggregate_total:
        print("[2/7] Aggregating to total series..."); agg=aggregate_total_series(rs); agg=add_time_features(agg)
        print("[3/7] Feature engineering (tabular)..."); fe=add_lag_roll_total(agg, cfg.lags+(24,), cfg.roll_windows+(24,))
        print("[4/7] Train/test split..."); train_df,test_df,cutoff=build_train_test(fe,cfg.test_fraction_time); print(f"Cutoff time: {cutoff}")
        y_true=test_df.set_index('time_stamp')['total_power'].sort_index()
        print("[5/7] Baselines..."); steps_daily=steps_for_period(cfg.resample_rule,"1D"); naive_last=forecast_naive_last_agg(test_df,train_df); naive_seas=forecast_naive_seasonal_agg(test_df,train_df,steps_daily)
        preds={}; details={}; print("[6/7] Training & forecasting (aggregate)...")
        print("  - GBM"); y_gbm,info=train_eval_gbm_agg(train_df,test_df,cfg); preds['GBM']=y_gbm; details['GBM']=info
        try:
            import joblib; joblib.dump(info['model'], os.path.join(cfg.out_dir,"models","gbm_agg_model.joblib"))
            with open(os.path.join(cfg.out_dir,"models","gbm_agg_meta.json"),"w") as f: json.dump({"best_params":info['best_params'],"n_features":info['n_features'],"feature_cols":info.get('feature_cols',[])},f,indent=2)
        except Exception as e: print("[WARN] save GBM:",e)
        print("  - KNN"); y_knn,info=train_eval_knn_agg(train_df,test_df,cfg); preds['KNN']=y_knn; details['KNN']=info
        try:
            import joblib; joblib.dump(info['model'], os.path.join(cfg.out_dir,"models","knn_agg_model.joblib")); joblib.dump(info['scaler'], os.path.join(cfg.out_dir,"models","knn_agg_scaler.joblib"))
            with open(os.path.join(cfg.out_dir,"models","knn_agg_meta.json"),"w") as f: json.dump({"best_params":info['best_params'],"feature_cols":info.get('feature_cols',[])},f,indent=2)
        except Exception as e: print("[WARN] save KNN:",e)
        train_tf,test_tf,_=build_train_test(agg,cfg.test_fraction_time)
        if HAS_TF:
            print("  - GRU"); y_gru,info=train_eval_rnn_agg(train_tf,test_tf,cfg,cfg.out_dir); preds['GRU']=y_gru; details['GRU']=info
            try:
                import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); info['model'].save(d/"gru_agg_best.keras"); joblib.dump(info['scaler'], d/"gru_agg_scaler.joblib")
                meta={k:v for k,v in info.items() if k not in ("model","scaler")}; meta["scaler_path"]=str(d/"gru_agg_scaler.joblib"); open(d/"gru_agg_meta.json","w").write(json.dumps(meta,indent=2))
            except Exception as e: print("[WARN] save GRU:",e)
            print("  - TCN"); y_tcn,info=train_eval_tcn_agg(train_tf,test_tf,cfg,cfg.out_dir); preds['TCN']=y_tcn; details['TCN']=info
            try:
                import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); info['model'].save(d/"tcn_agg_best.keras"); joblib.dump(info['scaler'], d/"tcn_agg_scaler.joblib")
                meta={k:v for k,v in info.items() if k not in ("model","scaler")}; meta["scaler_path"]=str(d/"tcn_agg_scaler.joblib"); open(d/"tcn_agg_meta.json","w").write(json.dumps(meta,indent=2))
            except Exception as e: print("[WARN] save TCN:",e)
            print("  - Transformer"); y_trf,info=train_eval_transformer_agg(train_tf,test_tf,cfg,cfg.out_dir); preds['Transformer']=y_trf; details['Transformer']=info
            try:
                import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); info['model'].save(d/"transformer_agg_best.keras"); joblib.dump(info['scaler'], d/"transformer_agg_scaler.joblib")
                meta={k:v for k,v in info.items() if k not in ("model","scaler")}; meta["scaler_path"]=str(d/"transformer_agg_scaler.joblib"); open(d/"transformer_agg_meta.json","w").write(json.dumps(meta,indent=2))
            except Exception as e: print("[WARN] save Transformer:",e)
        else:
            print("[WARN] TensorFlow not available; skipping deep models.")
        print("[7/7] Evaluating & saving...")
        rows=[{"model":k,**compute_metrics(y_true,v)} for k,v in preds.items()]
        rows.append({"model":"NaiveLast",**compute_metrics(y_true,naive_last)}); rows.append({"model":"NaiveSeasonalDaily",**compute_metrics(y_true,naive_seas)})
        res=pd.DataFrame(rows).sort_values('RMSE'); res.to_csv(os.path.join(cfg.out_dir,"metrics_summary_agg.csv"),index=False); print(res)
        comb=pd.DataFrame({"time_stamp":y_true.index,"y":y_true.values})
        for k,v in preds.items(): comb=comb.merge(v.rename(f'yhat_{k}').reset_index(), on='time_stamp', how='left')
        comb.to_csv(os.path.join(cfg.out_dir,"predictions","all_predictions_agg.csv"),index=False)
        fig=go.Figure(); 
        for m in ['MAE','RMSE','sMAPE%']: fig.add_trace(go.Bar(x=res['model'],y=res[m],name=m))
        fig.update_layout(barmode='group',title="Aggregate Total — Model Comparison",xaxis_title="Model",yaxis_title="Score",width=1200,height=500)
        fig.write_html(os.path.join(cfg.out_dir,"plots","agg_model_comparison.html"))
        try: fig.write_image(os.path.join(cfg.out_dir,"plots","agg_model_comparison.png"),width=1200,height=500,scale=2)
        except Exception as e: print("[WARN] kaleido:",e)
        fig2=go.Figure(); fig2.add_trace(go.Scatter(x=comb['time_stamp'],y=comb['y'],mode='lines',name='Actual'))
        for k in preds.keys(): fig2.add_trace(go.Scatter(x=comb['time_stamp'],y=comb[f'yhat_{k}'],mode='lines',name=k))
        fig2.update_layout(title="Aggregate Total Power — Actual vs Predicted (Test Period)",xaxis_title="Time",yaxis_title="Total Power (W)",width=1200,height=500)
        fig2.write_html(os.path.join(cfg.out_dir,"plots","agg_actual_vs_pred.html"))
        try: fig2.write_image(os.path.join(cfg.out_dir,"plots","agg_actual_vs_pred.png"),width=1200,height=500,scale=2)
        except Exception as e: print("[WARN] kaleido:",e)
        open(os.path.join(cfg.out_dir,"experiment_manifest_agg.json"),"w").write(json.dumps({"cutoff_time":str(cutoff),"models":list(preds.keys()),"config":asdict(cfg)},indent=2))
        print("Done. Aggregate results saved under:", cfg.out_dir)
    else:
        print("[2/7] Feature engineering (tabular)..."); fe=add_lag_roll_time_features(rs,cfg.lags,cfg.roll_windows)
        print("[3/7] Train/test split..."); train_df,test_df,cutoff=build_train_test(fe,cfg.test_fraction_time); print(f"Cutoff time: {cutoff}")
        y_true=test_df.set_index(['time_stamp','device_id'])['power'].sort_index()
        print("[4/7] Baselines..."); steps_daily=steps_for_period(cfg.resample_rule,"1D"); naive_last=forecast_naive_last(test_df,train_df); naive_seas=forecast_naive_seasonal(test_df,train_df,steps_daily)
        preds={}; details={}; print("[5/7] Training & forecasting...")
        print("  - GBM"); y_gbm,info=train_eval_gbm(train_df,test_df,cfg); preds['GBM']=y_gbm; details['GBM']=info
        try:
            import joblib; joblib.dump(info['model'], os.path.join(cfg.out_dir,"models","gbm_model.joblib")); open(os.path.join(cfg.out_dir,"models","gbm_meta.json"),"w").write(json.dumps({"best_params":info['best_params'],"n_features":info['n_features']},indent=2))
        except Exception as e: print("[WARN] save GBM:",e)
        print("  - KNN"); y_knn,info=train_eval_knn(train_df,test_df,cfg); preds['KNN']=y_knn; details['KNN']=info
        try:
            import joblib; joblib.dump(info['model'], os.path.join(cfg.out_dir,"models","knn_model.joblib")); joblib.dump(info['scaler'], os.path.join(cfg.out_dir,"models","knn_scaler.joblib"))
        except Exception as e: print("[WARN] save KNN:",e)
        train_tf,test_tf,_=build_train_test(rs,cfg.test_fraction_time)
        if HAS_TF:
            print("  - GRU"); y_gru,info=train_eval_rnn(train_tf,test_tf,cfg,cfg.out_dir); preds['GRU']=y_gru; details['GRU']=info
            try:
                import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); info['model'].save(d/"gru_best.keras"); joblib.dump(info['scaler'], d/"gru_scaler.joblib")
                meta={k:v for k,v in info.items() if k not in ("model","scaler")}; meta["scaler_path"]=str(d/"gru_scaler.joblib"); open(d/"gru_meta.json","w").write(json.dumps(meta,indent=2))
            except Exception as e: print("[WARN] save GRU:",e)
            print("  - TCN"); y_tcn,info=train_eval_tcn(train_tf,test_tf,cfg,cfg.out_dir); preds['TCN']=y_tcn; details['TCN']=info
            try:
                import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); info['model'].save(d/"tcn_best.keras"); joblib.dump(info['scaler'], d/"tcn_scaler.joblib")
                meta={k:v for k,v in info.items() if k not in ("model","scaler")}; meta["scaler_path"]=str(d/"tcn_scaler.joblib"); open(d/"tcn_meta.json","w").write(json.dumps(meta,indent=2))
            except Exception as e: print("[WARN] save TCN:",e)
            print("  - Transformer"); y_trf,info=train_eval_transformer(train_tf,test_tf,cfg,cfg.out_dir); preds['Transformer']=y_trf; details['Transformer']=info
            try:
                import joblib, pathlib; d=pathlib.Path(cfg.out_dir)/"models"; d.mkdir(parents=True,exist_ok=True); info['model'].save(d/"transformer_best.keras"); joblib.dump(info['scaler'], d/"transformer_scaler.joblib")
                meta={k:v for k,v in info.items() if k not in ("model","scaler")}; meta["scaler_path"]=str(d/"transformer_scaler.joblib"); open(d/"transformer_meta.json","w").write(json.dumps(meta,indent=2))
            except Exception as e: print("[WARN] save Transformer:",e)
        else:
            print("[WARN] TensorFlow not available; skipping deep models.")
        print("[6/7] Evaluating & saving...")
        rows=[{"model":k,**compute_metrics(y_true,v)} for k,v in preds.items()]; rows.append({"model":"NaiveLast",**compute_metrics(y_true,naive_last)}); rows.append({"model":"NaiveSeasonalDaily",**compute_metrics(y_true,naive_seas)})
        res=pd.DataFrame(rows).sort_values('RMSE'); res.to_csv(os.path.join(cfg.out_dir,"metrics_summary.csv"),index=False); print(res)
        comb=test_df.set_index(['time_stamp','device_id'])[['power']].rename(columns={'power':'y'}).sort_index()
        for k,v in preds.items(): comb[f'yhat_{k}']=v
        comb.reset_index().to_csv(os.path.join(cfg.out_dir,"predictions","all_predictions.csv"),index=False)
        fig=go.Figure(); 
        for m in ['MAE','RMSE','sMAPE%']: fig.add_trace(go.Bar(x=res['model'],y=res[m],name=m))
        fig.update_layout(barmode='group',title="Global Multivariate Model Comparison",xaxis_title="Model",yaxis_title="Score",width=1200,height=500)
        fig.write_html(os.path.join(cfg.out_dir,"plots","global_model_comparison.html"))
        try: fig.write_image(os.path.join(cfg.out_dir,"plots","global_model_comparison.png"),width=1200,height=500,scale=2)
        except Exception as e: print("[WARN] kaleido:",e)
        dev_sample=test_df['device_id'].value_counts().idxmax(); s=test_df[test_df['device_id']==dev_sample][['time_stamp','device_id','power']].sort_values('time_stamp')
        fig2=go.Figure(); fig2.add_trace(go.Scatter(x=s['time_stamp'],y=s['power'],mode='lines',name='Actual'))
        for k,ser in preds.items(): d=ser.reset_index().query('device_id == @dev_sample')[['time_stamp','yhat']].sort_values('time_stamp'); fig2.add_trace(go.Scatter(x=d['time_stamp'],y=d['yhat'],mode='lines',name=k))
        fig2.update_layout(title=f"Actual vs Predicted — Device {dev_sample} (Test Period)",xaxis_title="Time",yaxis_title="Power (W)",width=1200,height=500)
        fig2.write_html(os.path.join(cfg.out_dir,"plots",f"actual_vs_pred_{dev_sample}.html"))
        try: fig2.write_image(os.path.join(cfg.out_dir,"plots",f"actual_vs_pred_{dev_sample}.png"),width=1200,height=500,scale=2)
        except Exception as e: print("[WARN] kaleido:",e)
        open(os.path.join(cfg.out_dir,"experiment_manifest.json"),"w").write(json.dumps({"cutoff_time":str(cutoff),"models":list(preds.keys()),"config":asdict(cfg)},indent=2))
        print("Done. Global results saved under:", cfg.out_dir)
if __name__=="__main__": main()
