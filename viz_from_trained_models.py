#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Plotly plots from saved predictions.

Inputs
------
--predictions_csv : path to predictions CSV produced by the experiment script
                    (global: outputs_global/predictions/all_predictions.csv
                     aggregate: outputs_global/predictions/all_predictions_agg.csv)
--out_dir         : directory to save plots and derived CSVs
--device_id       : (optional, for global predictions) device to visualize in actual-vs-pred plot;
                    if omitted, the most frequent device in the CSV is used.
--clip_pct        : (optional float) winsorize residuals for plotting only (e.g., 1 or 2)
--metrics_csv     : (optional) if provided, use this for the bar chart; otherwise metrics are computed from predictions

What it produces
----------------
- model_comparison_bar.(html|png)  : Grouped bar chart of MAE / RMSE / sMAPE
- actual_vs_pred.(html|png)        : Actual vs predicted time series (aggregate or single device)
- residual_time_series.(html|png)  : Residuals (yhat - y) over time for each model
- metrics_from_predictions.csv     : Metrics computed (if metrics_csv not supplied)
- errors_long.csv                  : Long-format residuals (per time step / model)

PNG export requires `kaleido` to be installed; HTML is always written.
"""

import os
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

EPS = 1e-6

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def mape(y, yhat):
    d = np.where(np.abs(y) < EPS, EPS, np.abs(y))
    return 100.0 * np.mean(np.abs((y - yhat) / d))

def smape(y, yhat):
    d = np.abs(y) + np.abs(yhat)
    d = np.where(d < EPS, EPS, d)
    return 100.0 * np.mean(2.0 * np.abs(yhat - y) / d)

def load_predictions(predictions_csv: str) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    if 'time_stamp' in df.columns:
        try:
            df['time_stamp'] = pd.to_datetime(df['time_stamp'], utc=True, errors='coerce')
        except Exception:
            pass
    # harmonize target name
    if 'y' not in df.columns:
        if 'power' in df.columns:
            df = df.rename(columns={'power':'y'})
        elif 'total_power' in df.columns:
            df = df.rename(columns={'total_power':'y'})
    return df

def melt_long(df: pd.DataFrame, clip_pct: float = None) -> pd.DataFrame:
    model_cols = [c for c in df.columns if c.startswith('yhat_')]
    id_vars = ['time_stamp','device_id','y']
    id_vars = [c for c in id_vars if c in df.columns]
    m = df[id_vars + model_cols].melt(id_vars=id_vars, value_vars=model_cols,
                                      var_name='model', value_name='yhat')
    m['model'] = m['model'].str.replace('yhat_', '', regex=False)
    m['error'] = m['yhat'] - m['y']
    if clip_pct is not None and 0 < clip_pct < 50:
        lo = np.percentile(m['error'].dropna(), clip_pct)
        hi = np.percentile(m['error'].dropna(), 100 - clip_pct)
        m = m[(m['error'] >= lo) & (m['error'] <= hi)].copy()
    return m

def compute_metrics_from_long(df_long: pd.DataFrame) -> pd.DataFrame:
    out = []
    for model, g in df_long.groupby('model'):
        y = g['y'].to_numpy()
        yhat = g['yhat'].to_numpy()
        e = g['error'].to_numpy()
        out.append({
            'model': model,
            'MAE': float(np.mean(np.abs(e))),
            'MSE': float(np.mean(e**2)),
            'RMSE': float(np.sqrt(np.mean(e**2))),
            'MAPE%': float(mape(y, yhat)),
            'sMAPE%': float(smape(y, yhat)),
        })
    return pd.DataFrame(out).sort_values('RMSE')

def plot_bar(metrics_df: pd.DataFrame, out_path: str):
    # Show three bars per model: MAE, RMSE, sMAPE%
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics_df['model'], y=metrics_df['MAE'], name='MAE'))
    fig.add_trace(go.Bar(x=metrics_df['model'], y=metrics_df['RMSE'], name='RMSE'))
    fig.add_trace(go.Bar(x=metrics_df['model'], y=metrics_df['sMAPE%'], name='sMAPE%'))
    fig.update_layout(barmode='group', title='Model Comparison', xaxis_title='Model', yaxis_title='Score',
                      width=1200, height=500)
    fig.write_html(out_path + '.html')
    try:
        fig.write_image(out_path + '.png', width=1200, height=500, scale=2)
    except Exception as e:
        print("[WARN] PNG export failed; install kaleido. ", e)

def plot_actual_vs_pred(df: pd.DataFrame, out_path: str, device_id: str = None):
    # Aggregate if no device_id column
    if 'device_id' not in df.columns:
        # aggregate: single series
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time_stamp'], y=df['y'], mode='lines', name='Actual'))
        for col in [c for c in df.columns if c.startswith('yhat_')]:
            name = col.replace('yhat_', '')
            fig.add_trace(go.Scatter(x=df['time_stamp'], y=df[col], mode='lines', name=name))
        fig.update_layout(title='Actual vs Predicted — Aggregate',
                          xaxis_title='Time', yaxis_title='Total Power', width=1200, height=500)
    else:
        # global (multi-device). Choose device
        if device_id is None:
            device_id = df['device_id'].value_counts().idxmax()
        d = df[df['device_id'] == device_id].sort_values('time_stamp')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d['time_stamp'], y=d['y'], mode='lines', name='Actual'))
        for col in [c for c in d.columns if c.startswith('yhat_')]:
            name = col.replace('yhat_', '')
            fig.add_trace(go.Scatter(x=d['time_stamp'], y=d[col], mode='lines', name=name))
        fig.update_layout(title=f'Actual vs Predicted — Device {device_id}',
                          xaxis_title='Time', yaxis_title='Power', width=1200, height=500)
    fig.write_html(out_path + '.html')
    try:
        fig.write_image(out_path + '.png', width=1200, height=500, scale=2)
    except Exception as e:
        print("[WARN] PNG export failed; install kaleido. ", e)

def plot_residual_time_series(df_long: pd.DataFrame, out_path: str):
    # Residuals over time, overlaid by model
    fig = go.Figure()
    for model, g in df_long.groupby('model'):
        # If device_id present, aggregate residual by timestamp (mean across devices)
        if 'device_id' in g.columns:
            gd = g.groupby('time_stamp', as_index=False)['error'].mean()
            fig.add_trace(go.Scatter(x=gd['time_stamp'], y=gd['error'], mode='lines', name=model))
        else:
            fig.add_trace(go.Scatter(x=g['time_stamp'], y=g['error'], mode='lines', name=model))
    fig.add_hline(y=0, line_dash='dash', opacity=0.6)
    fig.update_layout(title='Residuals Over Time (yhat - y)',
                      xaxis_title='Time', yaxis_title='Residual', width=1200, height=500)
    fig.write_html(out_path + '.html')
    try:
        fig.write_image(out_path + '.png', width=1200, height=500, scale=2)
    except Exception as e:
        print("[WARN] PNG export failed; install kaleido. ", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions_csv', required=True, help='Path to all_predictions(_agg).csv')
    ap.add_argument('--out_dir', default='viz_from_models', help='Where to save plots')
    ap.add_argument('--device_id', default=None, help='(global only) device to plot for actual-vs-pred')
    ap.add_argument('--clip_pct', type=float, default=None, help='Winsorize residuals for plotting (e.g., 1 or 2)')
    ap.add_argument('--metrics_csv', default=None, help='Optional: metrics CSV to use for bars (else computed)')
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    df = load_predictions(args.predictions_csv)
    df_long = melt_long(df, clip_pct=args.clip_pct)
    df_long.to_csv(os.path.join(args.out_dir, 'errors_long.csv'), index=False)

    # Metrics
    if args.metrics_csv and os.path.exists(args.metrics_csv):
        metrics_df = pd.read_csv(args.metrics_csv)
    else:
        metrics_df = compute_metrics_from_long(df_long)
        metrics_df.to_csv(os.path.join(args.out_dir, 'metrics_from_predictions.csv'), index=False)

    # Plots
    plot_bar(metrics_df, os.path.join(args.out_dir, 'model_comparison_bar'))
    plot_actual_vs_pred(df, os.path.join(args.out_dir, 'actual_vs_pred'), device_id=args.device_id)
    plot_residual_time_series(df_long, os.path.join(args.out_dir, 'residual_time_series'))

    print("Saved plots to:", args.out_dir)

if __name__ == '__main__':
    main()
