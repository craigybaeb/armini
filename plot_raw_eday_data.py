#!/usr/bin/env python3
"""
Script to plot raw E-day data and identify scaling issues.
This helps debug the 828 mean -> 7 max scaling problem.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
from datetime import datetime

def load_eday_data(csv_path, format_type='auto'):
    """
    Load E-day data with flexible format detection
    """
    print(f"Loading E-day data from: {csv_path}")
    
    # Try different loading approaches
    try:
        if format_type == 'auto':
            # First, peek at the file
            with open(csv_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            print("First 5 lines of file:")
            for i, line in enumerate(first_lines, 1):
                print(f"  {i}: {line[:100]}...")
            
            # Try loading with header detection
            df = pd.read_csv(csv_path)
            print(f"Auto-detected columns: {df.columns.tolist()}")
            
        elif format_type == 'no_header':
            # Load without header, common E-day format
            df = pd.read_csv(csv_path, header=None)
            print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            
        elif format_type == 'custom':
            # Custom format - adjust column names as needed
            expected_cols = ['timestamp', 'device_id', 'eday_value', 'other1', 'other2']
            df = pd.read_csv(csv_path, header=None, names=expected_cols[:5])
            
        else:
            df = pd.read_csv(csv_path)
    
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying alternative loading methods...")
        try:
            # Try with different separators
            df = pd.read_csv(csv_path, sep=';')
        except:
            try:
                df = pd.read_csv(csv_path, sep='\t')
            except:
                df = pd.read_csv(csv_path, header=None)
    
    print(f"Loaded dataframe shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    return df

def identify_eday_column(df):
    """
    Try to identify which column contains E-day values
    """
    print("\n=== IDENTIFYING E-DAY COLUMN ===")
    
    # Look for likely E-day column names
    eday_candidates = []
    for col in df.columns:
        col_str = str(col).lower()
        if any(keyword in col_str for keyword in ['eday', 'energy', 'supply', 'battery', 'kwh', 'wh']):
            eday_candidates.append(col)
    
    # If no obvious names, look at numeric columns
    if not eday_candidates:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                eday_candidates.append(col)
    
    print(f"E-day candidate columns: {eday_candidates}")
    
    # Analyze each candidate
    for col in eday_candidates:
        try:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) > 0:
                print(f"\nColumn '{col}' statistics:")
                print(f"  Count: {len(values)}")
                print(f"  Mean: {values.mean():.3f}")
                print(f"  Median: {values.median():.3f}")
                print(f"  Min: {values.min():.3f}")
                print(f"  Max: {values.max():.3f}")
                print(f"  Std: {values.std():.3f}")
                print(f"  Sample values: {values.head(10).tolist()}")
        except Exception as e:
            print(f"  Error analyzing column '{col}': {e}")
    
    return eday_candidates

def identify_timestamp_column(df):
    """
    Try to identify timestamp column
    """
    timestamp_candidates = []
    
    # Look for obvious timestamp names
    for col in df.columns:
        col_str = str(col).lower()
        if any(keyword in col_str for keyword in ['time', 'date', 'timestamp', 'ts']):
            timestamp_candidates.append(col)
    
    # If no obvious names, look for columns that might be timestamps
    if not timestamp_candidates:
        for col in df.columns:
            try:
                # Try to parse as datetime
                pd.to_datetime(df[col].dropna().head(100), errors='raise')
                timestamp_candidates.append(col)
            except:
                pass
    
    print(f"Timestamp candidate columns: {timestamp_candidates}")
    return timestamp_candidates

def create_eday_plots(df, eday_col, timestamp_col=None, device_col=None):
    """
    Create comprehensive plots of E-day data
    """
    print(f"\n=== CREATING PLOTS ===")
    
    # Convert E-day values to numeric
    eday_values = pd.to_numeric(df[eday_col], errors='coerce').dropna()
    
    # Handle timestamps
    if timestamp_col and timestamp_col in df.columns:
        try:
            timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
            df_plot = pd.DataFrame({
                'timestamp': timestamps,
                'eday': eday_values
            }).dropna()
            df_plot = df_plot.sort_values('timestamp')
            has_timestamps = True
        except:
            print("Could not parse timestamps, using index")
            has_timestamps = False
    else:
        has_timestamps = False
    
    # Create subplots
    if has_timestamps:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Raw E-day Time Series', 'E-day Value Distribution',
                'Daily E-day Pattern', 'Hourly E-day Pattern', 
                'E-day Differences', 'Cumulative E-day'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
    else:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Raw E-day Values (by Index)', 'E-day Value Distribution',
                'E-day Differences', 'Basic Statistics'
            ]
        )
    
    if has_timestamps:
        # 1. Time series plot
        fig.add_trace(
            go.Scatter(x=df_plot['timestamp'], y=df_plot['eday'], 
                      mode='lines+markers', name='E-day Values',
                      line=dict(width=1), marker=dict(size=3)),
            row=1, col=1
        )
        
        # 2. Distribution
        fig.add_trace(
            go.Histogram(x=eday_values, nbinsx=50, name='Distribution'),
            row=1, col=2
        )
        
        # 3. Daily pattern (if we have enough data)
        if len(df_plot) > 24:
            df_plot['hour'] = df_plot['timestamp'].dt.hour
            daily_pattern = df_plot.groupby('hour')['eday'].mean()
            fig.add_trace(
                go.Scatter(x=daily_pattern.index, y=daily_pattern.values,
                          mode='lines+markers', name='Hourly Average'),
                row=2, col=1
            )
        
        # 4. Day of week pattern
        if len(df_plot) > 7:
            df_plot['dayofweek'] = df_plot['timestamp'].dt.dayofweek
            weekly_pattern = df_plot.groupby('dayofweek')['eday'].mean()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig.add_trace(
                go.Scatter(x=[day_names[i] for i in weekly_pattern.index], 
                          y=weekly_pattern.values,
                          mode='lines+markers', name='Daily Average'),
                row=2, col=2
            )
        
        # 5. Differences (to detect resets)
        df_plot['eday_diff'] = df_plot['eday'].diff()
        fig.add_trace(
            go.Scatter(x=df_plot['timestamp'], y=df_plot['eday_diff'],
                      mode='markers', name='E-day Differences', 
                      marker=dict(size=2)),
            row=3, col=1
        )
        
        # 6. Cumulative (if it makes sense)
        fig.add_trace(
            go.Scatter(x=df_plot['timestamp'], y=df_plot['eday'].cumsum(),
                      mode='lines', name='Cumulative E-day'),
            row=3, col=2
        )
        
    else:
        # Simple plots without timestamps
        # 1. Index-based plot
        fig.add_trace(
            go.Scatter(y=eday_values, mode='lines+markers', name='E-day Values'),
            row=1, col=1
        )
        
        # 2. Distribution
        fig.add_trace(
            go.Histogram(x=eday_values, nbinsx=50, name='Distribution'),
            row=1, col=2
        )
        
        # 3. Differences
        fig.add_trace(
            go.Scatter(y=np.diff(eday_values), mode='markers', name='Differences'),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800 if has_timestamps else 600,
        title_text=f"Raw E-day Data Analysis (Column: {eday_col})",
        showlegend=True
    )
    
    return fig, df_plot if has_timestamps else pd.DataFrame({'eday': eday_values})

def main():
    parser = argparse.ArgumentParser(description='Plot raw E-day data')
    parser.add_argument('csv_path', help='Path to CSV file with E-day data')
    parser.add_argument('--format', choices=['auto', 'no_header', 'custom'], 
                       default='auto', help='CSV format type')
    parser.add_argument('--eday_col', help='Name/index of E-day column')
    parser.add_argument('--timestamp_col', help='Name/index of timestamp column')
    parser.add_argument('--device_col', help='Name/index of device ID column')
    parser.add_argument('--output', default='eday_analysis.html', 
                       help='Output HTML file name')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File {args.csv_path} does not exist")
        return
    
    # Load data
    df = load_eday_data(args.csv_path, args.format)
    
    # Identify columns automatically if not specified
    if args.eday_col:
        eday_col = args.eday_col
        if eday_col.isdigit():
            eday_col = int(eday_col)
    else:
        eday_candidates = identify_eday_column(df)
        if not eday_candidates:
            print("Error: Could not identify E-day column")
            return
        eday_col = eday_candidates[0]  # Use first candidate
        print(f"Using E-day column: {eday_col}")
    
    if args.timestamp_col:
        timestamp_col = args.timestamp_col
        if timestamp_col.isdigit():
            timestamp_col = int(timestamp_col)
    else:
        timestamp_candidates = identify_timestamp_column(df)
        timestamp_col = timestamp_candidates[0] if timestamp_candidates else None
        if timestamp_col:
            print(f"Using timestamp column: {timestamp_col}")
    
    # Create plots
    fig, processed_df = create_eday_plots(df, eday_col, timestamp_col, args.device_col)
    
    # Print summary statistics
    eday_values = pd.to_numeric(df[eday_col], errors='coerce').dropna()
    print(f"\n=== FINAL STATISTICS ===")
    print(f"Total records: {len(eday_values)}")
    print(f"Mean E-day value: {eday_values.mean():.3f}")
    print(f"Median E-day value: {eday_values.median():.3f}")
    print(f"Min E-day value: {eday_values.min():.3f}")
    print(f"Max E-day value: {eday_values.max():.3f}")
    print(f"Standard deviation: {eday_values.std():.3f}")
    
    # Check for potential issues
    print(f"\n=== POTENTIAL ISSUES ===")
    if eday_values.min() < 0:
        print(f"⚠️  Negative values detected (min: {eday_values.min()})")
    
    if eday_values.max() / eday_values.mean() > 100:
        print(f"⚠️  Very high max/mean ratio: {eday_values.max() / eday_values.mean():.1f}")
    
    # Check for resets (large negative differences)
    differences = eday_values.diff()
    large_negative_diffs = differences[differences < -100]
    if len(large_negative_diffs) > 0:
        print(f"⚠️  Detected {len(large_negative_diffs)} potential daily resets")
        print(f"   Largest reset: {large_negative_diffs.min():.3f}")
    
    # Save plot
    fig.write_html(args.output)
    print(f"\nPlot saved to: {args.output}")
    
    # Save processed data
    csv_output = args.output.replace('.html', '_processed.csv')
    processed_df.to_csv(csv_output, index=False)
    print(f"Processed data saved to: {csv_output}")

if __name__ == "__main__":
    main()