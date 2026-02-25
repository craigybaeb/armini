#!/usr/bin/env python3
"""
Quick script to examine E-day data file structure and identify available columns.
This will help debug the KeyError: 'E-day/daily power generation'
"""

import sys
import pandas as pd
import numpy as np

def analyze_eday_file(filepath):
    """Analyze the structure of an E-day data file"""
    print(f"🔍 Analyzing file: {filepath}")
    print("=" * 60)
    
    try:
        # Try reading the file with error handling for malformed CSV
        print("🔧 Attempting to read CSV file...")
        
        # First, try normal reading
        try:
            df = pd.read_csv(filepath)
            print(f"✅ File loaded successfully!")
            print(f"📊 Shape: {df.shape} (rows × columns)")
        except pd.errors.ParserError as e:
            print(f"⚠️  CSV parsing error: {e}")
            print("🔧 Trying alternative parsing methods...")
            
            # Try with error handling
            try:
                df = pd.read_csv(filepath, on_bad_lines='skip')
                print(f"✅ File loaded with some bad lines skipped!")
                print(f"📊 Shape: {df.shape} (rows × columns)")
            except:
                # Try with different settings
                try:
                    df = pd.read_csv(filepath, sep=',', quoting=1, on_bad_lines='skip')
                    print(f"✅ File loaded with bad lines skipped!")
                    print(f"📊 Shape: {df.shape} (rows × columns)")
                except:
                    # Last resort - read first few lines to get column names
                    print("🔧 Trying to read just the header and first few rows...")
                    df = pd.read_csv(filepath, nrows=100, on_bad_lines='skip')
                    print(f"✅ Loaded first 100 rows for column analysis!")
                    print(f"📊 Shape: {df.shape} (rows × columns)")
        print()
        
        # Show column names
        print("📋 Available columns:")
        print("-" * 30)
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. '{col}'")
        print()
        
        # Look for potential E-day related columns
        eday_candidates = []
        keywords = ['e-day', 'eday', 'daily', 'generation', 'power', 'energy']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in keywords):
                eday_candidates.append(col)
        
        if eday_candidates:
            print("🎯 Potential E-day related columns:")
            print("-" * 35)
            for i, col in enumerate(eday_candidates, 1):
                print(f"{i}. '{col}'")
            print()
            
            # Show basic stats for these columns
            print("📈 Basic statistics for candidate columns:")
            print("-" * 45)
            for col in eday_candidates:
                if df[col].dtype in ['int64', 'float64']:
                    try:
                        stats = df[col].describe()
                        print(f"\n'{col}':")
                        print(f"  Count: {stats['count']:.0f}")
                        print(f"  Mean:  {stats['mean']:.2f}")
                        print(f"  Min:   {stats['min']:.2f}")
                        print(f"  Max:   {stats['max']:.2f}")
                        print(f"  Std:   {stats['std']:.2f}")
                    except Exception as e:
                        print(f"\n'{col}': Error calculating stats - {e}")
                else:
                    print(f"\n'{col}': Non-numeric column")
                    print(f"  Sample values: {list(df[col].dropna().head(3))}")
        else:
            print("⚠️  No obvious E-day related columns found.")
            print("    Here are the first few rows of data:")
            print(df.head())
        
        # Show first few rows anyway
        print("\n" + "=" * 60)
        print("🔬 First 5 rows of data:")
        print("-" * 25)
        print(df.head())
        
        # Check for any columns with values around 828 (the expected mean)
        print("\n" + "=" * 60)
        print("🔍 Looking for columns with values around 828 (expected mean):")
        print("-" * 55)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                mean_val = df[col].mean()
                if 700 <= mean_val <= 1000:  # Range around 828
                    print(f"  ✅ '{col}': mean = {mean_val:.2f}")
                elif df[col].max() >= 700:  # Check if max is in range
                    print(f"  📊 '{col}': mean = {mean_val:.2f}, max = {df[col].max():.2f}")
            except:
                continue
        
    except FileNotFoundError:
        print(f"❌ Error: File not found!")
        print(f"   Make sure the path is correct: {filepath}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        print(f"   File might be corrupted or in unexpected format.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_eday_columns.py <path_to_eday_file.csv>")
        print()
        print("Example:")
        print("  python check_eday_columns.py your_eday_data.csv")
        return
    
    filepath = sys.argv[1]
    analyze_eday_file(filepath)
    
    print("\n" + "=" * 60)
    print("💡 Next steps:")
    print("   1. Identify the correct column name from the list above")
    print("   2. Use that exact column name in your analysis scripts")
    print("   3. Check if the scaling issue (828 → 7) appears in any column")

if __name__ == '__main__':
    main()