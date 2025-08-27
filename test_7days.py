import pandas as pd
import os
import sys

print("=== 7days Test Script ===")
print(f"Current directory: {os.getcwd()}")

# Check input file
input_file = "mumbai_static_areas_unique.csv"
print(f"Checking for input file: {input_file}")
print(f"File exists: {os.path.exists(input_file)}")

# Try to load the CSV
if os.path.exists(input_file):
    try:
        print("Loading CSV file...")
        df = pd.read_csv(input_file)
        print(f"Successfully loaded CSV with {len(df)} rows")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Print first few rows for validation
        print("\nFirst 2 rows of data:")
        print(df.head(2))
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
else:
    print(f"ERROR: Input file not found!")
    
# Check if data directory exists
data_dir = "data"
print(f"\nChecking data directory: {data_dir}")
print(f"Directory exists: {os.path.exists(data_dir)}")

# Check output file
output_file = "data/mumbai_regions_7day_forecast.csv"
print(f"Checking output file: {output_file}")
print(f"File exists: {os.path.exists(output_file)}")

if os.path.exists(output_file):
    try:
        print("Loading output CSV file...")
        out_df = pd.read_csv(output_file)
        print(f"Output CSV has {len(out_df)} rows")
    except Exception as e:
        print(f"Error loading output CSV: {e}")

print("\n=== Test Complete ===")
