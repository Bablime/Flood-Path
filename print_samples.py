import pandas as pd
import os

dirs = {
    "Cleaned & Merged DataSet.zip": "unzipped_data/cleaned",
    "Flood Shelter DataSet.zip": "unzipped_data/flood_shelter",
    "Local Road DataSet.zip": "unzipped_data/local_road",
    "Past Flood Level Dataset.zip": "unzipped_data/past_flood",
    "Raw Map DataSet.zip": "unzipped_data/raw_map"
}

for zip_name, d in dirs.items():
    print(f"\n{'='*50}")
    print(f"Contents of: {zip_name}")
    print(f"{'='*50}")
    for root, _, files in os.walk(d):
        for f in files:
            filepath = os.path.join(root, f)
            print(f"\nFile: {f}")
            try:
                if f.endswith('.csv'):
                    df = pd.read_csv(filepath, nrows=5)
                elif f.endswith('.xlsx') or f.endswith('.xls'):
                    df = pd.read_excel(filepath, nrows=5)
                else:
                    continue
                
                print("Columns:", list(df.columns))
                print("\nSample rows (first 3):")
                print(df.head(3).to_string(index=False))
            except Exception as e:
                print(f"Could not read {f}: {e}")
