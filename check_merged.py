import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv('unzipped_data/cleaned/Cleaned & Merged DataSet.csv')
    
    print("=== 1. Full Dataset Bounding Box ===")
    min_lat = df['start_lat'].min()
    max_lat = df['start_lat'].max()
    min_lon = df['start_lon'].min()
    max_lon = df['start_lon'].max()
    print(f"Lat: [{min_lat:.4f}, {max_lat:.4f}]")
    print(f"Lon: [{min_lon:.4f}, {max_lon:.4f}]")
    
    print("\n=== 2. Rows with BOTH True ===")
    both_true = df[(df['flood_data_available'] == True) & (df['shelter_data_available'] == True)]
    print(f"Count: {len(both_true)} out of {len(df)}")
    
    if len(both_true) > 0:
        print("\n=== 3. Bounding Box for BOTH True ===")
        min_lat_b = both_true['start_lat'].min()
        max_lat_b = both_true['start_lat'].max()
        min_lon_b = both_true['start_lon'].min()
        max_lon_b = both_true['start_lon'].max()
        print(f"Lat: [{min_lat_b:.4f}, {max_lat_b:.4f}]")
        print(f"Lon: [{min_lon_b:.4f}, {max_lon_b:.4f}]")
except Exception as e:
    print(f"Error: {e}")
