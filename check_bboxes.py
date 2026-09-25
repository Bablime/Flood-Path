import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=== 1. Local Road DataSet ===")
try:
    df_road = pd.read_csv('unzipped_data/local_road/Local Road DataSet.csv')
    min_lat = min(df_road['start_lat'].min(), df_road['end_lat'].min())
    max_lat = max(df_road['start_lat'].max(), df_road['end_lat'].max())
    min_lon = min(df_road['start_lon'].min(), df_road['end_lon'].min())
    max_lon = max(df_road['start_lon'].max(), df_road['end_lon'].max())
    print(f"Bounding Box:")
    print(f"  Lat: [{min_lat:.4f}, {max_lat:.4f}]")
    print(f"  Lon: [{min_lon:.4f}, {max_lon:.4f}]")
except Exception as e:
    print(f"Error: {e}")

print("\n=== 2. Flood Shelter DataSet ===")
try:
    df_shelter = pd.read_excel(
        'unzipped_data/flood_shelter/reach_bgd_database_cyclone-shelters-in-ukhiya-and-teknaf_november_2019(Flood Shelter Location).xlsx',
        sheet_name='DRRO + other identified shelter'
    )
    # Strip spaces from column names just in case
    df_shelter.columns = df_shelter.columns.str.strip()
    min_lat = df_shelter['lat'].min()
    max_lat = df_shelter['lat'].max()
    min_lon = df_shelter['lon'].min()
    max_lon = df_shelter['lon'].max()
    print(f"Bounding Box:")
    print(f"  Lat: [{min_lat:.4f}, {max_lat:.4f}]")
    print(f"  Lon: [{min_lon:.4f}, {max_lon:.4f}]")
    print("Coverage (Unique):")
    print(f"  Divisions: {df_shelter['Division'].dropna().unique().tolist()}")
    print(f"  Districts: {df_shelter['District'].dropna().unique().tolist()}")
    print(f"  Upazilas: {df_shelter['Upazila'].dropna().unique().tolist()}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== 3. Past Flood Level Dataset ===")
try:
    df_flood = pd.read_excel(
        'unzipped_data/past_flood/geolocations_stations(Past Flood Levels).xlsx'
    )
    df_flood.columns = df_flood.columns.str.strip()
    min_lat = df_flood['Latitude'].min()
    max_lat = df_flood['Latitude'].max()
    min_lon = df_flood['Longitude'].min()
    max_lon = df_flood['Longitude'].max()
    print(f"Bounding Box:")
    print(f"  Lat: [{min_lat:.4f}, {max_lat:.4f}]")
    print(f"  Lon: [{min_lon:.4f}, {max_lon:.4f}]")
    print("Coverage (Unique):")
    print(f"  Divisions: {df_flood['Division'].dropna().unique().tolist()}")
    print(f"  Districts: {df_flood['District'].dropna().unique().tolist()}")
except Exception as e:
    print(f"Error: {e}")
