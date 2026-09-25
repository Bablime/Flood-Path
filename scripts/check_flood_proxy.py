import pandas as pd
import numpy as np
import pathlib

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def main():
    root = pathlib.Path('/Users/khansayem/Flood-Path/Flood-Path')
    loc_file = root / 'data/locations.csv'
    shelter_file = root / 'data/shelters.csv'
    
    loc_df = pd.read_csv(loc_file)
    shelter_df = pd.read_csv(shelter_file)
    
    print(f"Loaded {len(loc_df)} locations and {len(shelter_df)} shelters.")
    
    # 1 & 4. Calculate min distance to shelter
    lats = loc_df['lat'].values[:, np.newaxis]
    lons = loc_df['lon'].values[:, np.newaxis]
    s_lats = shelter_df['lat'].values
    s_lons = shelter_df['lon'].values
    
    dists = haversine(lats, lons, s_lats, s_lons)
    min_dists = dists.min(axis=1)
    
    # 4. Describe min_dists
    print("\n--- Distance to Nearest Shelter Stats ---")
    print(f"Min: {min_dists.min():.4f} km")
    print(f"Median: {np.median(min_dists):.4f} km")
    print(f"Max: {min_dists.max():.4f} km")
    pct_under_2km = (min_dists <= 2.0).mean() * 100
    print(f"% <= 2km: {pct_under_2km:.2f}%")
    
    # 2. Compare proxy
    rederived_proxy = (min_dists <= 2.0).astype(int)
    existing_proxy = loc_df['historical_flood'].values
    
    mismatches = (rederived_proxy != existing_proxy).sum()
    print(f"\n--- Proxy Comparison ---")
    print(f"Number of mismatches: {mismatches}")
    
    if mismatches > 0:
        loc_df['rederived'] = rederived_proxy
        loc_df['min_dist'] = min_dists
        diffs = loc_df[loc_df['historical_flood'] != loc_df['rederived']]
        print("Example mismatches:")
        print(diffs.head(10)[['location_id', 'lat', 'lon', 'historical_flood', 'rederived', 'min_dist']])
        
    # 3. Geographic plausibility
    min_lat, max_lat = 20.6, 21.35
    min_lon, max_lon = 92.0, 92.35
    
    height_km = haversine(min_lat, min_lon, max_lat, min_lon)
    width_km_bottom = haversine(min_lat, min_lon, min_lat, max_lon)
    width_km_top = haversine(max_lat, min_lon, max_lat, max_lon)
    width_km = (width_km_bottom + width_km_top) / 2
    bbox_area = height_km * width_km
    print(f"\n--- Geographic Plausibility ---")
    print(f"Bounding Box Area: ~{bbox_area:.2f} km^2")
    
    # Rasterize grid ~100m
    grid_lat = np.arange(min_lat, max_lat, 0.0009)
    grid_lon = np.arange(min_lon, max_lon, 0.0009)
    print(f"Grid shape: {len(grid_lat)} x {len(grid_lon)} ({len(grid_lat)*len(grid_lon)} points)")
    
    covered_cells = 0
    total_cells = len(grid_lat) * len(grid_lon)
    
    GLAT, GLON = np.meshgrid(grid_lat, grid_lon)
    flat_glat = GLAT.ravel()
    flat_glon = GLON.ravel()
    
    batch_size = 10000
    for i in range(0, len(flat_glat), batch_size):
        b_lat = flat_glat[i:i+batch_size][:, np.newaxis]
        b_lon = flat_glon[i:i+batch_size][:, np.newaxis]
        b_dists = haversine(b_lat, b_lon, s_lats, s_lons)
        b_min = b_dists.min(axis=1)
        covered_cells += (b_min <= 2.0).sum()
        
    pct_covered = (covered_cells / total_cells) * 100
    print(f"Grid Coverage: {pct_covered:.2f}% (approx {covered_cells}/{total_cells} cells)")

if __name__ == '__main__':
    main()
