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
    roads_file = root / 'data/roads.csv'
    shelters_file = root / 'data/shelters.csv'
    locations_file = root / 'data/locations.csv'
    
    roads_df = pd.read_csv(roads_file)
    shelters_df = pd.read_csv(shelters_file)
    locations_df = pd.read_csv(locations_file)
    
    # 1. Road graph nodes
    road_nodes = set(roads_df['from_id']).union(set(roads_df['to_id']))
    
    # 2. Shelter IDs
    shelter_ids = set(shelters_df['shelter_id'])
    
    # 3. Overlap
    overlap = shelter_ids.intersection(road_nodes)
    print(f"Overlap: {len(overlap)} shelters appear in the road graph out of {len(shelter_ids)} total shelters.")
    
    # If overlap is 0 or very low, find nearest road node for each shelter
    if len(overlap) < len(shelter_ids) * 0.1:
        s_lats = shelters_df['lat'].values[:, np.newaxis]
        s_lons = shelters_df['lon'].values[:, np.newaxis]
        n_lats = locations_df['lat'].values
        n_lons = locations_df['lon'].values
        
        # dists shape: (157 shelters, 1883 locations)
        dists = haversine(s_lats, s_lons, n_lats, n_lons)
        min_dists = dists.min(axis=1)
        
        print("\n--- Distance from Shelter to Nearest Road Node ---")
        print(f"Min: {min_dists.min():.4f} km")
        print(f"Median: {np.median(min_dists):.4f} km")
        print(f"Max: {min_dists.max():.4f} km")

if __name__ == '__main__':
    main()
