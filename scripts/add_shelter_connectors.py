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
    shelters_file = root / 'data/shelters.csv'
    roads_file = root / 'data/roads.csv'
    risk_scores_file = root / 'data/risk_scores.csv'
    
    loc_df = pd.read_csv(loc_file)
    shelters_df = pd.read_csv(shelters_file)
    roads_df = pd.read_csv(roads_file)
    risk_df = pd.read_csv(risk_scores_file)
    
    s_lats = shelters_df['lat'].values[:, np.newaxis]
    s_lons = shelters_df['lon'].values[:, np.newaxis]
    n_lats = loc_df['lat'].values
    n_lons = loc_df['lon'].values
    
    # Calculate distances
    dists = haversine(s_lats, s_lons, n_lats, n_lons)
    
    min_dists = dists.min(axis=1)
    nearest_idx = dists.argmin(axis=1)
    
    shelter_ids = shelters_df['shelter_id'].values
    nearest_location_ids = loc_df.iloc[nearest_idx]['location_id'].values
    
    new_edges = []
    exceeded_shelters = []
    
    for i, (s_id, loc_id, dist) in enumerate(zip(shelter_ids, nearest_location_ids, min_dists)):
        if dist <= 2.5:
            # Add forward edge
            new_edges.append({'from_id': s_id, 'to_id': loc_id, 'distance_km': dist, 'edge_type': 'shelter_connector'})
            # Add backward edge
            new_edges.append({'from_id': loc_id, 'to_id': s_id, 'distance_km': dist, 'edge_type': 'shelter_connector'})
        else:
            exceeded_shelters.append((s_id, dist))
            
    print(f"Total shelters processed: {len(shelter_ids)}")
    print(f"Total connector edges to add: {len(new_edges)}")
    
    if exceeded_shelters:
        print("\nWARNING: The following shelters exceeded the 2.5km threshold and were NOT connected:")
        for s, d in exceeded_shelters:
            print(f"  - {s}: {d:.4f} km")
    else:
        print("\nAll shelters were within 2.5km and successfully connected.")
        
    # Append edges
    if new_edges:
        new_edges_df = pd.DataFrame(new_edges)
        updated_roads = pd.concat([roads_df, new_edges_df], ignore_index=True)
        updated_roads.to_csv(roads_file, index=False)
        print(f"\nUpdated roads.csv saved. New row count: {len(updated_roads)} (was {len(roads_df)})")
        
    # Add to risk scores
    new_risk_rows = []
    for s_id in shelter_ids:
        new_risk_rows.append({
            'location_id': s_id,
            'risk_label': 'low',
            'risk_cluster': 0,
            'risk_score_knn': 0.0
        })
    
    new_risk_df = pd.DataFrame(new_risk_rows)
    updated_risk = pd.concat([risk_df, new_risk_df], ignore_index=True)
    updated_risk.to_csv(risk_scores_file, index=False)
    
    print(f"\nUpdated risk_scores.csv saved. New row count: {len(updated_risk)} (was {len(risk_df)})")
    
if __name__ == '__main__':
    main()
