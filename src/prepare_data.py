import os
import pandas as pd
import numpy as np

def prepare_data():
    os.makedirs('data', exist_ok=True)
    
    # Bounding Box
    lat_min, lat_max = 20.6, 21.35
    lon_min, lon_max = 92.0, 92.35
    
    # 1. SHELTERS
    shelter_file = 'unzipped_data/flood_shelter/reach_bgd_database_cyclone-shelters-in-ukhiya-and-teknaf_november_2019(Flood Shelter Location).xlsx'
    shelters_df = pd.read_excel(shelter_file, sheet_name='DRRO + other identified shelter')
    
    shelters_mask = (
        (shelters_df['lat'] >= lat_min) & (shelters_df['lat'] <= lat_max) &
        (shelters_df['lon'] >= lon_min) & (shelters_df['lon'] <= lon_max)
    )
    shelters_filtered = shelters_df[shelters_mask].copy()
    
    missing_cap = shelters_filtered['DRRO Capacity'].isna() | (shelters_filtered['DRRO Capacity'] == 0)
    print(f"Number of shelters with missing/zero capacity: {missing_cap.sum()}")
    shelters_filtered.loc[missing_cap, 'DRRO Capacity'] = 500
    
    shelters_out = pd.DataFrame({
        'shelter_id': shelters_filtered['CS_ID'],
        'name': shelters_filtered['Cyclone Shelter Name'],
        'lat': shelters_filtered['lat'],
        'lon': shelters_filtered['lon'],
        'capacity': shelters_filtered['DRRO Capacity']
    })
    shelters_out.to_csv('data/shelters.csv', index=False)
    
    # 2. ROADS
    roads_file = 'unzipped_data/local_road/Local Road DataSet.csv'
    roads_df = pd.read_csv(roads_file)
    roads_mask = (
        (roads_df['start_lat'] >= lat_min) & (roads_df['start_lat'] <= lat_max) &
        (roads_df['start_lon'] >= lon_min) & (roads_df['start_lon'] <= lon_max) &
        (roads_df['end_lat'] >= lat_min) & (roads_df['end_lat'] <= lat_max) &
        (roads_df['end_lon'] >= lon_min) & (roads_df['end_lon'] <= lon_max)
    )
    roads_filtered = roads_df[roads_mask].copy()
    
    roads_filtered['start_lat'] = roads_filtered['start_lat'].round(4)
    roads_filtered['start_lon'] = roads_filtered['start_lon'].round(4)
    roads_filtered['end_lat'] = roads_filtered['end_lat'].round(4)
    roads_filtered['end_lon'] = roads_filtered['end_lon'].round(4)
    
    # Create unique nodes
    starts = roads_filtered[['start_lat', 'start_lon']].rename(columns={'start_lat': 'lat', 'start_lon': 'lon'})
    ends = roads_filtered[['end_lat', 'end_lon']].rename(columns={'end_lat': 'lat', 'end_lon': 'lon'})
    nodes = pd.concat([starts, ends]).drop_duplicates().reset_index(drop=True)
    nodes['location_id'] = ['N' + str(i+1) for i in range(len(nodes))]
    
    # Map back to roads
    node_map = dict(zip(zip(nodes['lat'], nodes['lon']), nodes['location_id']))
    
    roads_filtered['from_id'] = roads_filtered.apply(lambda row: node_map[(row['start_lat'], row['start_lon'])], axis=1)
    roads_filtered['to_id'] = roads_filtered.apply(lambda row: node_map[(row['end_lat'], row['end_lon'])], axis=1)
    
    def haversine_dist_vec(lat1, lon1, lat2, lon2):
        R = 6371.0 # km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
        
    roads_filtered['distance_km'] = haversine_dist_vec(
        roads_filtered['start_lat'], roads_filtered['start_lon'],
        roads_filtered['end_lat'], roads_filtered['end_lon']
    )
    
    roads_out = roads_filtered[['from_id', 'to_id', 'distance_km']]
    roads_out.to_csv('data/roads.csv', index=False)
    
    # 3. LOCATIONS
    # distance_to_coast_km: approximate the coastline for this region as a line at longitude 92.0
    nodes['distance_to_coast_km'] = haversine_dist_vec(nodes['lat'], nodes['lon'], nodes['lat'], pd.Series([92.0]*len(nodes)))
    
    # elevation_m: no elevation data available
    print("No elevation data is available in our files. Setting elevation_m to NaN.")
    nodes['elevation_m'] = np.nan
    
    # historical_flood: 1 if within 2km of a shelter location, else 0
    # We assume that shelters are typically built in flood-prone areas, thus using proximity to shelter as a proxy
    if len(shelters_out) > 0 and len(nodes) > 0:
        s_lat = np.radians(shelters_out['lat'].values).reshape(1, -1)
        s_lon = np.radians(shelters_out['lon'].values).reshape(1, -1)
        n_lat = np.radians(nodes['lat'].values).reshape(-1, 1)
        n_lon = np.radians(nodes['lon'].values).reshape(-1, 1)
        
        dlat = s_lat - n_lat
        dlon = s_lon - n_lon
        
        a = np.sin(dlat / 2)**2 + np.cos(n_lat) * np.cos(s_lat) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        dist_matrix = 6371.0 * c
        
        nodes['historical_flood'] = (np.min(dist_matrix, axis=1) <= 2.0).astype(int)
    else:
        nodes['historical_flood'] = 0
        
    locations_out = nodes[['location_id', 'lat', 'lon', 'distance_to_coast_km', 'elevation_m', 'historical_flood']]
    locations_out.to_csv('data/locations.csv', index=False)
    
    print(f"\n--- Summary ---")
    print(f"Number of shelters: {len(shelters_out)}")
    print(f"Number of unique road nodes: {len(locations_out)}")
    print(f"Number of road segments: {len(roads_out)}\n")
    
    print("First 5 rows of shelters.csv:")
    print(shelters_out.head())
    print("\nFirst 5 rows of roads.csv:")
    print(roads_out.head())
    print("\nFirst 5 rows of locations.csv:")
    print(locations_out.head())

if __name__ == "__main__":
    prepare_data()
