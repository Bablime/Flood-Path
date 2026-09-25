import pandas as pd
import numpy as np

def check_connectivity():
    roads = pd.read_csv('data/roads.csv')
    locations = pd.read_csv('data/locations.csv')
    shelters = pd.read_csv('data/shelters.csv')
    
    # Build adjacency list
    adj = {node: [] for node in locations['location_id']}
    for _, row in roads.iterrows():
        u, v = row['from_id'], row['to_id']
        adj[u].append(v)
        adj[v].append(u)
        
    visited = set()
    components = []
    
    for node in adj:
        if node not in visited:
            comp = set()
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append(comp)
            
    num_components = len(components)
    print(f"Total number of connected components: {num_components}")
    
    largest_cc = max(components, key=len)
    print(f"Size of largest connected component: {len(largest_cc)} nodes")
    
    nodes_outside = len(locations) - len(largest_cc)
    print(f"Number of road nodes outside largest component: {nodes_outside} (out of {len(locations)})")
    
    def haversine_dist_vec(lat1, lon1, lat2, lon2):
        R = 6371.0 # km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
        
    node_lats = locations['lat'].values
    node_lons = locations['lon'].values
    node_ids = locations['location_id'].values
    
    shelters_outside = 0
    for idx, shelter in shelters.iterrows():
        s_lat, s_lon = shelter['lat'], shelter['lon']
        dists = haversine_dist_vec(s_lat, s_lon, node_lats, node_lons)
        nearest_idx = np.argmin(dists)
        nearest_node = node_ids[nearest_idx]
        
        if nearest_node not in largest_cc:
            shelters_outside += 1
            
    print(f"Number of shelters outside largest component: {shelters_outside} (out of {len(shelters)})")
    
    # Decision logic
    if len(largest_cc) / len(locations) > 0.8 and (len(shelters) - shelters_outside) / len(shelters) > 0.8:
        print("\nLargest component contains a reasonable majority. Filtering...")
        
        locations_filtered = locations[locations['location_id'].isin(largest_cc)]
        roads_filtered = roads[roads['from_id'].isin(largest_cc) & roads['to_id'].isin(largest_cc)]
        
        locations_filtered.to_csv('data/locations.csv', index=False)
        roads_filtered.to_csv('data/roads.csv', index=False)
        
        print(f"New locations count: {len(locations_filtered)}")
        print(f"New roads count: {len(roads_filtered)}")
    else:
        print("\nGraph is highly fragmented. Not silently filtering!")

if __name__ == '__main__':
    check_connectivity()
