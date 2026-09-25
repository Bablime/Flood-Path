import pandas as pd
import numpy as np

def haversine_dist_matrix(lats, lons):
    lats = np.radians(lats)
    lons = np.radians(lons)
    dlat = lats[:, np.newaxis] - lats
    dlon = lons[:, np.newaxis] - lons
    a = np.sin(dlat/2)**2 + np.cos(lats[:, np.newaxis]) * np.cos(lats) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1]*n
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            return True
        return False
    def get_components(self):
        comps = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in comps:
                comps[root] = []
            comps[root].append(i)
        return list(comps.values())

def main():
    threshold_km = 1.0
    
    roads = pd.read_csv('data/roads.csv')
    locations = pd.read_csv('data/locations.csv')
    shelters = pd.read_csv('data/shelters.csv')
    
    n = len(locations)
    dsu = DSU(n)
    
    node_to_idx = {node: i for i, node in enumerate(locations['location_id'])}
    idx_to_node = {i: node for i, node in enumerate(locations['location_id'])}
    
    # 1. Add edge_type to original roads
    roads['edge_type'] = 'real'
    
    # Add original edges to DSU
    for _, row in roads.iterrows():
        u = node_to_idx[row['from_id']]
        v = node_to_idx[row['to_id']]
        dsu.union(u, v)
        
    # Calculate distances
    lats = locations['lat'].values
    lons = locations['lon'].values
    dist_matrix = haversine_dist_matrix(lats, lons)
    
    # Find all valid edges
    row_idx, col_idx = np.triu_indices(n, k=1)
    dists = dist_matrix[row_idx, col_idx]
    
    valid_mask = dists <= threshold_km
    valid_rows = row_idx[valid_mask]
    valid_cols = col_idx[valid_mask]
    valid_dists = dists[valid_mask]
    
    # Sort edges by distance
    sorted_indices = np.argsort(valid_dists)
    
    new_edges = []
    
    for idx in sorted_indices:
        u = valid_rows[idx]
        v = valid_cols[idx]
        if dsu.union(u, v):
            new_edges.append({
                'from_id': idx_to_node[u],
                'to_id': idx_to_node[v],
                'distance_km': valid_dists[idx],
                'edge_type': 'bridged'
            })
            
    if new_edges:
        new_roads_df = pd.DataFrame(new_edges)
        roads = pd.concat([roads, new_roads_df], ignore_index=True)
        
    # Get largest component
    components = dsu.get_components()
    largest_cc = max(components, key=len)
    largest_cc_nodes = set(idx_to_node[i] for i in largest_cc)
    
    # Filter Locations
    locations_filtered = locations[locations['location_id'].isin(largest_cc_nodes)].copy()
    
    # Filter Roads
    roads_filtered = roads[roads['from_id'].isin(largest_cc_nodes) & roads['to_id'].isin(largest_cc_nodes)].copy()
    
    # Filter Shelters
    node_lats = locations['lat'].values
    node_lons = locations['lon'].values
    node_ids = locations['location_id'].values
    
    shelters_to_keep = []
    for idx, shelter in shelters.iterrows():
        s_lat, s_lon = shelter['lat'], shelter['lon']
        s_lat_rad, s_lon_rad = np.radians(s_lat), np.radians(s_lon)
        n_lat_rad, n_lon_rad = np.radians(node_lats), np.radians(node_lons)
        dlat = s_lat_rad - n_lat_rad
        dlon = s_lon_rad - n_lon_rad
        a = np.sin(dlat/2)**2 + np.cos(n_lat_rad)*np.cos(s_lat_rad)*np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        s_dists = 6371.0 * c
        nearest_node = node_ids[np.argmin(s_dists)]
        
        if nearest_node in largest_cc_nodes:
            shelters_to_keep.append(True)
        else:
            shelters_to_keep.append(False)
            
    shelters_filtered = shelters[shelters_to_keep].copy()
    
    # Save files
    locations_filtered.to_csv('data/locations.csv', index=False)
    roads_filtered.to_csv('data/roads.csv', index=False)
    shelters_filtered.to_csv('data/shelters.csv', index=False)
    
    # Print counts
    print(f"Final locations count: {len(locations_filtered)}")
    print(f"Final road segments count: {len(roads_filtered)}")
    print(f"  - Real edges: {len(roads_filtered[roads_filtered['edge_type'] == 'real'])}")
    print(f"  - Bridged edges: {len(roads_filtered[roads_filtered['edge_type'] == 'bridged'])}")
    print(f"Final shelters count: {len(shelters_filtered)}\n")
    
    print("First 5 rows of locations.csv:")
    print(locations_filtered.head())
    print("\nFirst 5 rows of roads.csv:")
    print(roads_filtered.head())
    print("\nFirst 5 rows of shelters.csv:")
    print(shelters_filtered.head())

if __name__ == '__main__':
    main()
