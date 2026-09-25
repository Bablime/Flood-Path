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

def evaluate_threshold(threshold_km, locations, roads, shelters, node_to_idx, idx_to_node, dist_matrix):
    n = len(locations)
    dsu = DSU(n)
    
    # 1. Add original edges
    for _, row in roads.iterrows():
        u = node_to_idx[row['from_id']]
        v = node_to_idx[row['to_id']]
        dsu.union(u, v)
        
    # 2. Find all valid edges
    row_idx, col_idx = np.triu_indices(n, k=1)
    dists = dist_matrix[row_idx, col_idx]
    
    valid_mask = dists <= threshold_km
    valid_rows = row_idx[valid_mask]
    valid_cols = col_idx[valid_mask]
    valid_dists = dists[valid_mask]
    
    # Sort edges by distance
    sorted_indices = np.argsort(valid_dists)
    
    added_edges = 0
    for idx in sorted_indices:
        u = valid_rows[idx]
        v = valid_cols[idx]
        if dsu.union(u, v):
            added_edges += 1
            
    # Compute component stats
    components = dsu.get_components()
    largest_cc = max(components, key=len)
    largest_cc_nodes = set(idx_to_node[i] for i in largest_cc)
    
    # Shelters in largest cc
    node_lats = locations['lat'].values
    node_lons = locations['lon'].values
    node_ids = locations['location_id'].values
    
    shelters_in_largest = 0
    for _, shelter in shelters.iterrows():
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
            shelters_in_largest += 1
            
    print(f"--- Results for threshold {threshold_km} km ---")
    print(f"Bridged edges added: {added_edges}")
    print(f"Total number of connected components: {len(components)}")
    print(f"Size of largest connected component: {len(largest_cc)} nodes")
    print(f"Number of road nodes outside largest component: {n - len(largest_cc)} (out of {n})")
    print(f"Number of shelters in largest component: {shelters_in_largest} (out of {len(shelters)})")
    print(f"Number of shelters outside largest component: {len(shelters) - shelters_in_largest}")
    print()

def main():
    roads = pd.read_csv('data/roads.csv')
    locations = pd.read_csv('data/locations.csv')
    shelters = pd.read_csv('data/shelters.csv')
    
    lats = locations['lat'].values
    lons = locations['lon'].values
    dist_matrix = haversine_dist_matrix(lats, lons)
    
    node_to_idx = {node: i for i, node in enumerate(locations['location_id'])}
    idx_to_node = {i: node for i, node in enumerate(locations['location_id'])}
    
    print("Evaluating bridging thresholds...\n")
    evaluate_threshold(1.0, locations, roads, shelters, node_to_idx, idx_to_node, dist_matrix)
    evaluate_threshold(2.0, locations, roads, shelters, node_to_idx, idx_to_node, dist_matrix)
    
if __name__ == '__main__':
    main()
