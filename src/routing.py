"""
Stage 2 – Routing
=================
Multi-goal A* that finds the cheapest path from every road-network
location (locations.csv, 1 883 nodes) to the nearest shelter
(shelters.csv, 157 nodes), using a risk-aware edge cost function.

Outputs
-------
data/routes.csv  – one row per location with routing results
"""

import csv
import heapq
import itertools
import json
import math
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ROADS_CSV      = os.path.join(DATA_DIR, "roads.csv")
SHELTERS_CSV   = os.path.join(DATA_DIR, "shelters.csv")
LOCATIONS_CSV  = os.path.join(DATA_DIR, "locations.csv")
RISK_CSV       = os.path.join(DATA_DIR, "risk_scores.csv")
ROUTES_CSV     = os.path.join(DATA_DIR, "routes.csv")


# ---------------------------------------------------------------------------
# Haversine distance (returns km)
# ---------------------------------------------------------------------------
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    # risk scores
    risk_scores: dict = {}
    with open(RISK_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            risk_scores[row["location_id"]] = float(row["risk_score_knn"])

    # shelters
    shelters: dict = {}
    with open(SHELTERS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shelters[row["shelter_id"]] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "capacity": row["capacity"],
            }

    # locations (road nodes only, 1883)
    locations: dict = {}
    with open(LOCATIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            locations[row["location_id"]] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "historical_flood": row.get("historical_flood", "0"),
            }

    # roads -> adjacency list
    graph: dict = defaultdict(list)
    with open(ROADS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u     = row["from_id"]
            v     = row["to_id"]
            dist  = float(row["distance_km"])
            etype = row["edge_type"]

            if etype == "shelter_connector":
                # Both directions already present as separate rows
                graph[u].append((v, dist, etype))
            else:
                # real / bridged: stored once, add both directions
                graph[u].append((v, dist, etype))
                graph[v].append((u, dist, etype))

    return risk_scores, shelters, locations, graph


# ---------------------------------------------------------------------------
# Edge cost functions
# ---------------------------------------------------------------------------
def edge_cost(from_id, to_id, dist_km, etype, risk_scores):
    edge_risk      = max(risk_scores.get(from_id, 0.0), risk_scores.get(to_id, 0.0))
    risk_penalty   = edge_risk * 0.5
    bridge_penalty = 0.15 if etype == "bridged" else 0.0
    return dist_km * (1.0 + risk_penalty) * (1.0 + bridge_penalty)


def edge_cost_naive(dist_km):
    return dist_km


# ---------------------------------------------------------------------------
# Build combined coordinate lookup
# ---------------------------------------------------------------------------
def build_coords(locations, shelters):
    coords = {}
    for nid, info in locations.items():
        coords[nid] = (info["lat"], info["lon"])
    for sid, info in shelters.items():
        coords[sid] = (info["lat"], info["lon"])
    return coords


# ---------------------------------------------------------------------------
# Build heuristic (min haversine to any shelter)
# ---------------------------------------------------------------------------
def build_heuristic(shelters):
    shelter_coords = [(info["lat"], info["lon"]) for info in shelters.values()]

    def h(node_lat, node_lon):
        return min(
            haversine(node_lat, node_lon, slat, slon)
            for slat, slon in shelter_coords
        )
    return h


# ---------------------------------------------------------------------------
# Multi-goal A*
# ---------------------------------------------------------------------------
def astar(start, shelter_ids, graph, coords, risk_scores, heuristic, use_risk=True):
    """
    Single search: finds the cheapest path from `start` to whichever shelter
    is reachable at lowest cost.

    Returns a result dict or None if unreachable.
    """
    if start not in coords:
        return None

    counter = itertools.count()
    visited = {}  # node_id -> best g_score finalised

    slat, slon = coords[start]
    h0 = heuristic(slat, slon)
    # heap: (f, tie_counter, node_id, path_list, g_score)
    open_set = []
    heapq.heappush(open_set, (h0, next(counter), start, [start], 0.0))

    while open_set:
        f, _, current, path, g = heapq.heappop(open_set)

        # Skip if a cheaper route to `current` was already finalised
        if current in visited and visited[current] <= g:
            continue
        visited[current] = g

        # Goal test
        if current in shelter_ids:
            raw_dist = 0.0
            n_bridged = 0
            n_shelter_conn = 0
            for u, v in zip(path[:-1], path[1:]):
                best_edge = None
                for nb, d, et in graph[u]:
                    if nb == v:
                        if best_edge is None or d < best_edge[0]:
                            best_edge = (d, et)
                if best_edge is None:
                    ulat, ulon = coords[u]
                    vlat, vlon = coords[v]
                    best_edge = (haversine(ulat, ulon, vlat, vlon), "real")
                d, et = best_edge
                raw_dist += d
                if et == "bridged":
                    n_bridged += 1
                if et == "shelter_connector":
                    n_shelter_conn += 1
            return {
                "nearest_shelter_id": current,
                "path": path,
                "total_cost": g,
                "path_length_km": raw_dist,
                "num_bridged_edges_used": n_bridged,
                "num_shelter_connector_edges_used": n_shelter_conn,
            }

        # Expand neighbours
        for neighbour, dist_km, etype in graph[current]:
            if use_risk:
                step = edge_cost(current, neighbour, dist_km, etype, risk_scores)
            else:
                step = edge_cost_naive(dist_km)
            new_g = g + step

            if neighbour in visited and visited[neighbour] <= new_g:
                continue

            if neighbour in coords:
                nlat, nlon = coords[neighbour]
                h_val = heuristic(nlat, nlon)
            else:
                h_val = 0.0

            new_f = new_g + h_val
            heapq.heappush(
                open_set,
                (new_f, next(counter), neighbour, path + [neighbour], new_g),
            )

    return None  # unreachable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Stage 2 – Multi-Goal A* Routing")
    print("=" * 60)

    # 1. Load
    print("\n[1/4] Loading data ...")
    risk_scores, shelters, locations, graph = load_data()
    shelter_ids = set(shelters.keys())
    coords      = build_coords(locations, shelters)
    heuristic   = build_heuristic(shelters)

    print(f"      Road nodes : {len(locations):,}")
    print(f"      Shelters   : {len(shelters):,}")
    total_edges = sum(len(v) for v in graph.values())
    print(f"      Graph edges: {total_edges:,}  (directed)")

    # 2. Route all locations
    print(f"\n[2/4] Running A* for all {len(locations):,} locations ...")
    results     = []
    unreachable = []

    for i, (loc_id, _) in enumerate(locations.items(), 1):
        if i % 200 == 0 or i == 1:
            print(f"      ... {i:4d} / {len(locations)}", end="\r", flush=True)

        result = astar(
            start       = loc_id,
            shelter_ids = shelter_ids,
            graph       = graph,
            coords      = coords,
            risk_scores = risk_scores,
            heuristic   = heuristic,
            use_risk    = True,
        )

        if result is None:
            unreachable.append(loc_id)
        else:
            results.append({
                "location_id"                    : loc_id,
                "nearest_shelter_id"             : result["nearest_shelter_id"],
                "path"                           : json.dumps(result["path"]),
                "total_cost"                     : result["total_cost"],
                "path_length_km"                 : result["path_length_km"],
                "num_bridged_edges_used"         : result["num_bridged_edges_used"],
                "num_shelter_connector_edges_used": result["num_shelter_connector_edges_used"],
            })

    print(f"\n      Done.  Routed: {len(results):,}  |  Unreachable: {len(unreachable):,}")

    # 3. Save routes.csv
    print("\n[3/4] Saving data/routes.csv ...")
    fieldnames = [
        "location_id",
        "nearest_shelter_id",
        "path",
        "total_cost",
        "path_length_km",
        "num_bridged_edges_used",
        "num_shelter_connector_edges_used",
    ]
    with open(ROUTES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"      Written: {ROUTES_CSV}")

    # 4. Summary stats
    print("\n[4/4] Summary Statistics")
    print("-" * 40)
    print(f"  Locations routed    : {len(results):,}")
    print(f"  Unreachable         : {len(unreachable):,}")
    if results:
        avg_cost    = sum(r["total_cost"]             for r in results) / len(results)
        avg_dist    = sum(r["path_length_km"]         for r in results) / len(results)
        avg_bridged = sum(r["num_bridged_edges_used"] for r in results) / len(results)
        print(f"  Avg total_cost      : {avg_cost:.4f}")
        print(f"  Avg path_length_km  : {avg_dist:.4f} km")
        print(f"  Avg bridged edges   : {avg_bridged:.3f}")

    if unreachable:
        print(f"\n  WARNING: {len(unreachable)} unreachable location(s):")
        print("  ", unreachable)

    # 5. Comparison demo
    flood_example = None
    with open(LOCATIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("historical_flood") == "1":
                flood_example = row["location_id"]
                break

    if flood_example is None:
        print("\n  (No historical_flood==1 location found for demo.)")
        return

    print("\n" + "=" * 60)
    print("  Comparison Demo — Risk-Aware vs. Naive Shortest Path")
    print(f"  Example location: {flood_example}")
    print("=" * 60)

    risk_result = astar(
        start=flood_example, shelter_ids=shelter_ids, graph=graph,
        coords=coords, risk_scores=risk_scores, heuristic=heuristic, use_risk=True,
    )
    naive_result = astar(
        start=flood_example, shelter_ids=shelter_ids, graph=graph,
        coords=coords, risk_scores=risk_scores, heuristic=heuristic, use_risk=False,
    )

    def _print_route(label, res):
        if res is None:
            print(f"\n  [{label}]  UNREACHABLE")
            return
        print(f"\n  [{label}]")
        print(f"    Shelter reached    : {res['nearest_shelter_id']}")
        print(f"    Total cost         : {res['total_cost']:.6f}")
        print(f"    Path length (km)   : {res['path_length_km']:.4f} km")
        print(f"    Nodes in path      : {len(res['path'])}")
        print(f"    Bridged edges used : {res['num_bridged_edges_used']}")
        print(f"    Shelter-conn edges : {res['num_shelter_connector_edges_used']}")
        print(f"    Full path          : {res['path']}")

    _print_route("a) Risk-Aware", risk_result)
    _print_route("b) Naive (distance only)", naive_result)

    print()
    if risk_result and naive_result:
        same_shelter = risk_result["nearest_shelter_id"] == naive_result["nearest_shelter_id"]
        same_path    = risk_result["path"] == naive_result["path"]
        if same_shelter and same_path:
            print("  VERDICT: SAME shelter AND same path — both approaches agree for this location.")
        elif same_shelter:
            print(
                f"  VERDICT: SAME shelter ({risk_result['nearest_shelter_id']}) "
                "but DIFFERENT paths — risk-aware routing chose a safer detour."
            )
        else:
            print(
                f"  VERDICT: DIVERGED — risk-aware -> {risk_result['nearest_shelter_id']} "
                f"| naive -> {naive_result['nearest_shelter_id']} "
                "— risk-aware routing redirected to a different shelter entirely."
            )
    else:
        print("  Could not compare (one or both routes unreachable).")

    print("\nRouting complete.")


if __name__ == "__main__":
    main()
