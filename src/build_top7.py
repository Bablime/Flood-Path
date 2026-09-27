"""
Stage 3 – Part A: Top-7 Domain Construction
============================================
For each of the 1,883 locations, runs ONE continued multi-goal A* search
that keeps expanding until 7 DISTINCT shelter nodes have been popped
(or the reachable component is exhausted). Records each shelter with its
rank, total_cost, and path_length_km.

Outputs
-------
data/routes_top7.csv  – location_id, rank (1-7), shelter_id,
                        total_cost, path_length_km, failure_reason

Verification
------------
For a random sample of 20 locations, confirms that rank-1 shelter_id
and total_cost match data/routes.csv exactly (same A* search, not
stopped early).

NOTE: All core functions (load_data, build_coords, build_heuristic,
      edge_cost, haversine) are imported directly from src/routing.py
      to avoid any code duplication.
"""

import csv
import heapq
import itertools
import os
import random
import sys

# ---------------------------------------------------------------------------
# Make src/ importable regardless of working directory
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Re-use ALL data-loading and cost functions from Stage 2 – no duplication.
from routing import (
    load_data,
    build_coords,
    build_heuristic,
    edge_cost,
    haversine,
    ROUTES_CSV,
)

DATA_DIR        = os.path.join(BASE_DIR, "data")
ROUTES_TOP7_CSV = os.path.join(DATA_DIR, "routes_top7.csv")

TOP_K = 7  # maximum number of distinct shelters to find per location


# ---------------------------------------------------------------------------
# Continued multi-goal A* – finds up to TOP_K distinct shelters
# ---------------------------------------------------------------------------
def astar_top_k(start, shelter_ids, graph, coords, risk_scores, heuristic, k=TOP_K):
    """
    Single A* search from `start` that continues expanding after the first
    shelter is popped, recording each NEWLY DISTINCT shelter in pop order.

    Returns a list of dicts, each with:
        shelter_id, total_cost, path_length_km, rank

    The heuristic h(n) = min haversine distance to any shelter is admissible
    and consistent, so nodes are popped in non-decreasing g-cost order once
    settled — the standard optimality proof applies across ALL goals, not
    just the first.  Recording shelters in pop order therefore gives the
    true rank-ordered cheapest paths.

    If the reachable component is exhausted before k shelters are found,
    returns however many were found (caller annotates the rest as
    failure_reason="structurally_unreachable").
    """
    if start not in coords:
        return []

    counter   = itertools.count()
    visited   = {}          # node_id -> best g finalised
    found     = []          # list of result dicts, in rank order
    found_ids = set()       # shelter_ids already recorded

    slat, slon = coords[start]
    h0 = heuristic(slat, slon)
    # heap: (f, tie_counter, node_id, path_list, g_score)
    open_set = []
    heapq.heappush(open_set, (h0, next(counter), start, [start], 0.0))

    while open_set and len(found) < k:
        f, _, current, path, g = heapq.heappop(open_set)

        # Skip stale entries (a better path to `current` was already settled)
        if current in visited and visited[current] <= g:
            continue
        visited[current] = g

        # ----- Goal test: newly discovered shelter -----
        if current in shelter_ids and current not in found_ids:
            found_ids.add(current)

            # Recompute raw path_length_km from the stored path
            raw_dist = 0.0
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
                raw_dist += best_edge[0]

            found.append({
                "shelter_id"     : current,
                "total_cost"     : g,
                "path_length_km" : raw_dist,
                "rank"           : len(found) + 1,
            })

            # Do NOT return — keep expanding to find more shelters.

        # ----- Expand neighbours (always, even for shelter nodes) -----
        for neighbour, dist_km, etype in graph[current]:
            step  = edge_cost(current, neighbour, dist_km, etype, risk_scores)
            new_g = g + step

            if neighbour in visited and visited[neighbour] <= new_g:
                continue

            if neighbour in coords:
                nlat, nlon = coords[neighbour]
                h_val = heuristic(nlat, nlon)
            else:
                h_val = 0.0

            heapq.heappush(
                open_set,
                (new_g + h_val, next(counter), neighbour, path + [neighbour], new_g),
            )

    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Stage 3 – Part A: Top-7 Domain Construction")
    print("=" * 65)

    # 1. Load data (reusing Stage 2's load_data)
    print("\n[1/4] Loading data ...")
    risk_scores, shelters, locations, graph = load_data()
    shelter_ids = set(shelters.keys())
    coords      = build_coords(locations, shelters)
    heuristic   = build_heuristic(shelters)

    print(f"      Road nodes : {len(locations):,}")
    print(f"      Shelters   : {len(shelters):,}")
    total_edges = sum(len(v) for v in graph.values())
    print(f"      Graph edges: {total_edges:,}  (directed)")

    # 2. Run continued multi-goal A* for every location
    print(f"\n[2/4] Running top-{TOP_K} A* for all {len(locations):,} locations ...")
    all_rows          = []
    stats_found       = {r: 0 for r in range(1, TOP_K + 1)}
    fully_unreachable = 0

    for i, (loc_id, _) in enumerate(locations.items(), 1):
        if i % 100 == 0 or i == 1:
            print(f"      ... {i:4d} / {len(locations)}", end="\r", flush=True)

        found = astar_top_k(
            start       = loc_id,
            shelter_ids = shelter_ids,
            graph       = graph,
            coords      = coords,
            risk_scores = risk_scores,
            heuristic   = heuristic,
            k           = TOP_K,
        )

        if not found:
            fully_unreachable += 1
            for rank in range(1, TOP_K + 1):
                all_rows.append({
                    "location_id"   : loc_id,
                    "rank"          : rank,
                    "shelter_id"    : "",
                    "total_cost"    : "",
                    "path_length_km": "",
                    "failure_reason": "structurally_unreachable",
                })
        else:
            for entry in found:
                all_rows.append({
                    "location_id"   : loc_id,
                    "rank"          : entry["rank"],
                    "shelter_id"    : entry["shelter_id"],
                    "total_cost"    : entry["total_cost"],
                    "path_length_km": entry["path_length_km"],
                    "failure_reason": "",
                })
                stats_found[entry["rank"]] = stats_found.get(entry["rank"], 0) + 1

            # Fill in missing higher ranks as structurally_unreachable
            found_count = len(found)
            for rank in range(found_count + 1, TOP_K + 1):
                all_rows.append({
                    "location_id"   : loc_id,
                    "rank"          : rank,
                    "shelter_id"    : "",
                    "total_cost"    : "",
                    "path_length_km": "",
                    "failure_reason": "structurally_unreachable",
                })

    print(f"\n      Done.  Rows written: {len(all_rows):,}")
    print(f"      Fully unreachable locations (0 shelters found): {fully_unreachable}")

    # 3. Save routes_top7.csv
    print("\n[3/4] Saving data/routes_top7.csv ...")
    fieldnames = [
        "location_id", "rank", "shelter_id",
        "total_cost", "path_length_km", "failure_reason",
    ]
    with open(ROUTES_TOP7_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"      Written: {ROUTES_TOP7_CSV}")

    # 4. Verification – compare rank-1 results against routes.csv
    print("\n[4/4] Verification: rank-1 vs routes.csv (random sample of 20) ...")

    routes_ref = {}
    with open(ROUTES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            routes_ref[row["location_id"]] = {
                "shelter_id": row["nearest_shelter_id"],
                "total_cost": float(row["total_cost"]),
            }

    rank1_new = {}
    for row in all_rows:
        if int(row["rank"]) == 1 and row["shelter_id"]:
            rank1_new[row["location_id"]] = {
                "shelter_id": row["shelter_id"],
                "total_cost": float(row["total_cost"]),
            }

    common_ids = [lid for lid in routes_ref if lid in rank1_new]
    random.seed(42)
    sample = random.sample(common_ids, min(20, len(common_ids)))

    mismatches = 0
    print(f"\n  {'Location':<12} {'Ref shelter':<12} {'New shelter':<12} "
          f"{'Ref cost':>12} {'New cost':>12} {'Match?'}")
    print("  " + "-" * 70)
    for lid in sorted(sample):
        ref = routes_ref[lid]
        new = rank1_new[lid]
        shelter_match = ref["shelter_id"] == new["shelter_id"]
        cost_match    = abs(ref["total_cost"] - new["total_cost"]) < 1e-6
        ok = shelter_match and cost_match
        if not ok:
            mismatches += 1
        marker = "OK" if ok else "MISMATCH"
        print(f"  {lid:<12} {ref['shelter_id']:<12} {new['shelter_id']:<12} "
              f"{ref['total_cost']:12.6f} {new['total_cost']:12.6f}  {marker}")

    print()
    if mismatches == 0:
        print(f"  VERIFICATION: PASS -- 0 mismatches in {len(sample)} sampled locations.")
    else:
        print(f"  VERIFICATION: FAIL -- {mismatches} mismatches in {len(sample)} sampled locations.")

    print("\n  Rank distribution (how many locations have each rank filled):")
    for rank in range(1, TOP_K + 1):
        n = stats_found.get(rank, 0)
        print(f"    Rank {rank}: {n:,} locations ({100*n/len(locations):.1f}%)")

    print("\nPart A complete.\n")
    return ROUTES_TOP7_CSV


if __name__ == "__main__":
    main()
