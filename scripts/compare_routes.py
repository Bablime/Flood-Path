"""
scripts/compare_routes.py
=========================
Diagnostic script that finds the most significant divergences between
risk-aware A* and naive (distance-only) A* routing.

Imports all logic from src/routing.py — no duplication.
Read-only: does NOT write any files.
"""

import csv
import os
import sys

# ── Make src/ importable ────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from routing import (
    load_data,
    build_coords,
    build_heuristic,
    astar,
    LOCATIONS_CSV,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_route_block(label: str, res: dict | None) -> None:
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


def divergence_key(item: dict) -> tuple:
    """
    Sort key — descending priority:
      1. shelter differs  (True > False → negate for ascending sort)
      2. abs cost difference  (larger = more significant → negate)
      3. abs km difference    (larger = more significant → negate)
    """
    return (
        0 if item["shelter_differs"] else 1,           # shelter differs first
        -item["cost_diff"],
        -item["km_diff"],
    )


def run_search(locations_subset: list, label: str, risk_scores, shelters,
               locations, graph) -> tuple[list, int, int]:
    """
    Run both A* variants for every location in `locations_subset`.
    Returns (divergent_cases, n_diverged, n_identical).
    """
    shelter_ids = set(shelters.keys())
    coords      = build_coords(locations, shelters)
    heuristic   = build_heuristic(shelters)

    divergent   = []
    n_identical = 0
    n_diverged  = 0

    total = len(locations_subset)
    for i, (loc_id, _) in enumerate(locations_subset, 1):
        if i % 200 == 0 or i == 1:
            print(f"    ... {i:4d} / {total}", end="\r", flush=True)

        risk_res  = astar(loc_id, shelter_ids, graph, coords,
                          risk_scores, heuristic, use_risk=True)
        naive_res = astar(loc_id, shelter_ids, graph, coords,
                          risk_scores, heuristic, use_risk=False)

        if risk_res is None or naive_res is None:
            # Can't compare if either is unreachable
            n_identical += 1
            continue

        same_shelter = risk_res["nearest_shelter_id"] == naive_res["nearest_shelter_id"]
        same_path    = risk_res["path"] == naive_res["path"]

        if same_shelter and same_path:
            n_identical += 1
        else:
            n_diverged += 1
            divergent.append({
                "location_id"     : loc_id,
                "shelter_differs" : not same_shelter,
                "path_differs"    : not same_path,
                "cost_diff"       : abs(risk_res["total_cost"] - naive_res["total_cost"]),
                "km_diff"         : abs(risk_res["path_length_km"] - naive_res["path_length_km"]),
                "risk_result"     : risk_res,
                "naive_result"    : naive_res,
            })

    print()  # clear the progress line
    return divergent, n_diverged, n_identical


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 62)
    print("  Route Divergence Analysis — Risk-Aware vs. Naive A*")
    print("=" * 62)

    # Load shared data once
    print("\n[1/3] Loading data ...")
    risk_scores, shelters, locations, graph = load_data()

    # Partition locations by historical_flood flag
    flood_locs    = [(lid, info) for lid, info in locations.items()
                     if info["historical_flood"] == "1"]
    no_flood_locs = [(lid, info) for lid, info in locations.items()
                     if info["historical_flood"] == "0"]

    print(f"      historical_flood==1 : {len(flood_locs):,} locations")
    print(f"      historical_flood==0 : {len(no_flood_locs):,} locations")

    # ── Primary search: flood-prone locations ──────────────────────────────
    print(f"\n[2/3] Comparing routes for {len(flood_locs):,} "
          "historical_flood==1 locations ...")
    divergent, n_div, n_same = run_search(
        flood_locs, "flood", risk_scores, shelters, locations, graph
    )

    # ── Stats ──────────────────────────────────────────────────────────────
    total_checked = n_div + n_same
    pct = 100.0 * n_div / total_checked if total_checked else 0.0
    print(f"\n[3/3] Divergence Summary (historical_flood==1)")
    print("-" * 50)
    print(f"  Locations checked           : {total_checked:,}")
    print(f"  Identical routes (both same): {n_same:,}")
    print(f"  Divergent routes            : {n_div:,}  ({pct:.1f}%)")

    # ── Fallback if no divergence found ───────────────────────────────────
    if not divergent:
        print("\n  No divergence found among historical_flood==1 locations.")
        print("  NOTE: Flood-prone areas may have fewer alternate roads —")
        print("  their road network topology often offers only one viable")
        print("  corridor to a shelter, so risk-aware re-weighting cannot")
        print("  redirect the path even when edge costs rise significantly.")
        print()
        print("  Falling back to historical_flood==0 locations ...")
        divergent, n_div_0, n_same_0 = run_search(
            no_flood_locs, "non-flood", risk_scores, shelters, locations, graph
        )
        pct0 = 100.0 * n_div_0 / (n_div_0 + n_same_0) if (n_div_0 + n_same_0) else 0.0
        print(f"  Divergent in flood==0 set   : {n_div_0:,}  ({pct0:.1f}%)")

    if not divergent:
        print("\n  No divergence found in any location. "
              "The graph may have insufficient alternate paths.")
        return

    # ── Rank and display top 3 ─────────────────────────────────────────────
    ranked = sorted(divergent, key=divergence_key)
    top3   = ranked[:3]

    print()
    print("=" * 62)
    print("  TOP DIVERGENT EXAMPLES (ranked by significance)")
    print("=" * 62)

    for rank, case in enumerate(top3, 1):
        loc_id  = case["location_id"]
        rr      = case["risk_result"]
        nr      = case["naive_result"]

        divergence_type = []
        if case["shelter_differs"]:
            divergence_type.append("DIFFERENT SHELTER")
        if case["path_differs"] and not case["shelter_differs"]:
            divergence_type.append("same shelter, different path")
        divergence_str = " + ".join(divergence_type) if divergence_type else "path differs"

        print(f"\n{'─'*62}")
        print(f"  Rank #{rank}  |  Location: {loc_id}  |  {divergence_str}")
        print(f"  Cost diff : {case['cost_diff']:.4f}  |  "
              f"km diff : {case['km_diff']:.4f} km")
        print(f"{'─'*62}")

        print_route_block("a) Risk-Aware", rr)
        print_route_block("b) Naive (distance only)", nr)

        print()
        if case["shelter_differs"]:
            print(f"  VERDICT: DIVERGED — risk-aware -> {rr['nearest_shelter_id']} "
                  f"| naive -> {nr['nearest_shelter_id']}"
                  "\n  Risk-aware routing redirected to a DIFFERENT shelter entirely.")
        else:
            print(f"  VERDICT: Same shelter ({rr['nearest_shelter_id']}) "
                  "but DIFFERENT path taken.")
            risk_b  = rr["num_bridged_edges_used"]
            naive_b = nr["num_bridged_edges_used"]
            if risk_b < naive_b:
                print(f"  Risk-aware avoided {naive_b - risk_b} bridged edge(s) "
                      "that the naive route crossed.")
            elif risk_b > naive_b:
                print(f"  Risk-aware used {risk_b - naive_b} more bridged edge(s) "
                      "(lower-risk area offset the bridge penalty).")

    print()
    print("=" * 62)
    print("  BEST EXAMPLE FOR REPORT/VIVA:")
    best = top3[0]
    print(f"  Location: {best['location_id']}")
    if best["shelter_differs"]:
        print(f"  Risk-aware -> {best['risk_result']['nearest_shelter_id']}  "
              f"| Naive -> {best['naive_result']['nearest_shelter_id']}")
        print("  (Different shelter — most dramatic divergence possible)")
    else:
        print(f"  Shelter: {best['risk_result']['nearest_shelter_id']} (same)")
        print(f"  Cost difference: {best['cost_diff']:.4f}")
        print(f"  Path differs: {best['path_differs']}")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
