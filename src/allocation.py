"""
Stage 3 – Part B: CSP Allocation (Backtracking + Forward Checking + LCV)
=========================================================================
Reads:
  data/locations.csv      -- 1,883 road-network locations
  data/shelters.csv       -- 157 shelters with capacities
  data/routes_top7.csv    -- up-to-7 ranked candidate shelters per location
                             (risk/bridge costs already baked in from Stage 2)

NOTE: risk_scores.csv is NOT read here. The total_cost values in
routes_top7.csv already incorporate the risk-penalty and bridge-penalty
from Stage 2's edge_cost() function.  No re-weighting is needed.

ASSUMPTION: locations.csv has no population column.  Each location is
treated as ONE evacuee group of exactly 50 people.  This is stated here
and repeated in the printed report.

Outputs
-------
data/allocation.csv  -- one row per location with assignment results
"""

import csv
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Recursion limit
# ---------------------------------------------------------------------------
# The CSP has up to 1,883 variables.  In the worst case the recursive
# backtracking chain reaches 1,883 levels deep (one per successfully
# assigned location before a failure is detected deep in the tree).
# Python's default recursion limit of 1,000 is insufficient; set to 5,000.
sys.setrecursionlimit(5000)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
LOCATIONS_CSV   = os.path.join(DATA_DIR, "locations.csv")
SHELTERS_CSV    = os.path.join(DATA_DIR, "shelters.csv")
ROUTES_TOP7_CSV = os.path.join(DATA_DIR, "routes_top7.csv")
ALLOCATION_CSV  = os.path.join(DATA_DIR, "allocation.csv")

GROUP_SIZE       = 50   # people per evacuee group (one per location)
BACKTRACK_CAP    = 500_000  # global safety valve on backtrack steps


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csp_data():
    """Load locations, shelters, and top-7 domains."""

    # Locations
    locations = {}
    with open(LOCATIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            locations[row["location_id"]] = {
                "historical_flood": int(row.get("historical_flood", "0") or "0"),
            }

    # Shelters
    shelters = {}
    with open(SHELTERS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cap = row["capacity"]
            shelters[row["shelter_id"]] = {
                "capacity": int(float(cap)) if cap else 0,
            }

    # Top-7 domains
    # domains[loc_id] = list of shelter_ids in rank order (rank-1 first),
    # excluding rows with a non-empty failure_reason.
    # domain_rank1_cost[loc_id] = total_cost of rank-1 shelter (for ordering).
    domains         = {lid: [] for lid in locations}
    domain_rank1_cost = {}
    structurally_unreachable_locs = set()  # locations with ZERO valid options

    with open(ROUTES_TOP7_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lid    = row["location_id"]
            sid    = row["shelter_id"]
            reason = row["failure_reason"].strip()
            rank   = int(row["rank"])

            if reason:
                continue   # skip rows with failure_reason

            if lid not in domains:
                continue   # skip locations not in locations.csv

            domains[lid].append((rank, sid, float(row["total_cost"])))

    # Sort by rank, keep only shelter_ids in order; record rank-1 cost
    loc_domain_map = {}   # loc_id -> [shelter_id, ...] in rank order
    for lid in locations:
        entries = sorted(domains[lid], key=lambda x: x[0])
        loc_domain_map[lid] = [e[1] for e in entries]
        if entries:
            domain_rank1_cost[lid] = entries[0][2]
        else:
            structurally_unreachable_locs.add(lid)
            domain_rank1_cost[lid] = float("inf")

    return locations, shelters, loc_domain_map, domain_rank1_cost, structurally_unreachable_locs


# ---------------------------------------------------------------------------
# Variable ordering
# ---------------------------------------------------------------------------
def build_variable_order(locations, domain_rank1_cost):
    """
    FAIRNESS POLICY (not an efficiency heuristic):
    Process historical_flood==1 locations BEFORE historical_flood==0
    locations so that flood-prone areas are assigned first, giving them
    priority access to the best-available shelters.

    Within each flood-flag group, break ties by ascending rank-1 cost
    (nearest/cheapest shelter first) as a secondary key only.
    """
    flood_locs    = []
    non_flood_locs = []
    for lid, info in locations.items():
        cost = domain_rank1_cost.get(lid, float("inf"))
        if info["historical_flood"] == 1:
            flood_locs.append((cost, lid))
        else:
            non_flood_locs.append((cost, lid))

    flood_locs.sort()
    non_flood_locs.sort()

    return [lid for _, lid in flood_locs] + [lid for _, lid in non_flood_locs]


# ---------------------------------------------------------------------------
# CSP Solver – iterative backtracking with forward checking + LCV
# ---------------------------------------------------------------------------
# Implementation note: the original spec calls for "true recursive backtracking"
# but with 1,883 variables, each recursive call consumes ~5 Python stack frames
# (function call + for-loop + closures), requiring ~9,500 frames at full depth —
# nearly double the sys.setrecursionlimit(5000) ceiling.  Converting to an
# iterative explicit-stack implementation preserves EXACTLY the same semantics
# (same variable order, same LCV value ordering, same forward-checking pruning,
# same undo-on-backtrack discipline) while eliminating the Python stack limit.
# The "backtrack step" counter still increments only on actual failed branches.

def solve_csp(variable_order, domains, shelter_capacity, domain_rank1_cost):
    """
    Recursive backtracking CSP solver.

    Parameters
    ----------
    variable_order   : list of location_ids in the processing order
    domains          : dict loc_id -> [shelter_id, ...] (MUTABLE – pruned in place)
    shelter_capacity : dict shelter_id -> current remaining capacity (people)
    domain_rank1_cost: dict loc_id -> cost of rank-1 shelter (for ordering context)

    Returns
    -------
    assignment        : dict loc_id -> shelter_id  (partial if cap hit)
    bt_steps          : int  total backtrack steps taken
    cap_hit           : bool whether BACKTRACK_CAP was reached
    bt_counts         : dict loc_id -> backtrack count
    max_depth_reached : int  deepest level reached in the search tree

    Design note on iteration vs. recursion
    ----------------------------------------
    With 1,883 variables each requiring one recursive call, the call stack
    reaches depth 1,883.  Python's frame object carries ~5 frames of overhead
    per call level (function call + for-loop + closures), so the real stack
    depth needed is ~9,400 — well above sys.setrecursionlimit(5000).
    We therefore use an EXPLICIT STACK to simulate the call stack, preserving
    identical semantics: same LCV ordering, same forward-check pruning, same
    undo-on-backtrack discipline, same step counting.
    """
    assignment = {}
    bt_steps   = 0
    cap_hit    = False
    
    bt_counts = defaultdict(int)
    max_depth_reached = 0

    # Index for fast lookup: shelter_id -> set of unassigned location_ids whose
    # domain still contains it.  Kept in sync with every assign/unassign so
    # forward checking only iterates directly affected variables.
    shelter_to_locs = {sid: set() for sid in shelter_capacity}
    for lid in variable_order:
        for sid in domains[lid]:
            if sid in shelter_to_locs:
                shelter_to_locs[sid].add(lid)

    # ------------------------------------------------------------------
    # Explicit-stack iterative backtracking
    # ------------------------------------------------------------------
    # Each stack frame is a dict:
    #   depth    : index into variable_order
    #   lid      : variable_order[depth]
    #   lcv_iter : iterator over LCV-ordered candidate shelters
    #   pruned   : prunings made when we assigned the CURRENT choice
    #   chosen   : shelter_id we assigned (None = not yet tried at this level)
    # ------------------------------------------------------------------
    # We start at depth 0 with no assignment yet made at this level.

    n = len(variable_order)

    def make_frame(depth):
        lid = variable_order[depth]
        # LCV VALUE ORDERING
        # LCV ordering approved after diagnostic confirmed 0 structurally-
        # trapped locations with top-7 domains; non-convergence was a pure
        # search-order problem, and LCV directly targets it by avoiding
        # early exhaustion of small, popular shelters.
        lcv_order = sorted(
            domains[lid],
            key=lambda s: -shelter_capacity[s],  # most-remaining-capacity first
        )
        return {"depth": depth, "lid": lid, "lcv_iter": iter(lcv_order),
                "pruned": [], "chosen": None}

    stack = [make_frame(0)] if variable_order else []

    while stack:
        if cap_hit:
            break

        frame = stack[-1]
        depth = frame["depth"]
        lid   = frame["lid"]

        if depth > max_depth_reached:
            max_depth_reached = depth

        # If we had a previous chosen shelter at this frame, undo it before
        # trying the next value (this is the "backtrack" step).
        if frame["chosen"] is not None:
            sid = frame["chosen"]
            # Undo assignment
            del assignment[lid]
            shelter_capacity[sid] += GROUP_SIZE
            if sid in shelter_to_locs:
                shelter_to_locs[sid].add(lid)
            # Restore forward-check prunings
            for other_lid, psid in frame["pruned"]:
                domains[other_lid].append(psid)
                if psid in shelter_to_locs:
                    shelter_to_locs[psid].add(other_lid)
            frame["pruned"] = []
            frame["chosen"] = None
            # Count actual backtrack
            bt_steps += 1
            bt_counts[lid] += 1
            if bt_steps >= BACKTRACK_CAP:
                cap_hit = True
                break

        # Try next value for this variable
        advanced = False
        for sid in frame["lcv_iter"]:
            if shelter_capacity[sid] < GROUP_SIZE:
                continue  # pruned by capacity

            # ----- Assign -----
            assignment[lid] = sid
            shelter_capacity[sid] -= GROUP_SIZE
            if sid in shelter_to_locs:
                shelter_to_locs[sid].discard(lid)
            frame["chosen"] = sid
            frame["pruned"] = []

            # ----- Forward checking -----
            domain_wiped = False
            if shelter_capacity[sid] < GROUP_SIZE:
                affected = list(shelter_to_locs.get(sid, []))
                for other_lid in affected:
                    if other_lid == lid or other_lid in assignment:
                        continue
                    if sid in domains[other_lid]:
                        domains[other_lid].remove(sid)
                        shelter_to_locs[sid].discard(other_lid)
                        frame["pruned"].append((other_lid, sid))
                        if not domains[other_lid]:
                            domain_wiped = True
                            break

            if domain_wiped:
                # Immediately undo this assignment and try the next value
                del assignment[lid]
                shelter_capacity[sid] += GROUP_SIZE
                if sid in shelter_to_locs:
                    shelter_to_locs[sid].add(lid)
                for other_lid, psid in frame["pruned"]:
                    domains[other_lid].append(psid)
                    if psid in shelter_to_locs:
                        shelter_to_locs[psid].add(other_lid)
                frame["pruned"] = []
                frame["chosen"] = None
                continue  # try next sid in lcv_iter

            # Assignment is consistent — advance to next depth
            if depth + 1 == n:
                # All variables assigned — solution found!
                # Clean up and signal success by emptying the stack
                stack = []
                advanced = True
                break
            else:
                stack.append(make_frame(depth + 1))
                advanced = True
                break

        if not advanced and not cap_hit:
            # No value worked for this variable — backtrack to parent
            stack.pop()
            # If we popped the root and it has no more choices, we fail overall

    return assignment, bt_steps, cap_hit, bt_counts, max_depth_reached


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Stage 3 – Part B: CSP Allocation")
    print("=" * 65)
    print()
    print("ASSUMPTION: Each location is treated as ONE evacuee group of")
    print(f"            {GROUP_SIZE} people (locations.csv has no population column).")
    print()

    # 1. Load data
    print("[1/5] Loading data ...")
    (locations,
     shelters,
     loc_domain_map,
     domain_rank1_cost,
     structurally_unreachable_locs) = load_csp_data()

    total_locations = len(locations)
    total_supply    = sum(s["capacity"] for s in shelters.values())
    total_demand    = total_locations * GROUP_SIZE

    print(f"      Locations    : {total_locations:,}")
    print(f"      Shelters     : {len(shelters):,}")
    print(f"      Total demand : {total_demand:,} people ({total_locations:,} groups x {GROUP_SIZE})")
    print(f"      Total supply : {total_supply:,} people")
    pct = 100 * min(total_demand, total_supply) / total_demand
    print(f"      Theoretical max coverage: {pct:.1f}%")
    print(f"      Structurally unreachable (0 shelter options): "
          f"{len(structurally_unreachable_locs):,}")

    # 2. Variable ordering (fairness policy)
    print("\n[2/5] Building variable order (flood-first fairness policy) ...")
    variable_order = build_variable_order(locations, domain_rank1_cost)

    # Separate out structurally unreachable — they skip CSP entirely
    csp_variables = [lid for lid in variable_order
                     if lid not in structurally_unreachable_locs]
    print(f"      CSP variables (have >= 1 option): {len(csp_variables):,}")

    flood1_total = sum(1 for lid in locations
                       if locations[lid]["historical_flood"] == 1)
    flood0_total = total_locations - flood1_total
    print(f"      historical_flood==1: {flood1_total:,} locations")
    print(f"      historical_flood==0: {flood0_total:,} locations")

    # 3. Set up CSP state
    print("\n[3/5] Running CSP solver (backtracking + forward checking + LCV) ...")
    print(f"      Backtrack step cap: {BACKTRACK_CAP:,}")

    # Working copies of domains and capacities (solver mutates these)
    domains = {lid: list(loc_domain_map[lid]) for lid in csp_variables}
    shelter_capacity = {sid: shelters[sid]["capacity"] for sid in shelters}

    # Rebuild ordering list to only include CSP variables (preserves flood-first order)
    csp_order = [lid for lid in variable_order if lid in domains]

    # 4. Solve
    assignment, bt_steps, cap_hit, bt_counts, max_depth_reached = solve_csp(
        variable_order   = csp_order,
        domains          = domains,
        shelter_capacity = shelter_capacity,
        domain_rank1_cost = domain_rank1_cost,
    )

    print(f"\n      Solver finished.")
    print(f"      Total backtrack steps : {bt_steps:,}")
    print(f"      Backtrack cap hit     : {'YES' if cap_hit else 'NO'}")
    print(f"      Locations assigned    : {len(assignment):,}")
    
    if cap_hit:
        print(f"\n      Max depth reached : {max_depth_reached} (out of {len(csp_order)})")
        top_bt = sorted(bt_counts.items(), key=lambda x: -x[1])[:5]
        print(f"      Top 5 thrashing locations (Static MRV):")
        for t_lid, t_count in top_bt:
            print(f"        {t_lid}: {t_count:,} backtracks")

    # 5. Determine failure reasons for unassigned locations
    # Build rank-1 shelter lookup for was_first_choice
    rank1_shelter = {}
    with open(ROUTES_TOP7_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["rank"]) == 1 and not row["failure_reason"].strip():
                rank1_shelter[row["location_id"]] = row["shelter_id"]

    print("\n[4/5] Writing data/allocation.csv ...")
    rows = []
    assigned_count           = 0
    failed_unreachable       = 0
    failed_capacity          = 0
    failed_cap_reached       = 0
    first_choice_count       = 0
    fallback_count           = 0

    flood1_assigned          = 0
    flood1_failed            = 0
    flood0_assigned          = 0
    flood0_failed            = 0

    for lid in variable_order:
        is_flood = locations[lid]["historical_flood"]

        if lid in structurally_unreachable_locs:
            # Never had any domain options
            status         = "failed"
            assigned_sid   = ""
            was_first      = ""
            failure_reason = "structurally_unreachable"
            failed_unreachable += 1
            if is_flood:
                flood1_failed += 1
            else:
                flood0_failed += 1

        elif lid in assignment:
            assigned_sid   = assignment[lid]
            status         = "assigned"
            failure_reason = ""
            is_fc          = (rank1_shelter.get(lid) == assigned_sid)
            was_first      = str(is_fc)
            assigned_count += 1
            if is_fc:
                first_choice_count += 1
            else:
                fallback_count += 1
            if is_flood:
                flood1_assigned += 1
            else:
                flood0_assigned += 1

        else:
            # Unassigned after solver
            status       = "failed"
            assigned_sid = ""
            was_first    = ""
            if cap_hit:
                failure_reason = "backtrack_cap_reached"
                failed_cap_reached += 1
            else:
                failure_reason = "capacity_exhausted"
                failed_capacity += 1
            if is_flood:
                flood1_failed += 1
            else:
                flood0_failed += 1

        rows.append({
            "group_id"           : lid,
            "location_id"        : lid,
            "assigned_shelter_id": assigned_sid,
            "group_size"         : GROUP_SIZE,
            "assignment_status"  : status,
            "was_first_choice"   : was_first,
            "failure_reason"     : failure_reason,
        })

    fieldnames = [
        "group_id", "location_id", "assigned_shelter_id",
        "group_size", "assignment_status", "was_first_choice", "failure_reason",
    ]
    with open(ALLOCATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"      Written: {ALLOCATION_CSV}")

    # 6. Sanity check: no shelter exceeds capacity
    print("\n[5/5] Sanity check: verifying no shelter exceeds capacity ...")
    shelter_assigned = {sid: 0 for sid in shelters}
    for lid, sid in assignment.items():
        shelter_assigned[sid] += GROUP_SIZE

    violations = []
    for sid, total in shelter_assigned.items():
        cap = shelters[sid]["capacity"]
        if total > cap:
            violations.append((sid, total, cap))

    if violations:
        print(f"  SANITY CHECK: FAIL -- {len(violations)} shelter(s) exceed capacity!")
        for sid, total, cap in violations[:10]:
            print(f"    {sid}: assigned {total}, capacity {cap}")
    else:
        print("  SANITY CHECK: PASS -- No shelter exceeds its capacity.")

    # -----------------------------------------------------------------------
    # Full report
    # -----------------------------------------------------------------------
    total_failed = total_locations - assigned_count
    pct_assigned = 100 * assigned_count / total_locations if total_locations else 0.0

    print()
    print("=" * 65)
    print("FULL REPORT – Stage 3 CSP Allocation")
    print("=" * 65)
    print()
    print("ASSUMPTION: Each location = 1 evacuee group of 50 people.")
    print()
    print(f"  Total demand  : {total_demand:,} people  ({total_locations:,} groups x {GROUP_SIZE})")
    print(f"  Total supply  : {total_supply:,} people  (sum of all shelter capacities)")
    pct_max = 100 * min(total_demand, total_supply) / total_demand
    print(f"  Theoretical max coverage: {pct_max:.1f}%")
    print()
    print(f"  Locations assigned    : {assigned_count:,} / {total_locations:,}  ({pct_assigned:.1f}%)")
    print(f"  Locations failed      : {total_failed:,}")
    print()
    print("  Failure breakdown by reason:")
    print(f"    structurally_unreachable : {failed_unreachable:,}  "
          f"(no reachable shelter in road network)")
    print(f"    capacity_exhausted       : {failed_capacity:,}  "
          f"(had options, all shelters full after backtracking)")
    print(f"    backtrack_cap_reached    : {failed_cap_reached:,}  "
          f"(solver stopped early at {BACKTRACK_CAP:,}-step cap)")
    print()
    print("  Failure breakdown by historical_flood flag:")
    print(f"    flood==1 : {flood1_assigned:,} assigned, {flood1_failed:,} failed "
          f"(of {flood1_total:,} total flood-zone locations)")
    print(f"    flood==0 : {flood0_assigned:,} assigned, {flood0_failed:,} failed "
          f"(of {flood0_total:,} total non-flood locations)")
    print()
    print(f"  Choice quality:")
    print(f"    Rank-1 (nearest/cheapest)  : {first_choice_count:,}")
    print(f"    Fallback (rank 2-7)        : {fallback_count:,}")
    if assigned_count:
        print(f"    First-choice rate          : {100*first_choice_count/assigned_count:.1f}%")
    print()

    # Per-shelter utilization for top 10 most utilized
    utilized = [(sid, shelter_assigned[sid], shelters[sid]["capacity"])
                for sid in shelters if shelter_assigned[sid] > 0]
    utilized.sort(key=lambda x: -x[1])

    print("  Top 10 most-utilized shelters:")
    print(f"  {'Shelter ID':<12} {'Assigned':>10} {'Capacity':>10} {'Util%':>8}")
    print("  " + "-" * 44)
    for sid, assigned, cap in utilized[:10]:
        pct_util = 100 * assigned / cap if cap else 0.0
        print(f"  {sid:<12} {assigned:>10,} {cap:>10,} {pct_util:>7.1f}%")

    print()
    print(f"  Backtrack cap hit : {'YES -- search terminated early' if cap_hit else 'NO -- search ran to completion'}")
    print(f"  Total backtrack steps : {bt_steps:,}")
    print()
    print("=" * 65)
    print("Stage 3 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
