# Stage 3 CSP Bottleneck Diagnostic Notes

## Summary of Findings

1. **The Bottleneck**
   The static-order CSP solver hit the 500,000-step backtrack cap reaching a maximum of 65.0% coverage (1,224 locations assigned). Through detailed telemetry, we traced the deepest thrashing point to location **N1226** at depth 1224. 

2. **Diagnosis: Algorithmic Pathology**
   We proved that this thrashing is a pathology of naive **chronological backtracking**, *not* a signal of genuine geographic shelter scarcity. A cross-check of N1226's domain against its immediate predecessors in the search tree (N1861, N1242) revealed that they share **zero** overlapping shelters and are geographically scattered. Thus, when N1226 failed due to capacity exhausted by earlier locations, the solver futilely iterated through millions of combinations of N1861 and N1242 instead of backjumping to the actual constraints. 

3. **Dynamic MRV Testing**
   To resolve the thrashing without violating our strict `historical_flood` fairness constraint, we implemented and tested a Dynamic Minimum Remaining Values (MRV) heuristic. However, MRV performed *worse* than the static baseline (1,037 assigned, 55.1% coverage), hitting a new bottleneck earlier in the tree (N69, depth 1043). This proved the limitation is structural to chronological backtracking itself—MRV merely surfaced the exact same pathology earlier in the tree. 

4. **Why CBJ Was Deferred**
   The theoretically correct fix for this structural pathology is **Conflict-Directed Backjumping (CBJ)**, which tracks conflict sets and allows the solver to jump directly back to the variables responsible for a dead end. However, implementing CBJ is complex and introduces a high risk of correctness bugs. Given project timeline constraints, CBJ implementation is out of scope for this stage.

5. **Final Conclusion & Stage 4 Context**
   The static-order result of **1,224 assigned (65.0%)** is accepted as final for Stage 3. It is explicitly understood that the remaining 659 unassigned locations failed due to an **algorithmic constraint**, not a physical capacity or geographical constraint (total supply exceeds demand by ~16%, and all locations have structurally reachable options). This distinction is critical for Stage 4: the "failed" locations from Stage 3 should *not* be treated as genuine signals of physical shelter scarcity when designing the Hill Climbing new-shelter-placement logic.
