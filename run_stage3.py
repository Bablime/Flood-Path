"""
Stage 3 – Combined runner: Part A then Part B
"""
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print("=" * 65)
print("STAGE 3 – RUNNING PART A (Top-7 domain construction) ...")
print("=" * 65)
import build_top7
build_top7.main()

print()
print("=" * 65)
print("STAGE 3 – RUNNING PART B (CSP Allocation) ...")
print("=" * 65)
import allocation
allocation.main()
