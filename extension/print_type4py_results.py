#!/usr/bin/env python3
"""
Print evaluation tables for Type4Py + TypeCare from output/type4py_result.pkl.

Usage:
    python extension/print_type4py_results.py
"""

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
RESULT_PKL = REPO_ROOT / "output" / "type4py_result.pkl"

if not RESULT_PKL.exists():
    print(f"Error: Result file not found at {RESULT_PKL}")
    print("Please make sure TypeCare analysis has been run.")
    sys.exit(1)

# Ensure dependencies are available or provide mock fallbacks for standalone execution
try:
    import tabulate
except ImportError:
    tabulate_mod = types.ModuleType("tabulate")
    def _tabulate(rows, headers=None, **kwargs):
        headers = headers or []
        widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]
        header_str = " | ".join(f"{str(h):<{widths[i]}}" for i, h in enumerate(headers))
        sep_str = "-+-".join("-" * widths[i] for i in range(len(headers)))
        body_lines = [" | ".join(f"{str(r[i]):<{widths[i]}}" for i in range(len(headers))) for r in rows]
        return "\n".join([header_str, sep_str] + body_lines)
    tabulate_mod.tabulate = _tabulate
    tabulate_mod.SEPARATING_LINE = "-" * 40
    sys.modules["tabulate"] = tabulate_mod

# Provide lightweight mock for typet5 if not installed in current environment
if "typet5" not in sys.modules:
    try:
        import typet5
    except ImportError:
        typet5_mod = types.ModuleType("typet5")
        tc_mod = types.ModuleType("typet5.type_check")
        class MockTypeNode:
            def __init__(self, s):
                self.s = str(s)
                self.head = str(s).split("[")[0].split(".")[-1]
            def normalized(self): return self
            def __str__(self): return self.s
        tc_mod.parse_type_str = lambda s: MockTypeNode(s)
        tc_mod.normalize_type = lambda s: MockTypeNode(s)
        typet5_mod.type_check = tc_mod
        sys.modules["typet5"] = typet5_mod
        sys.modules["typet5.type_check"] = tc_mod

# Provide dummy analysis.run module definition so pickle loads without importing heavy dependencies
if "analysis.run" not in sys.modules:
    try:
        from analysis.run import Type4PyAnalysis
    except Exception:
        import evaluation.eval
        analysis_mod = types.ModuleType("analysis")
        analysis_run_mod = types.ModuleType("analysis.run")
        class TigerAnalysis:
            def print_top_k(self): self.problem_list.print_top_k()
            def print_base_top_k(self): self.problem_list.print_base_top_k()
            def print_categorized_top_k(self): self.problem_list.print_categorized_top_k()
            def print_arg_ret_top_k(self): self.problem_list.print_arg_ret_top_k()
        class Type4PyAnalysis(TigerAnalysis): pass
        analysis_run_mod.TigerAnalysis = TigerAnalysis
        analysis_run_mod.Type4PyAnalysis = Type4PyAnalysis
        sys.modules["analysis"] = analysis_mod
        sys.modules["analysis.run"] = analysis_run_mod

import pickle

with open(RESULT_PKL, "rb") as f:
    res = pickle.load(f)

print("=" * 70)
print(f" TypeCare + Type4Py Evaluation Results (ManyTypes4Py Benchmark)")
print(f" Total targets evaluated: {len(res.problem_list.get_problem_list()):,}")
print("=" * 70)

print("\n[1] Overall Top-K Accuracy (Exact Match)")
print("-" * 70)
res.print_top_k()

print("\n[2] Base Top-K Accuracy (Base Type Match)")
print("-" * 70)
res.print_base_top_k()

print("\n[3] Categorized Top-K Accuracy")
print("-" * 70)
res.print_categorized_top_k()

print("\n[4] Argument vs. Return Type Top-K Accuracy")
print("-" * 70)
res.print_arg_ret_top_k()
