#!/usr/bin/env python3
"""
Generate Type4Py predictions for the ManyTypes4Py evaluation benchmark in TypeCare.

This script runs the pretrained Type4Py model on the 2,687 test targets
used by TypeCare (matching Tiger and TypeGen benchmarks), producing
prediction/Type4Py/transformed_result.json with top-10 candidate types.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("type4py_prediction_generator")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Type4Py predictions for the TypeCare ManyTypes4Py benchmark."
    )
    repo_root = Path(__file__).resolve().parent.parent
    default_model_dir = os.environ.get(
        "TYPE4PY_MODEL_DIR",
        str(Path("/Volumes/My Passport for Mac/TypeCare/t4py_model_files")),
    )
    default_benchmark_dir = str(repo_root / "ManyTypes4Py" / "repos")
    default_target_file = str(repo_root / "prediction" / "Tiger" / "transformed_result.json")
    default_output_file = str(repo_root / "prediction" / "Type4Py" / "transformed_result.json")

    parser.add_argument(
        "--model_dir",
        type=str,
        default=default_model_dir,
        help="Directory containing Type4Py pretrained model files.",
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=default_benchmark_dir,
        help="Directory containing ManyTypes4Py repository clones.",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default=default_target_file,
        help="Path to Tiger or TypeGen transformed_result.json defining benchmark targets.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=default_output_file,
        help="Output path for Type4Py transformed_result.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of targets to process (useful for testing).",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Save checkpoint to output file every N targets.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from existing output file; restart from scratch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to run ONNX Runtime on.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top predictions to retain per target.",
    )
    parser.add_argument(
        "--annoy_k",
        type=int,
        default=50,
        help="Number of nearest neighbors to query from Annoy before deduplicating cluster labels.",
    )

    return parser.parse_args()


def load_model(model_dir: str, device: str = "cpu"):
    """Load pretrained Type4Py model with PCA."""
    from type4py.deploy.infer import PretrainedType4Py

    logger.info(f"Loading PretrainedType4Py from {model_dir} (device={device}, use_pca=True)...")
    pt4py = PretrainedType4Py(model_dir, device=device, use_pca=True)
    pt4py.load_pretrained_model()
    logger.info("Type4Py model loaded successfully.")
    return pt4py


def find_target_slot_index(
    ext: Dict[str, Any], target: str, param: str, all_type_slots: List[Tuple[Any, str]]
) -> Optional[int]:
    """Find the index in all_type_slots that matches target function/method and param."""
    if "." in target:
        c_name, m_name = target.split(".", 1)
        for c in ext.get("classes", []):
            if c.get("name") == c_name:
                for fn in c.get("funcs", []):
                    if fn.get("name") == m_name:
                        target_dict = fn if param == "__RET__" else fn.get("params_p", {})
                        target_key = "ret_type_p" if param == "__RET__" else param
                        for idx, (p_dict, p_key) in enumerate(all_type_slots):
                            if p_dict is target_dict and p_key == target_key:
                                return idx
    else:
        for fn in ext.get("funcs", []):
            if fn.get("name") == target:
                target_dict = fn if param == "__RET__" else fn.get("params_p", {})
                target_key = "ret_type_p" if param == "__RET__" else param
                for idx, (p_dict, p_key) in enumerate(all_type_slots):
                    if p_dict is target_dict and p_key == target_key:
                        return idx
    return None


def main():
    args = parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    target_file = Path(args.target_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not target_file.exists():
        logger.error(f"Target file does not exist: {target_file}")
        sys.exit(1)

    if not benchmark_dir.exists():
        logger.error(f"Benchmark repos directory does not exist: {benchmark_dir}")
        sys.exit(1)

    with open(target_file, "r", encoding="utf-8") as f:
        all_targets: List[Dict[str, Any]] = json.load(f)

    if args.limit:
        all_targets = all_targets[: args.limit]
        logger.info(f"Limited processing to first {len(all_targets)} targets.")

    total_targets = len(all_targets)
    logger.info(f"Loaded {total_targets} benchmark targets from {target_file}")

    # Check for existing checkpoint
    completed_results: List[Dict[str, Any]] = []
    processed_keys = set()
    if not args.no_resume and output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                completed_results = json.load(f)
            processed_keys = {
                (r["repo_name"], r["file_path"], r["target"], tuple(r["params"]))
                for r in completed_results
            }
            logger.info(
                f"Resuming: found {len(completed_results)} already completed targets in {output_file}"
            )
        except Exception as e:
            logger.warning(f"Could not read existing output file ({e}), starting fresh.")
            completed_results = []
            processed_keys = set()

    # Load model and helper functions
    pt4py = load_model(args.model_dir, device=args.device)
    from libsa4py.cst_extractor import Extractor
    from type4py.deploy.infer import (
        get_dps_single_file,
        infer_single_dp,
        param2vec,
        ret2vec,
        type_embed_single_dp,
        var2vec,
    )

    # Group targets by file to avoid re-running feature extraction on the same file
    targets_by_file: Dict[Tuple[str, str], List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, item in enumerate(all_targets):
        key = (item["repo_name"], item["file_path"], item["target"], tuple(item["params"]))
        if key in processed_keys:
            continue
        file_key = (item["repo_name"], item["file_path"])
        targets_by_file[file_key].append((idx, item))

    remaining_targets = sum(len(ts) for ts in targets_by_file.values())
    logger.info(
        f"Processing {remaining_targets} remaining targets across {len(targets_by_file)} files..."
    )

    start_time = time.time()
    targets_done = 0

    # Build output dictionary keyed by original index so order is preserved
    results_by_idx: Dict[int, Dict[str, Any]] = {}
    for item in completed_results:
        for orig_idx, orig_target in enumerate(all_targets):
            if (
                orig_target["repo_name"] == item["repo_name"]
                and orig_target["file_path"] == item["file_path"]
                and orig_target["target"] == item["target"]
                and orig_target["params"] == item["params"]
            ):
                results_by_idx[orig_idx] = item
                break

    for file_key, target_list in targets_by_file.items():
        repo_name, file_path_rel = file_key
        full_file_path = benchmark_dir / repo_name / file_path_rel

        if not full_file_path.exists():
            logger.warning(f"File not found: {full_file_path}. Filling fallback.")
            for orig_idx, item in target_list:
                result_item = dict(item)
                result_item["predictions"] = [["typing.Any"]]
                result_item["total_predictions"] = None
                results_by_idx[orig_idx] = result_item
                targets_done += 1
            continue

        try:
            with open(full_file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()

            ext = Extractor.extract(code, include_seq2seq=False).to_dict()
            all_type_slots, vars_type_hints, params_type_hints, rets_type_hints = (
                get_dps_single_file(ext)
            )

            id_dps, code_tks_dps, vth_dps = [], [], []
            if len(vars_type_hints) != 0:
                v_id, v_code, v_vth = var2vec(vars_type_hints, pt4py.w2v_model)
                id_dps.append(v_id)
                code_tks_dps.append(v_code)
                vth_dps.append(v_vth)
            if len(params_type_hints) != 0:
                p_id, p_code, p_vth = param2vec(params_type_hints, pt4py.w2v_model)
                id_dps.append(p_id)
                code_tks_dps.append(p_code)
                vth_dps.append(p_vth)
            if len(rets_type_hints) != 0:
                r_id, r_code, r_vth = ret2vec(rets_type_hints, pt4py.w2v_model)
                id_dps.append(r_id)
                code_tks_dps.append(r_code)
                vth_dps.append(r_vth)

            if len(id_dps) == 0:
                # No typed slots in this file
                preds = []
            else:
                id_dps = np.concatenate(tuple(id_dps)) if len(id_dps) > 1 else id_dps[0]
                code_tks_dps = (
                    np.concatenate(tuple(code_tks_dps)) if len(code_tks_dps) > 1 else code_tks_dps[0]
                )
                vth_dps = np.concatenate(tuple(vth_dps)) if len(vth_dps) > 1 else vth_dps[0]

                preds = type_embed_single_dp(pt4py.type4py_model, id_dps, code_tks_dps, vth_dps)
                if pt4py.use_pca:
                    preds = pt4py.type4py_pca.transform(preds)

            for orig_idx, item in target_list:
                param_name = item["params"][0] if item["params"] else "__RET__"
                target_idx = find_target_slot_index(
                    ext, item["target"], param_name, all_type_slots
                )

                if (
                    target_idx is not None
                    and len(preds) > 0
                    and target_idx < len(preds)
                ):
                    top_clusters = infer_single_dp(
                        pt4py.type_clusters_idx,
                        args.annoy_k,
                        pt4py.type_clusters_labels,
                        preds[target_idx],
                    )
                    pred_labels = [
                        pt4py.label_enc.inverse_transform([p])[0] for p, s in top_clusters
                    ]
                    # Filter and retain top_k unique candidate strings
                    unique_candidates = []
                    seen = set()
                    for lbl in pred_labels:
                        if lbl not in seen:
                            seen.add(lbl)
                            unique_candidates.append(lbl)
                            if len(unique_candidates) >= args.top_k:
                                break
                    candidate_preds = [[c] for c in unique_candidates] if unique_candidates else [["typing.Any"]]
                else:
                    candidate_preds = [["typing.Any"]]

                result_item = dict(item)
                result_item["predictions"] = candidate_preds
                result_item["total_predictions"] = None
                results_by_idx[orig_idx] = result_item
                targets_done += 1

        except Exception as e:
            logger.error(f"Error processing {full_file_path}: {e}")
            for orig_idx, item in target_list:
                result_item = dict(item)
                result_item["predictions"] = [["typing.Any"]]
                result_item["total_predictions"] = None
                results_by_idx[orig_idx] = result_item
                targets_done += 1

        # Checkpoint periodically
        if targets_done > 0 and targets_done % args.checkpoint_every == 0:
            ordered_results = [results_by_idx[i] for i in sorted(results_by_idx.keys())]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(ordered_results, f, indent=4)
            elapsed = time.time() - start_time
            rate = targets_done / elapsed if elapsed > 0 else 0
            logger.info(
                f"Checkpoint: {len(ordered_results)}/{total_targets} targets done "
                f"({rate:.2f} targets/sec, elapsed: {elapsed:.1f}s)"
            )

    # Final save
    ordered_results = [results_by_idx[i] for i in sorted(results_by_idx.keys())]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=4)

    total_time = time.time() - start_time
    logger.info(
        f"Done! Successfully generated predictions for {len(ordered_results)} targets "
        f"in {total_time:.1f}s. Saved to {output_file}"
    )


if __name__ == "__main__":
    main()
