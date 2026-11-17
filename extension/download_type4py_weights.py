#!/usr/bin/env python3
"""
Download and extract official pre-trained Type4Py weights from ghcr.io.

Designed to be portable across different machines, storage drives, and Docker environments.
"""

import os
import sys
import argparse
from pathlib import Path
import requests
import tarfile
from tqdm import tqdm

DEFAULT_EXTERNAL_DIR = Path("/Volumes/My Passport for Mac/TypeCare/t4py_model_files")
REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_FILES = [
    "type4py_complete_model.onnx",
    "type4py_complete_type_cluster_reduced",
    "type_clusters_pca.pkl",
    "w2v_token_model.bin",
    "label_encoder_all.pkl",
    "type4py_complete_true.npy",
]


def resolve_default_model_dir() -> Path:
    """Dynamically determine a suitable destination path based on environment."""
    env_dir = os.environ.get("TYPE4PY_MODEL_DIR")
    if env_dir:
        return Path(env_dir)
    if DEFAULT_EXTERNAL_DIR.parent.exists():
        return DEFAULT_EXTERNAL_DIR
    return REPO_ROOT / "t4py_model_files"


def download_and_extract_weights(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Type4Py] Target directory: {output_dir}")

    # Check if files already exist
    existing = [f for f in EXPECTED_FILES if (output_dir / f).exists()]
    if len(existing) == len(EXPECTED_FILES):
        print("[Type4Py] All model files already present. Skipping download.")
        return

    print("[Type4Py] Requesting auth token from GitHub Container Registry (ghcr.io)...")
    token_url = "https://ghcr.io/token?service=ghcr.io&scope=repository:saltudelft/type4py:pull"
    res = requests.get(token_url, timeout=30)
    res.raise_for_status()
    token = res.json().get("token")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
    }

    print("[Type4Py] Fetching image manifest for ghcr.io/saltudelft/type4py:latest...")
    manifest_url = "https://ghcr.io/v2/saltudelft/type4py/manifests/latest"
    m_res = requests.get(manifest_url, headers=headers, timeout=30)
    m_res.raise_for_status()
    manifest = m_res.json()

    # Find the layer containing model weights (>500 MB)
    model_layer = None
    for layer in manifest.get("layers", []):
        if layer.get("size", 0) > 500 * 1024 * 1024:
            model_layer = layer
            break

    if not model_layer:
        # Fallback to known digest
        digest = "sha256:b5a33d8ff6b41aee495934e47de5c23b380ec94dd213f105e5d95dba080bee52"
        layer_size = 1124 * 1024 * 1024
    else:
        digest = model_layer["digest"]
        layer_size = model_layer["size"]

    print(f"[Type4Py] Downloading model layer ({round(layer_size / (1024**2), 2)} MB)...")
    blob_url = f"https://ghcr.io/v2/saltudelft/type4py/blobs/{digest}"
    blob_res = requests.get(blob_url, headers=headers, stream=True, timeout=60)
    blob_res.raise_for_status()

    # Stream & extract in real-time
    prefix = "type4py/t4py_model_files/"
    extracted_count = 0

    with tqdm.wrapattr(blob_res.raw, "read", total=layer_size, desc="Downloading & Extracting") as raw_stream:
        with tarfile.open(fileobj=raw_stream, mode="r|gz") as tar:
            for member in tar:
                if member.name.startswith(prefix) and member.isfile():
                    rel_name = member.name[len(prefix):]
                    dest_path = output_dir / rel_name
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, open(dest_path, "wb") as target:
                        chunk = source.read(1024 * 1024)
                        while chunk:
                            target.write(chunk)
                            chunk = source.read(1024 * 1024)
                    extracted_count += 1
                    print(f" -> Extracted: {rel_name} ({round(member.size / (1024**2), 2)} MB)")

    print(f"[Type4Py] Extraction completed ({extracted_count} files).")
    print("\nVerifying model files:")
    for f in EXPECTED_FILES:
        path = output_dir / f
        if path.exists():
            size_mb = round(path.stat().st_size / (1024**2), 2)
            print(f"  [OK] {f} ({size_mb} MB)")
        else:
            print(f"  [MISSING] {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Type4Py pre-trained model files.")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=resolve_default_model_dir(),
        help="Target folder for model files (default: detects external drive or repo root)",
    )
    args = parser.parse_args()
    download_and_extract_weights(args.output_dir)
