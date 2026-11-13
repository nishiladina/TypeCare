"""Download the pinned CrossDomainTypes4Py repositories.

Run from the TypeCare root in Docker::

    python benchmarks/cdtypes4py/download_crossdomaintypes4py.py
    python benchmarks/cdtypes4py/download_crossdomaintypes4py.py --domain web --limit 5

Requires Git on PATH; uses only the Python standard library. Repositories are
stored at CrossDomainTypes4Py/<domain>/repos/<owner>/<repository>. Existing
directories are never overwritten: only repositories already at the requested
commit are skipped. Each domain gets a download_report.json for auditing failures.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SPECS = {
    "web": SCRIPT_DIR / "web_domain.spec",
    "scientific": SCRIPT_DIR / "scientific_calculation_domain.spec",
}


def read_spec(path):
    """Validate the manifest and remove duplicate entries, preserving order."""
    entries = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected a repository URL and commit hash")
        url, commit = fields
        parsed = urlparse(url)
        parts = parsed.path.removeprefix("/").removesuffix(".git").split("/")
        if (parsed.scheme != "https" or parsed.netloc != "github.com"
                or parsed.query or parsed.fragment or len(parts) != 2
                or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
                       or part in {".", ".."} for part in parts)
                or not re.fullmatch(r"[0-9a-fA-F]{40}", commit)):
            raise ValueError(f"{path}:{number}: invalid GitHub URL or full commit hash")
        name = "/".join(parts)
        commit = commit.lower()
        if name in entries and entries[name][1] != commit:
            raise ValueError(f"{path}:{number}: conflicting commits for {name}")
        entries[name] = (url, commit)
    return [(name, url, commit) for name, (url, commit) in entries.items()]


def run_git(*args, timeout):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    result = subprocess.run(
        ["git", *map(str, args)], capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env, timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def download_single(entry, repos_dir, timeout):
    name, url, commit = entry
    destination = repos_dir / name
    result = {"repo_name": name, "url": url, "commit": commit,
              "path": str(destination)}
    try:
        if destination.exists():
            if not (destination / ".git").is_dir():
                raise RuntimeError("Destination exists but is not a Git repository")
            head = run_git("-C", destination, "rev-parse", "HEAD", timeout=timeout)
            if head != commit:
                raise RuntimeError(f"Existing HEAD is {head}; expected {commit}. "
                                   "Use a different output directory or resolve manually.")
            result["status"] = "skipped"
            return result

        destination.parent.mkdir(parents=True, exist_ok=True)
        # Clone into a temporary sibling so a failed download cannot look complete.
        with tempfile.TemporaryDirectory(prefix=".cdtypes-", dir=destination.parent) as temp:
            staging = Path(temp) / "repo"
            run_git("clone", "--no-checkout", "--", url, staging, timeout=timeout)
            run_git("-C", staging, "checkout", "--detach", commit, timeout=timeout)
            head = run_git("-C", staging, "rev-parse", "HEAD", timeout=timeout)
            if head != commit:
                raise RuntimeError(f"Checkout produced {head}; expected {commit}")
            staging.rename(destination)
        result["status"] = "downloaded"
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result.update(status="failed", error=str(exc))
    return result


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=["all", *SPECS], default="all")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "CrossDomainTypes4Py")
    parser.add_argument("--workers", type=positive_int, default=10)
    parser.add_argument("--limit", type=positive_int,
                        help="Download only the first N unique repositories per domain")
    parser.add_argument("--timeout", type=positive_int, default=1800,
                        help="Timeout in seconds per Git command (default: 1800)")
    args = parser.parse_args()
    if shutil.which("git") is None:
        parser.error("Git must be installed and available on PATH")

    domains = list(SPECS) if args.domain == "all" else [args.domain]
    # Validate every selected manifest before starting any downloads.
    try:
        manifests = {domain: read_spec(SPECS[domain])[:args.limit] for domain in domains}
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    failures = 0
    for domain, entries in manifests.items():
        domain_dir = args.output_dir.resolve() / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        results = []
        print(f"{domain}: processing {len(entries)} repositories", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_single, entry, domain_dir / "repos",
                                       args.timeout) for entry in entries]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[{len(results)}/{len(entries)}] {result['status']}: "
                      f"{result['repo_name']}", flush=True)
                if result["status"] == "failed":
                    print(f"  {result['error']}", flush=True)

        report = domain_dir / "download_report.json"
        report.write_text(json.dumps(sorted(results, key=lambda r: r["repo_name"]),
                                     indent=2) + "\n", encoding="utf-8")
        counts = {status: sum(r["status"] == status for r in results)
                  for status in ("downloaded", "skipped", "failed")}
        failures += counts["failed"]
        print(f"{domain}: {counts}; report: {report}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
