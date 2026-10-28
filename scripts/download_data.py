"""Download the PhysioNet/CinC Challenge 2019 training data (sets A and B).

Files are fetched one by one from the PhysioNet directory listing, because the
project has no bulk archive endpoint. A pool of 16 threads is used, each with its
own keep alive HTTPS connection, with retries and exponential backoff. Files that
already exist are skipped, so the script can be restarted at any time.

Usage:
    python scripts/download_data.py [--sets A B] [--workers 16] [--limit N]
"""

import argparse
import http.client
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HOST = "physionet.org"
BASE = "/files/challenge-2019/1.0.0/training/training_set{}/"
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"

_local = threading.local()


def _conn():
    if getattr(_local, "conn", None) is None:
        _local.conn = http.client.HTTPSConnection(HOST, timeout=60)
    return _local.conn


def _get(path, retries=6):
    delay = 1.0
    for attempt in range(retries):
        try:
            conn = _conn()
            conn.request("GET", path, headers={"Connection": "keep-alive"})
            resp = conn.getresponse()
            body = resp.read()
            if resp.status == 200:
                return body
            raise IOError(f"HTTP {resp.status} for {path}")
        except Exception:
            try:
                _local.conn.close()
            except Exception:
                pass
            _local.conn = None
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return None


def list_files(set_name):
    html = _get(BASE.format(set_name)).decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="(p\d+\.psv)"', html)))


def fetch(set_name, name, out_dir):
    target = out_dir / name
    if target.exists() and target.stat().st_size > 0:
        return "skip"
    data = _get(BASE.format(set_name) + name)
    tmp = target.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["A", "B"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="files per set (for quick tests)")
    args = ap.parse_args()

    for set_name in args.sets:
        out_dir = ROOT / f"set{set_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        names = list_files(set_name)
        if args.limit:
            names = names[: args.limit]
        print(f"set {set_name}: {len(names)} files listed", flush=True)
        done = failed = 0
        start = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch, set_name, n, out_dir): n for n in names}
            for fut in as_completed(futures):
                try:
                    fut.result()
                    done += 1
                except Exception as exc:
                    failed += 1
                    print(f"failed {futures[fut]}: {exc}", flush=True)
                if done % 500 == 0:
                    rate = done / max(time.time() - start, 1e-9)
                    print(f"set {set_name}: {done}/{len(names)} ({rate:.1f} files/s)", flush=True)
        print(f"set {set_name}: finished, {done} ok, {failed} failed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
