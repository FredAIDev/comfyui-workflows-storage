#!/usr/bin/env python3
"""Refresh helper for model-sources/model-sources.json.

Polls the HuggingFace API for every frozen author (repo lists AND per-repo
file lists), diffs against the referential, and writes
model-sources/candidates.json for HUMAN ratification.

This script NEVER writes model-sources.json — the referential is updated by
hand from the candidates (see model-sources/README.md, "Update method").

Usage:
    python tools/refresh_model_sources.py            # full pass
    python tools/refresh_model_sources.py --authors unsloth QuantStack
    python tools/refresh_model_sources.py --no-files  # repo lists only (fast)

stdlib only — no dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "model-sources" / "model-sources.json"
CANDIDATES = ROOT / "model-sources" / "candidates.json"

API = "https://huggingface.co/api"
UA = {"User-Agent": "comfyui-workflows-storage/refresh_model_sources"}
MODEL_EXT = (".safetensors", ".gguf", ".sft", ".json")  # .json = runtime-quant configs (QuantFunc) + comfy_config


def get_json(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429 and attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            raise
    return None


def list_author_repos(author: str) -> list[dict]:
    """All model repos of an author: [{id, lastModified, downloads}]."""
    out, url = [], f"{API}/models?author={author}&limit=100&sort=lastModified"
    data = get_json(url)
    for m in data or []:
        out.append({
            "id": m.get("id") or m.get("modelId"),
            "lastModified": m.get("lastModified"),
            "downloads": m.get("downloads"),
        })
    return out


def list_repo_files(repo_id: str) -> list[dict] | None:
    """Model-weight files of a repo with their BYTES IDENTITY, or None if the
    repo is gone: [{file, size, sha256}] from the tree endpoint. sha256 is the
    LFS oid (the content hash the ComfyDock store is keyed on) — NEVER the git
    blob sha1 (`oid` field), which must not be aliased into sha256. Non-LFS
    files (small configs) legitimately carry sha256=None.

    Follows the tree endpoint's cursor pagination (Link: rel=\"next\")."""
    out: list[dict] = []
    url: str | None = f"{API}/models/{repo_id}/tree/main?recursive=true"
    first = True
    while url:
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    page = json.load(resp)
                    link = resp.headers.get("Link") or ""
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None if first else out
                if e.code == 429 and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
            except urllib.error.URLError:
                if attempt < 2:
                    time.sleep(5)
                    continue
                raise
        else:
            return None if first else out
        first = False
        for item in page or []:
            path = item.get("path") or ""
            if item.get("type") != "file" or not path.lower().endswith(MODEL_EXT):
                continue
            lfs = item.get("lfs") or {}
            out.append({"file": path,
                        "size": item.get("size"),
                        "sha256": lfs.get("oid")})
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return sorted(out, key=lambda f: f["file"])


def referenced_repos(sources: dict) -> set[str]:
    refs: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            repo = node.get("repo")
            if isinstance(repo, str) and "/" in repo:
                refs.add(repo)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(sources.get("model_lines", []))
    return refs


def frozen_authors(sources: dict) -> list[str]:
    tiers = sources.get("sources_frozen", {})
    out: list[str] = []
    for key, val in tiers.items():
        if key.startswith("_") or key == "watch_not_frozen":
            continue
        out.extend(val)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--authors", nargs="*", help="restrict to these authors")
    ap.add_argument("--no-files", action="store_true",
                    help="skip per-repo file enumeration (repo lists only)")
    args = ap.parse_args()

    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    authors = args.authors or frozen_authors(sources)
    refs = referenced_repos(sources)

    report = {
        "generated_by": "tools/refresh_model_sources.py",
        "note": "HUMAN-RATIFIED input for model-sources.json — never merged automatically.",
        "authors_polled": authors,
        "new_repos": {},        # author -> [repo ids not referenced yet]
        "gone_repos": [],       # referenced repos that 404
        "file_enumerations": {},  # repo id -> [{file, size, sha256}] (bytes identity rides with the name)
        "errors": [],
    }

    seen_repo_ids: set[str] = set()
    for author in authors:
        try:
            repos = list_author_repos(author)
        except Exception as e:  # noqa: BLE001 — report, keep going
            report["errors"].append(f"{author}: {e}")
            continue
        seen_repo_ids.update(r["id"] for r in repos)
        fresh = [r for r in repos if r["id"] not in refs]
        if fresh:
            report["new_repos"][author] = fresh
        print(f"[scan] {author}: {len(repos)} repos, {len(fresh)} not referenced")

    for repo in sorted(refs):
        author = repo.split("/")[0]
        if args.authors and author not in args.authors:
            continue
        try:
            files = list_repo_files(repo)
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"{repo}: {e}")
            continue
        if files is None:
            report["gone_repos"].append(repo)
            print(f"[GONE] {repo}")
        elif not args.no_files:
            report["file_enumerations"][repo] = files

    CANDIDATES.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    n_new = sum(len(v) for v in report["new_repos"].values())
    print(f"\n== {n_new} new repos · {len(report['gone_repos'])} gone · "
          f"{len(report['file_enumerations'])} file lists · "
          f"{len(report['errors'])} errors")
    print(f"-> {CANDIDATES.relative_to(ROOT)} (ratify by hand, then commit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
