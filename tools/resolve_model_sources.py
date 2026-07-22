#!/usr/bin/env python3
"""REF-3 — resolve the model-source referential to FILE level.

Deterministic OFFLINE join of the curated skeleton (model-sources.json)
with the enumerated file lists (candidates.json, produced by
refresh_model_sources.py) into `model-sources/model-sources.resolved.json`:
one flat entry per FILE with its exact download URL, parsed quant token,
component classification, and the catalog name (svdq- injection rule —
Fred 2026-07-19: if neither 'svdq' nor 'nunchaku' appears in the source
name of an svdq weight, inject 'svdq-'; keep the source name otherwise).

QuantFunc runtime-inference CONFIGS (weights-less lines, e.g. Ideogram 4)
are first-class entries: component 'runtime_config', loader
'quantfunc-engine'.

Usage: python tools/resolve_model_sources.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "model-sources" / "model-sources.json"
CANDIDATES = ROOT / "model-sources" / "candidates.json"
RESOLVED = ROOT / "model-sources" / "model-sources.resolved.json"

WEIGHT_EXT = (".safetensors", ".gguf", ".sft")

_GGUF_QUANT = re.compile(
    r"[-_.](UD-[A-Za-z0-9_]+|IQ\d[A-Za-z0-9_]*|Q\d_K_[SML]|Q\d_K|Q\d_[01]|BF16|F16|F32)\.gguf$",
    re.IGNORECASE)

# ordered: first match wins (most specific first)
_ST_TOKENS = [
    "svdq-fp4", "svdq-int4", "nvfp4_mixed", "nvfp4", "mxfp8",
    "fp8mixed", "fp8_mixed", "fp8_scaled", "fp8_e4m3fn", "fp8_e5m2", "fp8",
    "int8_convrot", "int8-convrot", "convrot", "int8mixed",
    "int8rowwise", "int8_rowwise", "int8", "fp4_mixed", "fp4", "int4",
    "bf16", "fp16", "bnb4", "nf4",
]
_RANK = re.compile(r"_r(\d+)[-_.]")
# QuantFunc tier aliases (their model cards: ultimate_speed=r32, balance=r128,
# best_quality=r256)
_TIER_RANK = {"ultimate_speed": "32", "balance": "128", "best_quality": "256"}
# diffusers-layout artifacts (sharded or generic dumps) are NOT loadable by
# ComfyUI single-file loaders — excluded from the resolved referential
_DIFFUSERS_DUMP = re.compile(r"(^|/)diffusion_pytorch_model(-\d+-of-\d+)?\.safetensors$")


def quant_of(fname: str) -> tuple[str, str | None]:
    """(quant token, rank) parsed from a file name; ('base', None) if unmarked."""
    low = fname.lower()
    m = _GGUF_QUANT.search(fname)
    if m:
        return m.group(1).lower(), None
    rank = _RANK.search(low)
    rank_val = rank.group(1) if rank else None
    if rank_val is None:
        for tier, r in _TIER_RANK.items():
            if tier in low:
                rank_val = r
                break
    for tok in _ST_TOKENS:
        if tok in low:
            return tok.replace("-", "_"), rank_val
    return "base", rank_val


def component_of(path: str) -> str:
    """Classify a repo file path into dit / text_encoder / vae / lora / other."""
    p = path.lower()
    segs = p.split("/")
    if any(s in ("vae",) for s in segs) or "vae" in segs[-1].split("_"):
        return "vae"
    if any(s in ("text_encoders", "text_encoder", "clip", "clip_vision") for s in segs):
        return "text_encoder"
    if any(s in ("loras", "lora") for s in segs) or "lora" in segs[-1]:
        return "lora"
    if any(s in ("diffusion_models", "unet", "transformer", "transformers",
                 "checkpoints", "split_files") for s in segs[:-1]):
        # split_files/<component>/ already caught above when te/vae; rest = dit
        return "dit"
    base = segs[-1]
    if any(t in base for t in ("t5", "umt5", "qwen2.5", "qwen3", "gemma",
                               "mistral", "ministral", "encoder", "mmproj")):
        return "text_encoder"
    return "dit"


def svdq_catalog_name(fname: str, is_svdq: bool) -> str:
    base = fname.split("/")[-1]
    if is_svdq and "svdq" not in base.lower() and "nunchaku" not in base.lower():
        return "svdq-" + base
    return base


def url_of(repo: str, file: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{file}"


def main() -> int:
    skeleton = json.loads(SOURCES.read_text(encoding="utf-8"))
    cands = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    # Enumerations carry the BYTES IDENTITY since 2026-07-22 ({file, size,
    # sha256 = LFS oid}); plain-string lists from older candidates still
    # resolve (without identity) so the join never breaks on a stale file.
    enums: dict[str, list[dict]] = {
        repo: [f if isinstance(f, dict) else {"file": f, "size": None, "sha256": None}
               for f in files]
        for repo, files in cands.get("file_enumerations", {}).items()
    }

    entries: list[dict] = []
    missing_repos: set[str] = set()

    def resolve_source(line_id: str, grade: str, loader: str, hw: str,
                       status: str, repo: str, want: str) -> None:
        """want: 'dit' for grade sources, or the companion component name."""
        files = enums.get(repo)
        if files is None:
            missing_repos.add(repo)
            return
        is_svdq = loader.startswith("nunchaku") or grade.startswith("svdq") or grade == "svdq"

        def identity(meta: dict) -> dict:
            """size_mb + sha256 (LFS content hash) — None when unknown."""
            size = meta.get("size")
            return {"size_mb": round(size / (1024 * 1024), 1) if size else None,
                    "sha256": meta.get("sha256")}

        if status == "config-only":
            for meta in files:
                f = meta["file"]
                if f.lower().endswith(".json") and "config" in f.lower():
                    entries.append({
                        "line": line_id, "grade": "qfunc", "loader": "quantfunc-engine",
                        "hw": hw, "component": "runtime_config", "repo": repo,
                        "file": f, "url": url_of(repo, f), "quant": "runtime",
                        "rank": None, "catalog_name": f.split("/")[-1],
                        **identity(meta)})
            return
        for meta in files:
            f = meta["file"]
            if not f.lower().endswith(WEIGHT_EXT):
                continue
            if _DIFFUSERS_DUMP.search(f.lower()):
                continue          # diffusers layout — not ComfyUI-loadable
            comp = component_of(f)
            if want == "dit" and comp not in ("dit", "lora"):
                continue
            if want != "dit" and comp != want:
                continue
            quant, rank = quant_of(f)
            entries.append({
                "line": line_id, "grade": grade, "loader": loader, "hw": hw,
                "component": comp, "repo": repo, "file": f,
                "url": url_of(repo, f), "quant": quant, "rank": rank,
                "catalog_name": svdq_catalog_name(f, is_svdq),
                **identity(meta)})

    # companion inheritance resolution
    lines = {ln["id"]: ln for ln in skeleton["model_lines"]}

    def companions_of(ln: dict) -> dict:
        comp = ln.get("companions") or {}
        if "_inherit" in comp:
            return companions_of(lines[comp["_inherit"]])
        return comp

    for ln in skeleton["model_lines"]:
        lid = ln["id"]
        for g in ln.get("grades", []):
            for src in g.get("sources", []):
                resolve_source(lid, g["grade"], g["loader"], g.get("hw", "any"),
                               g.get("status", "weights"), src["repo"], "dit")
        comps = companions_of(ln)
        for comp_name in ("text_encoder", "vae"):
            for src in comps.get(comp_name, []):
                resolve_source(lid, "companion", src.get("loader", "standard"),
                               "any", "weights", src["repo"], comp_name)

    # de-duplicate (inherited companions repeat across lines is INTENTIONAL —
    # dedupe only exact duplicates within a line)
    seen, unique = set(), []
    for e in entries:
        key = (e["line"], e["grade"], e["component"], e["repo"], e["file"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)

    by_line: dict[str, int] = {}
    for e in unique:
        by_line[e["line"]] = by_line.get(e["line"], 0) + 1

    out = {
        "version": 1,
        "generated_by": "tools/resolve_model_sources.py (deterministic join, re-runnable)",
        "note": "One entry per FILE: exact HF download url, quant token, component, catalog_name (svdq- rule applied). Consumed by the ComfyDock Derive feature. Regenerate after every refresh pass.",
        "counts": {"entries": len(unique), "by_line": dict(sorted(by_line.items()))},
        "missing_enumerations": sorted(missing_repos),
        "entries": unique,
    }
    RESOLVED.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"resolved {len(unique)} file entries across {len(by_line)} lines "
          f"-> {RESOLVED.relative_to(ROOT)}")
    if missing_repos:
        print(f"WARNING: {len(missing_repos)} referenced repos lack enumeration "
              f"(re-run refresh): {', '.join(sorted(missing_repos)[:6])}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
