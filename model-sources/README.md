# Model-source referential

Two shared data files make this repo the machine-readable statement of "what
ComfyDock can do" — for the user AND for third-party tools (Muse included).
The contract is the SCHEMA of these files, whatever the channel they are read
through (git clone, ComfyUI `/userdata`, or a future serving endpoint).

| File | Question it answers |
|---|---|
| `model-sources/model-sources.json` | which quantized weights EXIST in the world, where, and which are mutually replaceable |
| `workflows/families.json` | which workflow files share the same intention with different tradeoffs |

Doctrine (D-2026-07-18-01, Muse repo): ComfyDock = SUPPLY (conforms the bench),
Muse = DEMAND (adopts from the bench). These files carry ZERO Muse parameters;
every entry is creatable entirely outside Muse.

## The core notion: interchangeability

Within a `model_line`, **all DiT files whose grade shares the same `loader`
are mutually replaceable in that loader's ComfyUI node**. Same rule per
companion component (text_encoder, vae). Examples:

- `qwen_image_edit_2511` × loader `gguf`: Q8_0, Q6_K, Q5_K_M, Q4_K_M… all swap
  inside `Unet Loader (GGUF)`.
- `qwen_image_edit_2511` × loader `nunchaku`: svdq-fp4 and svdq-int4 (any rank)
  all swap inside `NunchakuQwenImageDiTLoader` (fp4 = Blackwell only, see `hw`).
- loader `standard`: bf16 ↔ fp8 ↔ fp8_scaled ↔ mxfp8 ↔ nvfp4 ↔ int8-convrot
  all swap inside `Load Diffusion Model` (comfy_quant metadata, ComfyUI ≥ 0.27).

**Derivation contract (for the one-click GUI): same `loader` ⇒ deriving a
variant workflow = replacing the component `file` values, nothing else.**
Different `loader` ⇒ a node swap, i.e. a different graph — out of scope of the
one-click replacement.

## Schema notes

- `sources_frozen`: the ratified publisher tiers (2026-07-18). Additions go
  through the update method below. ⚠️ Never reference `nunchaku-tech` on HF —
  it is an EMPTY org held by an unrelated user; the official org is
  `nunchaku-ai`.
- `status`: `weights` (verified present) · `config-only` (recipe, no weights —
  e.g. QuantFunc/Ideogram-4-Series) · `diffusers-only` (NOT ComfyUI-loadable —
  e.g. Krea 2 svdq) · `stale` (unmaintained 2024-era) · `experimental`.
- `hw`: `any` · `blackwell` (nvfp4, svdq-fp4) — instance-conformance input.
- Resolved entries carry the **bytes identity** since 2026-07-22: `size_mb`
  and `sha256` (the HF **LFS oid**, i.e. the content hash — never the git
  blob sha1). Non-LFS files (small configs) legitimately have `sha256: null`.
  Consumers (ComfyDock import) fill absent local values from these and
  NEVER overwrite a locally computed hash.
- `files: null` = not yet enumerated. Run the refresh script, then ratify.
- `companions._inherit`: reuse another line's companion set (e.g. all Qwen
  editions share the Qwen2.5-VL-7B TE and the qwen_image VAE).

## Update method (manual, ratified)

1. Run `python tools/refresh_model_sources.py` (network access to
   huggingface.co required). It polls the HF API for every frozen author —
   repo lists AND per-repo file lists (Comfy-Org adds quant files into
   existing repos continuously, so file-level re-poll matters).
2. It writes `model-sources/candidates.json` and prints a human diff:
   new repos not referenced, referenced repos gone/renamed, and candidate
   file enumerations for every `files: null` entry.
3. YOU ratify: copy the entries you accept into `model-sources.json`
   (pick the quant levels you actually want; pair the TE quant you trust).
   The script NEVER writes `model-sources.json` itself.
4. Commit both files. The git history is the referential's audit trail.

New publisher discovered? Add it to the right `sources_frozen` tier first
(that's a ruling, not a script action), then re-run.

## families.json

Grouping manifest for workflow files. ComfyDock (or a human) edits it; Muse
reads it and appends members additive-only when it pushes/pulls a variant.
A file absent from the manifest is just ungrouped — fail-open, no breakage.
Convention: every muse-compliant workflow lives under `workflows/muse/`,
even though it runs fine without Muse.
