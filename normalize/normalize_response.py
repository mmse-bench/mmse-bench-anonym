#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


# =========================================================
# 🔷 HELPERS
# =========================================================
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_none(x):
    if x in ["None", "", "null"]:
        return None
    return x


def normalize_finish_reason(fr):
    if fr is None:
        return None

    fr = str(fr)

    if "MAX_TOKENS" in fr:
        return "MAX_TOKENS"
    if "STOP" in fr:
        return "STOP"
    if "LENGTH" in fr:
        return "MAX_TOKENS"

    return fr.upper()


# =========================================================
# 🔷 TOKEN EXTRACTION
# =========================================================
def extract_tokens(row):
    # Priority: response_raw → direct fields → meta
    raw = row.get("response_raw") or {}

    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}

    prompt_tokens = (
        usage.get("input_tokens")
        or row.get("prompt_length_tokens")
        or (row.get("meta") or {}).get("prompt_tokens")
    )

    completion_tokens = (
        usage.get("output_tokens")
        or row.get("response_length_tokens")
        or (row.get("meta") or {}).get("completion_tokens")
    )

    total_tokens = (
        usage.get("total_tokens")
        or (row.get("meta") or {}).get("total_tokens")
    )

    return prompt_tokens, completion_tokens, total_tokens


# =========================================================
# 🔷 META BUILDER
# =========================================================
def build_meta(row):
    meta_old = row.get("meta") or {}
    cfg = row.get("generation_config") or {}

    prompt = row.get("prompt") or ""
    response = row.get("response") or ""

    prompt_tokens, completion_tokens, total_tokens = extract_tokens(row)

    return {
        "provider": row.get("provider") or "open_source",
        "model": row.get("model") or meta_old.get("model"),

        "temperature": cfg.get("temperature") or meta_old.get("temperature"),
        "top_p": cfg.get("top_p") or meta_old.get("top_p"),
        "max_tokens": cfg.get("max_tokens") or meta_old.get("max_new_tokens"),

        "language": row.get("language"),

        # 🔥 unified tokens
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,

        # 🔥 char lengths
        "prompt_length_chars": len(prompt),
        "response_length_chars": len(response),

        # 🔥 runtime
        "time_per_sample_sec": row.get("time_per_sample_sec")
            or meta_old.get("time_per_sample_sec"),

        # 🔥 finish reason
        "finish_reason": normalize_finish_reason(row.get("finish_reason")),

        # 🔥 misc
        "token_count_method": row.get("token_count_method"),
        "provider_response_id": normalize_none(row.get("provider_response_id")),

        # 🔥 provenance
        "source_file": row.get("source_file")
            or row.get("source_request_file"),

        # placeholders for consistency
        "batch_samples_per_sec": None,
        "was_truncated": None
    }


# =========================================================
# 🔷 MAIN NORMALIZER
# =========================================================
def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # normalize null-like fields
    row["error"] = normalize_none(row.get("error"))
    row["provider_response_id"] = normalize_none(row.get("provider_response_id"))

    # ensure response_raw exists
    if "response_raw" not in row or row["response_raw"] is None:
        row["response_raw"] = None

    # build unified meta
    meta = build_meta(row)

    # final unified structure
    return {
        "root_id": row.get("root_id"),
        "language": row.get("language"),
        "prompt": row.get("prompt"),
        "response": row.get("response"),

        "meta": meta,
        "eval": row.get("eval", {}),

        "response_raw": row.get("response_raw"),

        # optional debugging fields
        "custom_id": row.get("custom_id"),
        "error": row.get("error"),
    }

from typing import Dict, Any


def fix_normalized_row(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = row.get("meta", {})
    provider = meta.get("provider")
    resp_raw = row.get("response_raw")

    # -------------------------
    # 1. Token normalization
    # -------------------------
    prompt_tokens = meta.get("prompt_tokens")
    completion_tokens = meta.get("completion_tokens")

    if meta.get("total_tokens") is None:
        if prompt_tokens is not None and completion_tokens is not None:
            meta["total_tokens"] = prompt_tokens + completion_tokens

    # -------------------------
    # 2. Finish reason extraction
    # -------------------------
    finish_reason = meta.get("finish_reason")

    if finish_reason is None and resp_raw is not None:
        try:
            if provider == "openai":
                if resp_raw.get("status") == "incomplete":
                    finish_reason = (
                        resp_raw.get("incomplete_details", {}) or {}
                    ).get("reason")
                else:
                    finish_reason = "stop"

            elif provider == "anthropic":
                finish_reason = resp_raw.get("stop_reason")

            elif provider == "gemini":
                fr = resp_raw.get("finish_reason")
                if fr is not None:
                    finish_reason = str(fr)

        except Exception:
            finish_reason = None

    meta["finish_reason"] = finish_reason

    # -------------------------
    # 3. Truncation detection
    # -------------------------
    truncation_reasons = {
        "max_output_tokens",
        "MAX_TOKENS",
        "max_tokens",
        "length",
        "LENGTH",
    }

    was_truncated = False

    if finish_reason in truncation_reasons:
        was_truncated = True

    # fallback: detect via token saturation
    elif (
        meta.get("max_tokens") is not None
        and completion_tokens is not None
        and completion_tokens >= meta.get("max_tokens")
    ):
        was_truncated = True

    meta["was_truncated"] = was_truncated

    # -------------------------
    # 4. Error normalization
    # -------------------------
    response = row.get("response")
    error = row.get("error")

    if response == "ERROR" or response is None:
        if error is None:
            error = "generation_failed"
        row["response"] = None

    row["error"] = error

    # -------------------------
    # 5. Token method standardization
    # -------------------------
    if meta.get("token_count_method") is None:
        if provider == "open_source":
            meta["token_count_method"] = "model_tokenizer"
        else:
            meta["token_count_method"] = "provider_reported_or_partial"

    # -------------------------
    # 6. Top-p normalization
    # -------------------------
    if meta.get("top_p") is None:
        meta["top_p"] = 1.0

    # -------------------------
    # OPTIONAL: generation status (VERY USEFUL)
    # -------------------------
    if meta["was_truncated"]:
        meta["generation_status"] = "truncated"
    elif row.get("error") is not None:
        meta["generation_status"] = "error"
    else:
        meta["generation_status"] = "complete"

    # -------------------------
    # Write back meta
    # -------------------------
    row["meta"] = meta

    return row

# =========================================================
# 🔷 PIPELINE
# =========================================================
def normalize_file(input_path: Path, output_path: Path):
    rows = load_jsonl(input_path)
    normalized = [normalize_row(r) for r in rows]
    normalized_fixed = [fix_normalized_row(r) for r in normalized]
    write_jsonl(output_path, normalized_fixed)

    print(f"✅ Normalized file saved to: {output_path}")
    print(f"Total rows: {len(normalized_fixed)}")


# =========================================================
# 🔷 CLI
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    normalize_file(Path(args.input), Path(args.output))
