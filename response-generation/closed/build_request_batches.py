#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd


# =========================================================
# 🔷 HELPERS
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_tag(model_name: str) -> str:
    return model_name.lower().replace("/", "__").replace("-", "_").replace(":", "_")


def make_custom_id(model_name: str, root_id: str, language: str, prompt: str) -> str:
    """
    Stable and collision-resistant.
    """
    base = f"{model_name}__{root_id}__{language}"
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:10]
    return f"{base}__{h}"


# =========================================================
# 🔷 FILTER LOADING
# =========================================================
def load_filter_metadata(filter_file: Path) -> Tuple[Dict[str, set], Dict[Tuple[str, str], Dict[str, Any]]]:
    df_filter = pd.read_csv(filter_file)

    filter_map: Dict[str, set] = {}
    filter_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for _, row in df_filter.iterrows():
        rid = str(row["root_id"])
        lang = str(row["language"])

        if rid not in filter_map:
            filter_map[rid] = set()
        filter_map[rid].add(lang)

        filter_meta[(rid, lang)] = {
            "category": row.get("category"),
            "tier": row.get("tier"),
            "label": row.get("label"),
            "f1": row.get("f1"),
            "comet": row.get("comet"),
            "combined_score": row.get("combined_score"),
            "quality_bucket": row.get("quality_bucket"),
        }

    print(f"✅ Loaded filter metadata for {len(filter_map)} root_ids from {filter_file}")
    return filter_map, filter_meta


# =========================================================
# 🔷 FLATTEN BATCH JSON USING FILTER
# =========================================================
def flatten_prompts_filtered(
    data: List[Dict[str, Any]],
    filter_map: Dict[str, set],
    filter_meta: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    flat_data: List[Dict[str, Any]] = []

    for obj in data:
        rid = str(obj["id"])

        if rid not in filter_map:
            continue

        for lang, trans in obj.get("translation", {}).items():
            if lang not in filter_map[rid]:
                continue

            prompt = trans.get("prompt_translated_lang")
            if not prompt:
                continue

            flat_data.append({
                "root_id": rid,
                "language": lang,
                "prompt": prompt,
                "eval": filter_meta.get((rid, lang), {})
            })

    return flat_data


# =========================================================
# 🔷 BUILD PROVIDER-READY REQUEST OBJECTS
# =========================================================
def build_request_object(
    item: Dict[str, Any],
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float | None,
    seed: int | None,
    system_instruction: str,
) -> Dict[str, Any]:
    cid = make_custom_id(model_name, item["root_id"], item["language"], item["prompt"])

    return {
        "custom_id": cid,
        "provider": provider,
        "model": model_name,
        "root_id": item["root_id"],
        "language": item["language"],
        "prompt": item["prompt"],
        "system_instruction": system_instruction,
        "generation_config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": seed,
        },
        "eval": item.get("eval", {}),
    }


def build_request_batches_for_source_batch(
    source_batch_file: Path,
    filter_map: Dict[str, set],
    filter_meta: Dict[Tuple[str, str], Dict[str, Any]],
    provider: str,
    model_name: str,
    out_dir: Path,
    temperature: float,
    max_tokens: int,
    top_p: float | None,
    seed: int | None,
    system_instruction: str,
    chunk_size: int,
) -> List[Path]:
    data = load_json(source_batch_file)
    flat_data = flatten_prompts_filtered(data, filter_map, filter_meta)

    print(f"🚀 {source_batch_file.name}: {len(flat_data)} filtered prompt-language pairs")

    requests = [
        build_request_object(
            item=x,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            seed=seed,
            system_instruction=system_instruction,
        )
        for x in flat_data
    ]

    batch_tag = source_batch_file.stem  # e.g., batch_17
    req_dir = out_dir / provider / model_tag(model_name)
    ensure_dir(req_dir)

    out_paths: List[Path] = []
    for i in range(0, len(requests), chunk_size):
        chunk = requests[i:i + chunk_size]
        chunk_path = req_dir / f"{batch_tag}__chunk_{i // chunk_size:03d}.jsonl"
        save_jsonl(chunk, chunk_path)
        out_paths.append(chunk_path)

    return out_paths


# =========================================================
# 🔷 MAIN
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-file", required=True, help="CSV containing filtered root_id/language/eval metadata")
    parser.add_argument("--data-dir", required=True, help="Directory containing source batch_*.json files")
    parser.add_argument("--out-dir", required=True, help="Directory to write request batch JSONL files")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "gemini"])
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--end-batch", type=int, default=58)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    filter_map, filter_meta = load_filter_metadata(Path(args.filter_file))

    system_instruction = "You are a helpful, safe, and honest assistant."

    all_paths: List[Path] = []
    for batch_num in range(args.start_batch, args.end_batch + 1):
        source_batch = Path(args.data_dir) / f"batch_{batch_num}.json"
        if not source_batch.exists():
            print(f"⚠️ Skipping missing source batch: {source_batch}")
            continue

        out_paths = build_request_batches_for_source_batch(
            source_batch_file=source_batch,
            filter_map=filter_map,
            filter_meta=filter_meta,
            provider=args.provider,
            model_name=args.model_name,
            out_dir=Path(args.out_dir),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            seed=args.seed,
            system_instruction=system_instruction,
            chunk_size=args.chunk_size,
        )
        all_paths.extend(out_paths)

    print(f"\n✅ Total request batch files created: {len(all_paths)}")
    for p in all_paths:
        print(p)


if __name__ == "__main__":
    main()
