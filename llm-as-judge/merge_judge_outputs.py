#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import hashlib
import pandas as pd
import re
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def model_tag(model_name: str) -> str:
    return model_name.lower().replace("/", "__").replace("-", "_").replace(":", "_")

def load_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]

def save_jsonl(rows, p):
    print("save_jsonl called...")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 🔥 extract judge.response safely
def extract_judge(j):
    print("extract_judge called...")
    print("j: ", j)
    if not isinstance(j, dict):
        return {"judge_parse_error": True}
    resp = j.get("response")
    if isinstance(resp, dict):
        return resp
    return {"judge_parse_error": True}

# def make_custom_id(row: Dict[str, Any]) -> str:
#     base = f"{row.get('meta', {}).get('model')}__{row.get('root_id')}__{row.get('language')}"
#     # print("response: ", row.get("response"))
#     h = hashlib.md5(row.get("response", "").encode()).hexdigest()[:10]
#     return f"{base}__{h}"

def make_custom_id(row: Dict[str, Any]) -> str:
    # Anthropic requires: ^[a-zA-Z0-9_-]{1,64}$
    # — only alphanumeric, underscore, hyphen; max 64 chars.
    # Strategy: hash model name + sanitise lang, keep root_id prefix.

    def sanitise(s: str) -> str:
        """Remove any character not in [a-zA-Z0-9_-]."""
        return re.sub(r'[^a-zA-Z0-9_-]', '', s)

    model_full = row.get("meta", {}).get("model", "unknown")
    model_hash = hashlib.md5(model_full.encode()).hexdigest()[:8]   # 8 chars

    root_id   = sanitise(row.get("root_id")   or "noid")[:16]      # ≤16 chars
    lang      = sanitise(row.get("language")  or "nolang")[:10]    # ≤10 chars
    resp_hash = hashlib.md5(
                    row.get("response", "").encode()
                ).hexdigest()[:8]                                    # 8 chars

    # Format: 8__≤16__≤10__8  →  max = 8+2+16+2+10+2+8 = 48 chars
    cid = f"{model_hash}__{root_id}__{lang}__{resp_hash}"
    assert len(cid) <= 64 and re.match(r'^[a-zA-Z0-9_-]+$', cid), \
        f"Invalid custom_id: '{cid}' (len={len(cid)})"
    return cid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--judge-output-dir", required=True)
    parser.add_argument("--judge-merged-dir", required=True)
    parser.add_argument("--provider", required=True, help="Provider to use")
    parser.add_argument("--judge-model", required=True, help="Judge model to use")
    # parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()

    # ===============================
    # 1. Load original data
    # ===============================
    input_dir = Path(args.input_dir)
    judge_output_dir = Path(args.judge_output_dir) / args.provider / model_tag(args.judge_model)
    merged_dir = Path(args.judge_merged_dir) / args.provider / model_tag(args.judge_model)
    ensure_dir(judge_output_dir)
    ensure_dir(merged_dir)

    print("input_dir: ", input_dir)
    print("judge_output_dir: ", judge_output_dir)
    print("merged_dir: ", merged_dir)


    orig = []
    for f in Path(input_dir).glob("*.jsonl"):
        orig.extend(load_jsonl(f))
        print("orig.len: ", len(orig))

    # ===============================
    # 2. Load judge data → map by custom_id
    # ===============================
    judge_map = {}
    print("judge_output_dir: ", judge_output_dir)

    for f in Path(judge_output_dir).glob("*.jsonl"):
        rows = load_jsonl(f)
        print("rows.len: ", len(rows))

        for r in rows:
            # cid = r.get("custom_id")
            cid_full = r.get("custom_id", "")
            if cid_full:
                print("judge full cid: ", cid_full)
                cid = "__".join(cid_full.split("__")[1:3])
                print("judge -> cid[1:3]: ", cid)

                # 🔥 extract ONLY judge.response
                if "judge" in r:
                    judge_clean = r.get("judge")
                elif "response" in r:
                    judge_clean = r.get("response")
                # judge_clean = extract_judge(r.get("judge"))

                print("judge.cid.len: ", len(cid.split("__")[0]))
                judge_map[cid] = judge_clean
                
    print("judge_map.keys(): ", judge_map.keys())
    # ===============================
    # 3. Merge into original objects
    # ===============================
    merged = []

    for row in orig:
        # cid = row.get("custom_id")
        # cid_full = row.get("custom_id", "")

        # if "gpt-oss" in args.input_dir and row.get("response", ""):
        if row.get("response", ""):
            cid = make_custom_id(row)
            print("orig full cid: ", cid)
            cid = "__".join(cid.split("__")[1:3])
            print("orig cid[1:3]: ", cid)
            print("orig.cid.len: ", len(cid.split("__")[0]))
            
            # elif cid_full:
            #     cid = "__".join(cid_full.split("__")[:3])
            #     print("original -> cid: ", cid)

            # add judge field
            # row["judge"] = judge_map.get(cid, {"judge_missing": True})

            if cid in judge_map:
                row["judge"] = judge_map[cid]
                merged.append(row)
        

    # ===============================
    # 4. Save JSONL
    # ===============================
    print("merged.len: ", len(merged))
    save_jsonl(merged, merged_dir / Path("output.jsonl"))

    print(f"✅ merged {len(merged)} rows → output.jsonl")
    print("merged_dir: ", merged_dir)

if __name__ == "__main__":
    main()
