#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert the rows in Aegis AI Content Safety v2.0 (train split) into dictionary obejct and store as a list of such objects in jsonl file.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Optional, Any
from datasets import load_dataset  
from transformers import pipeline 
from transformers import AutoTokenizer, pipeline


def translate_batch(texts: List[str], translator) -> List[str]:
    outputs = translator(texts, max_length=512)
    return [o["translation_text"] for o in outputs]


def parse_categories(violated: Optional[str]) -> (str, Optional[str]):
    if not violated:
        return "Safe", None
    parts = [p.strip() for p in violated.split(",") if p.strip()]
    if not parts:
        return "Safe", None
    primary = parts[0]
    secondary = parts[1] if len(parts) > 1 else None
    return primary, secondary

# ==============================
# Severity Ranking
# ==============================
severity_order = [
    "Sexual (minor)",
    "Suicide and Self Harm",
    "Threat",
    "Violence",
    "Guns and Illegal Weapons",
    "Criminal Planning/Confessions",
    "Illegal Activity",
    "Malware",
    "Fraud/Deception",
    "Controlled/Regulated Substances",
    "Hate/Identity Hate",
    "Harassment",
    "Sexual",
    "Political/Misinformation/Conspiracy",
    "High Risk Gov Decision Making",
    "Unauthorized Advice",
    "PII/Privacy",
    "Manipulation",
    "Immoral/Unethical",
    "Profanity",
    "Copyright/Trademark/Plagiarism"
]

# strip to be safe
severity_map = {cat.strip(): i + 1 for i, cat in enumerate(severity_order)}

def select_most_severe(categories):

    if isinstance(categories, str):
        categories = [categories]   # convert to list
    elif isinstance(categories, (list, tuple, set)):
        categories = list(categories)
    else:
        categories = []

    if not categories:
        return None
    
    valid = [c for c in categories if c in severity_map]
    
    if not valid:
        return None
    return min(valid, key=lambda c: severity_map[c])


def main() -> None:
    print("Loading Aegis training split…")
    dataset = load_dataset(
        "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train"
    )

    records: List[Dict[str, Any]] = [ex for ex in dataset]
    converted: List[Dict[str, Optional[str]]] = []
    total = len(records)
    print(f"Processing {total} prompts…")

    records = []

    for idx, row in enumerate(dataset):
        categories = row.get("violated_categories", [])

        # for simplicity keep only one category
        most_severe = select_most_severe(categories)
        prompt_text = row.get("prompt")
        response_text = row.get("response")

        # --------------------------------------------------
        # we keep rows with non null prompt_label_source
        # --------------------------------------------------
        prompt_label_source = row.get("prompt_label_source")
        if not prompt_label_source:
            prompt_label_source = None

        # --------------------------------------------------
        # we keep rows with non null response_label_source
        # --------------------------------------------------
        response_label_source = row.get("response_label_source")
        if not response_label_source:
            response_label_source = None

        # --------------------------------------
        # Apply filters BEFORE building the record
        # --------------------------------------

        if (prompt_label_source is None 
                or response_label_source is None 
                or response_text is None or prompt_text is None or most_severe is None):
            continue   # skip this row entirely

        print(f"row at idx: {idx} passed all filters.")
        record = {
            "id": row.get("id"),
            "reconstruction_id_if_redacted": row.get("reconstruction_id_if_redacted"),
            "prompt": row.get("prompt"),
            "response": row.get("response"),
            "prompt_label": row.get("prompt_label"),
            "response_label": row.get("response_label"),

            # original Aegis list
            "violated_categories": categories,

            # NEW: most severe category
            "most_severe_category": most_severe,

            "prompt_label_source": prompt_label_source,
            "response_label_source": response_label_source
        }

        records.append(record)
        print("records.len: ", len(records))

    # ==============================
    # Save Output
    # ==============================
    output_file = "data/aegis.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} records to {output_file}")

if __name__ == "__main__":
    main()
