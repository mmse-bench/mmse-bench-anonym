#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Provider SDKs
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types as genai_types

# Tokenizer fallback for approximate counts
import tiktoken


# =========================================================
# 🔷 GLOBALS
# =========================================================
ENCODING = tiktoken.get_encoding("cl100k_base")


# =========================================================
# 🔷 HELPERS
# =========================================================

def make_serializable(obj):
    """Recursively convert non-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Fallback — convert unknown objects to string representation
        try:
            return obj.model_dump()   # pydantic objects (OpenAI SDK)
        except AttributeError:
            return str(obj)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def approx_token_count(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    try:
        return len(ENCODING.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def load_processed_ids(output_file: Path) -> set:
    processed = set()
    if not output_file.exists():
        return processed
    for row in load_jsonl(output_file):
        cid = row.get("custom_id")
        if cid:
            processed.add(cid)
    print(f"🔁 Resume loaded: {len(processed)} completed requests from {output_file}")
    return processed


def model_tag(model_name: str) -> str:
    return model_name.lower().replace("/", "__").replace("-", "_").replace(":", "_")


def extract_openai_usage(resp: Any, prompt: str, output_text: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    Try provider-reported usage first, fallback to approximation.
    """
    usage = getattr(resp, "usage", None)
    if usage is not None:
        in_tok = getattr(usage, "input_tokens", None)
        out_tok = getattr(usage, "output_tokens", None)
        if in_tok is not None or out_tok is not None:
            return in_tok, out_tok, "provider_reported_or_partial"
    return approx_token_count(prompt), approx_token_count(output_text), "tiktoken_cl100k_fallback"


def extract_anthropic_usage(resp: Any, prompt: str, output_text: str) -> Tuple[Optional[int], Optional[int], str]:
    usage = getattr(resp, "usage", None)
    if usage is not None:
        in_tok = getattr(usage, "input_tokens", None)
        out_tok = getattr(usage, "output_tokens", None)
        if in_tok is not None or out_tok is not None:
            return in_tok, out_tok, "provider_reported_or_partial"
    return approx_token_count(prompt), approx_token_count(output_text), "tiktoken_cl100k_fallback"


def extract_gemini_usage(resp: Any, prompt: str, output_text: str) -> Tuple[Optional[int], Optional[int], str]:
    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        in_tok = getattr(usage, "prompt_token_count", None)
        out_tok = getattr(usage, "candidates_token_count", None)
        if in_tok is not None or out_tok is not None:
            return in_tok, out_tok, "provider_reported_or_partial"
    return approx_token_count(prompt), approx_token_count(output_text), "tiktoken_cl100k_fallback"


# =========================================================
# 🔷 PROVIDER CLIENTS
# =========================================================
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    print("api_key: ", api_key)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=api_key)


def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
    return genai.Client(api_key=api_key)


# =========================================================
# 🔷 GENERATION FUNCTIONS
# =========================================================
def generate_openai(request_obj: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    prompt = request_obj["prompt"]
    cfg = request_obj["generation_config"]
    system_instruction = request_obj["system_instruction"]

    start = time.time()
    resp = client.responses.create(
        model=request_obj["model"],
        input=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        temperature=cfg["temperature"],
        max_output_tokens=cfg["max_tokens"],
        # seed=cfg.get("seed"),
    )
    print("resp: ", resp)
    latency = time.time() - start

    output_text = getattr(resp, "output_text", None)
    if not output_text:
        try:
            output_text = resp.output[0].content[0].text
        except Exception:
            output_text = ""

    in_tok, out_tok, token_method = extract_openai_usage(resp, prompt, output_text)

    return {
        "provider_response_id": getattr(resp, "id", None),
        "response": output_text,
        "response_raw": make_serializable(resp),
        "prompt_length_tokens": in_tok,
        "response_length_tokens": out_tok,
        "token_count_method": token_method,
        "time_per_sample_sec": latency,
        "finish_reason": None,
        "error": None,
    }


def generate_anthropic(request_obj: Dict[str, Any], client: anthropic.Anthropic) -> Dict[str, Any]:
    print("generate_anthropic called with request_obj: ", request_obj)
    prompt = request_obj["prompt"]
    cfg = request_obj["generation_config"]
    system_instruction = request_obj["system_instruction"]

    start = time.time()
    resp = client.messages.create(
        model=request_obj["model"],
        system=system_instruction,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    latency = time.time() - start

    chunks = []
    for block in getattr(resp, "content", []):
        txt = getattr(block, "text", None)
        if txt:
            chunks.append(txt)
    output_text = "".join(chunks)

    in_tok, out_tok, token_method = extract_anthropic_usage(resp, prompt, output_text)

    return {
        "provider_response_id": getattr(resp, "id", None),
        "response": output_text,
        "response_raw": make_serializable(resp),
        "prompt_length_tokens": in_tok,
        "response_length_tokens": out_tok,
        "token_count_method": token_method,
        "time_per_sample_sec": latency,
        "finish_reason": getattr(resp, "stop_reason", None),
        "error": None,
    }


def generate_gemini(request_obj: Dict[str, Any], client: genai.Client) -> Dict[str, Any]:
    prompt = request_obj["prompt"]
    cfg = request_obj["generation_config"]
    system_instruction = request_obj["system_instruction"]

    start = time.time()
    resp = client.models.generate_content(
        model=request_obj["model"],
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=cfg["temperature"],
            max_output_tokens=cfg["max_tokens"],
            top_p=cfg.get("top_p"),
            seed=cfg.get("seed"),
        ),
    )
    latency = time.time() - start

    output_text = getattr(resp, "text", "") or ""

    in_tok, out_tok, token_method = extract_gemini_usage(resp, prompt, output_text)
    print("Gemini Usage: resp=", resp)

    print("Gemini Usage: in_tok=", in_tok, " out_tok=", out_tok, " method=", token_method)

    finish_reason = None
    try:
        cands = getattr(resp, "candidates", None)
        if cands and len(cands) > 0:
            finish_reason = getattr(cands[0], "finish_reason", None)
    except Exception:
        pass

    return {
        "provider_response_id": None,
        "response": output_text,
        "prompt_length_tokens": in_tok,
        "response_length_tokens": out_tok,
        "token_count_method": token_method,
        "time_per_sample_sec": latency,
        "finish_reason": finish_reason,
        "error": None,
    }


def route_generate(request_obj: Dict[str, Any], clients: Dict[str, Any]) -> Dict[str, Any]:
    provider = request_obj["provider"]

    try:
        if provider == "openai":
            return generate_openai(request_obj, clients["openai"])
        elif provider == "anthropic":
            return generate_anthropic(request_obj, clients["anthropic"])
        elif provider == "gemini":
            return generate_gemini(request_obj, clients["gemini"])
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        return {
            "provider_response_id": None,
            "response": None,
            "prompt_length_tokens": None,
            "response_length_tokens": None,
            "token_count_method": None,
            "time_per_sample_sec": None,
            "finish_reason": None,
            "error": str(e),
        }


# =========================================================
# 🔷 BATCH RUNNER
# =========================================================
def process_request_file(request_file: Path, output_file: Path, clients: Dict[str, Any]) -> None:
    print(f"\n🚀 Processing request file: {request_file}")
    requests = load_jsonl(request_file)
    processed = load_processed_ids(output_file)

    pending = [r for r in requests if r["custom_id"] not in processed]
    print(f"Remaining requests in this file: {len(pending)}")

    if not pending:
        print("✅ Nothing to do.")
        return

    file_start = time.time()
    results_buffer: List[Dict[str, Any]] = []

    for idx, req in enumerate(pending, 1):
        gen = route_generate(req, clients)
        print("gen: ", gen)
        print("response_raw: ", gen.get("response_raw"))

        out_row = {
            "custom_id": req["custom_id"],
            "provider": req["provider"],
            "model": req["model"],
            "root_id": req["root_id"],
            "language": req["language"],
            "prompt": req["prompt"],
            "system_instruction": req["system_instruction"],
            "generation_config": req["generation_config"],
            "eval": req.get("eval", {}),
            "response": gen["response"],
            "response_raw": gen.get("response_raw"),
            "provider_response_id": gen["provider_response_id"],
            "prompt_length_tokens": gen["prompt_length_tokens"],
            "response_length_tokens": gen["response_length_tokens"],
            "token_count_method": gen["token_count_method"],
            "time_per_sample_sec": gen["time_per_sample_sec"],
            "finish_reason": gen["finish_reason"],
            "error": gen["error"],
            "source_request_file": str(request_file),
        }
        results_buffer.append(out_row)

        if len(results_buffer) >= 25:
            append_jsonl(output_file, results_buffer)
            print(f"💾 Appended {len(results_buffer)} rows to {output_file.name} [{idx}/{len(pending)}]")
            results_buffer = []

    if results_buffer:
        append_jsonl(output_file, results_buffer)

    total_elapsed = time.time() - file_start
    print(f"✅ Finished {request_file.name} in {total_elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-dir", required=True, help="Directory containing provider/model request JSONL files")
    parser.add_argument("--output-dir", required=True, help="Directory to write provider outputs")
    args = parser.parse_args()

    request_dir = Path(args.request_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    request_files = sorted(request_dir.glob("*.jsonl"))
    print("request_files: ", request_files)
    if not request_files:
        raise FileNotFoundError(f"No request files found in {request_dir}")

    provider = None
    if len(request_files) > 0:
        # infer provider from first line of first file
        first_row = load_jsonl(request_files[0])[0]
        provider = first_row["provider"]

    clients: Dict[str, Any] = {}
    if provider == "openai":
        print("if provider == openai:")
        clients["openai"] = get_openai_client()
    elif provider == "anthropic":
        print("if provider == anthropic:")
        clients["anthropic"] = get_anthropic_client()
    elif provider == "gemini":
        print("if provider == gemini:")
        clients["gemini"] = get_gemini_client()
    else:
        raise ValueError(f"Could not infer supported provider from request files in {request_dir}")

    for request_file in request_files:
        print("processing file: ", request_file)
        out_file = output_dir / request_file.name
        process_request_file(request_file, out_file, clients)


if __name__ == "__main__":
    main()
