# xguard_prompt_classifier.py

import argparse
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "saillab/x-guard"


def parse_label(text):
    t = text.lower()

    if re.search(r"\bunsafe\b", t):
        return "unsafe"
    if re.search(r"\bsafe\b", t):
        return "safe"

    return "unknown"


def classify_prompt(prompt, tokenizer, model):
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": "<USER TEXT STARTS>\n" + prompt + "\n<USER TEXT ENDS>"
        },
        {
            "role": "assistant",
            "content": "\n <think>"
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=1e-7,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

    raw_output = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    return {
        "model": MODEL_ID,
        "label": parse_label(raw_output),
        "raw_output": raw_output
    }


def process_jsonl(input_file, output_file):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )

    model.eval()

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Classifying prompts"):
            if not line.strip():
                continue

            obj = json.loads(line)
            prompt = obj.get("prompt", "")

            try:
                obj["classifier"] = classify_prompt(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    model=model
                )

            except BaseException as e:
                obj["classifier"] = {
                    "model": MODEL_ID,
                    "label": "error",
                    "error_type": type(e).__name__,
                    "error": repr(e)
                }

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    process_jsonl(args.input, args.output)
