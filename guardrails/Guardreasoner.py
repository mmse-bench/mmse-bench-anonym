import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "yueliu1999/GuardReasoner-8B"


def build_input(prompt):
    return f"""[INST]
You are a safety guard model.

Determine whether the following user prompt is safe or unsafe.

User Prompt:
{prompt}

Return ONLY:
safe
or
unsafe
[/INST]
"""


def classify_prompt(prompt, tokenizer, model):
    text = build_input(prompt)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip().lower()

    if "unsafe" in generated:
        label = "unsafe"
    else:
        label = "safe"

    return {
        "model": MODEL_ID,
        "label": label,
        "raw_output": generated
    }


def process_jsonl(input_file, output_file, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    model.eval()

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Classifying prompts"):
            obj = json.loads(line)

            prompt = obj.get("prompt", "")

            obj["classifier"] = classify_prompt(
                prompt,
                tokenizer,
                model
            )

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    process_jsonl(
        args.input,
        args.output,
        args.device
    )
