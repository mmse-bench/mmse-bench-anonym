# llama_guard_permissive_prompt_classifier.py

import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def build_conversation(prompt):
    return [{"role": "user", "content": prompt}]


def apply_chat_template(conversation):
    prompt = "[INST] Safety classification:\n\n"

    for turn in conversation:
        prompt += f"{turn['role']}: {turn['content']}\n"

    prompt += "\nReturn 'safe' or 'unsafe'. [/INST]"
    return prompt


def classify(prompt, tokenizer, model, max_len, safe_id, unsafe_id):
    conversation = build_conversation(prompt)
    text = apply_chat_template(conversation)

    inputs = tokenizer(
        text,
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True
    )

    logits = output.scores[0][:, [safe_id, unsafe_id]]
    probs = torch.softmax(logits, dim=-1)[0]

    safe_prob = probs[0].item()
    unsafe_prob = probs[1].item()

    label = "unsafe" if unsafe_prob >= safe_prob else "safe"

    return {
        "model": "LlamaGuard-Permissive",
        "label": label,
        "safe_probability": safe_prob,
        "unsafe_probability": unsafe_prob
    }


def process(input_file, output_file, device):
    model_id = "meta-llama/LlamaGuard-7b"
    adapter_id = "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0"

    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )

    model = PeftModel.from_pretrained(base_model, adapter_id).to(device)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    safe_id = tokenizer.encode("safe")[1]
    unsafe_id = tokenizer.encode("unsafe")[1]

    with open(input_file) as fin, open(output_file, "w") as fout:
        for line in tqdm(fin):
            obj = json.loads(line)
            prompt = obj.get("prompt", "")

            with torch.no_grad():
                obj["classifier"] = classify(
                    prompt, tokenizer, model,
                    config.max_position_embeddings,
                    safe_id, unsafe_id
                )

            fout.write(json.dumps(obj) + "\n")
            fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    process(args.input, args.output, args.device)
