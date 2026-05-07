# MMSE: Massively Multilingual Safety Evaluation Benchmark

Code and data for the paper:
"MMSE: A Massively Multilingual Benchmark for Typology-Aware 
Multi-Metric Safety Evaluation of Language Models and Guardrails"

NeurIPS 2026 Evaluation and Benchmarks Track (Anonymous Submission)


MMSE evaluates the safety of large language models across 98 languages. Starting from a curated set of English safety prompts (sourced from the Aegis v2 dataset), the pipeline translates prompts into 100 languages, generates model responses using both open-source and closed-source LLMs, and evaluates those responses with LLM-as-judge scoring and guardrails classifiers.

---

## Repository structure

```
.
├── dataset-sourcing/           # Filter and deduplicate Aegis v2 prompts
│   ├── filter_aegis_dataset.py
│   ├── unique_data.py
│   ├── data/                   # Filtered output (5,376 records → 2,894 unique)
│   └── script.sh
│
├── translation/                # Translate prompts into 98 languages + quality gating
│   ├── translate.py
│   ├── quality_gating.py
│   ├── data/
│   │   ├── gold_dataset.jsonl             # ~6,542 records (highest quality)
│   │   ├── high_quality_dataset.jsonl   # 59,807 records
│   │   └── standard_quality_dataset.jsonl # 140,533 records
│   └── script.sh
│
├── response-generation/        # Generate model responses for the gold dataset
│   ├── open/                   # Open-source models (GPT-OSS, Aya, Mistral)
│   │   └── response_generator.py
│   ├── closed/                 # Closed-source models (GPT-5.4-mini, Gemini 2.5 Flash, Claude Haiku 4.5)
│   │   ├── build_request_batches.py
│   │   ├── run_multi_provider_batches.py
│   │   └── merge_responses.py
│   ├── data/                   # Per-model response files (3 open + 3 closed = 6 total)
│   └── script.sh
│
├── llm-as-judge/               # Score model responses with GPT and Claude judges
│   ├── judge.py
│   ├── aggregate.py            # Majority voting and average scoring across judges
│   ├── data/                   # 6 GPT-judge files + 6 Claude-judge files → 6 aggregated files
│   ├── llm_as_judge.sh
│   ├── aggregate.sh
│   └── script.sh
│
├── guardrails/                 # Apply safety classifiers to prompts and responses
│   ├── classifier.py           # Prompt classification + response classification (6 × 6,542)
│   ├── safetyjudge.py          # Prompt safety judgment + response judgment (6 × 6,542)
│   ├── classifier.sh
│   ├── safetyjudge.sh
│   └── script.sh
│
├── analysis-and-results/       # Aggregate all signals and produce final results
│   ├── aegis_dataset_analysis.py
│   ├── response_analysis.py
│   ├── judge_response_analysis.py
│   ├── guardrails_response_analysis.py
│   ├── combined_analysis.py
│   └── script.sh
│
├── requirements.txt
└── README.md
```

---

## Pipeline overview

```
Aegis v2 dataset
      │
      ▼
[dataset-sourcing]  →  2,894 unique prompts
      │
      ▼
[translation]       →  283,612 translated prompts across 98 languages
                        Quality gating → gold (6,542) / strict (~59K) / moderate (~140K) / low (~76k)
      │
      ▼ (gold dataset, 6,542 records)
[response-generation]  →  3 model response files (3 open + 3 closed)
      │
      ▼
[llm-as-judge]      →  GPT judge + Claude judge → aggregated scores (6 files)
      │
      ▼
[guardrails]        →  Classifier + SafetyJudge applied to all prompts and responses
      │
      ▼
[analysis-and-results]  →  Combined evaluation report
```

---

## Data flow

Each module reads from its own `data/` directory and writes enriched output there. The gold dataset (`translation/data/gold_dataset.jsonl`) is the primary input for all downstream modules. Fields are progressively added at each stage: model responses in `response-generation`, judge scores in `llm-as-judge`, and classifier labels in `guardrails`.

| Stage | Input records | Output records | Key output |
|---|---|---|---|
| dataset-sourcing | 5,376 | 2,894 | Deduplicated English prompts |
| translation | 2,830 | 283,612 (→ 6,542 gold) | Multilingual prompts with quality scores |
| response-generation | 6,542 | 6,542 × 6 models | Model responses per language |
| llm-as-judge | 6,542 × 6 | 6,542 × 6 (aggregated) | Safety scores |
| guardrails | 6,542 + 6,542 × 6 | Same + labels | Classifier and safety judge labels |
| analysis-and-results | All above | — | Evaluation metrics and figures |

---

## Setup

**Requirements**: Python 3.10+

```bash
pip install -r requirements.txt
```

Set the following environment variables before running any module:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export HF_TOKEN=...          # for open-source models via HuggingFace
```

---

## Reproducing the full pipeline

Each module has a `script.sh` that runs its steps in order. To reproduce end-to-end:

```bash
cd dataset-sourcing   && bash script.sh
cd ../translation     && bash script.sh
cd ../response-generation && bash script.sh
cd ../llm-as-judge    && bash script.sh
cd ../guardrails      && bash script.sh
cd ../analysis-and-results && bash script.sh
```

> **Note:** Translation and response generation are compute-intensive. We recommend running them on a GPU cluster or using the batch API endpoints already integrated in `closed/run_multi_provider_batches.py`.

---

## Models evaluated

**Open-source** (University Model Cluster): GPT-OSS, Aya, Mistral

**Closed-source** (via API): GPT-5.4-mini, Gemini 2.5 Flash, Claude Haiku 4.5

**LLM judges**: GPT 5.5 (OpenAI), Claude Sonnet 4.6 (Anthropic)

**Guardrails**: LLM-based binary classifier (`classifier.py`), LLM-based safety judge (`safetyjudge.py`)

---

## Guardrails module note

The `guardrails/` module applies two independent evaluation tools to the same set of files:

- **`classifier.py`** — a LLM-based fine-tuned binary classifier that labels each prompt and response as safe or unsafe.
- **`safetyjudge.py`** — an LLM-based safety judge that produces fine-grained safety labels with reasoning.

These are complementary and are both applied to all 9 model response files (6 × 6,542 responses) as well as the 6,542 prompts directly.

---


## Dataset
Available at: https://huggingface.co/datasets/mmse-bench-anon/mmse-bench-anon

