# I was able to run the below guard models, tried other 10+ but was getting issues
# xguard classifier
python xguard.py \
  --input aya_expanse:32b_merged.jsonl \
  --output xguard_judge_output.jsonl 

# guardreasoner classifier
python guardreasoner.py \
  --input aya_expanse:32b_merged.jsonl \
  --output guardreasoner_judge_output.jsonl 

# llamaguard permissive
python llamaguard_permissive.py \
  --input aya_expanse:32b_merged.jsonl \
  --output llamaguard_permissive_judge_output.jsonl 

# MD Judge classifier
python mdjudge.py \
  --input aya_expanse:32b_merged.jsonl \
  --output mdjudge_judge_output.jsonl 

# nemotron guard classifier
python run_nemotron_guard_official.py \
  --input_file llama3.3:70b_merged.jsonl \
  --output_file prompt_classified_nemotron_official.jsonl \
  --model_name nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 \
  --max_new_tokens 100
