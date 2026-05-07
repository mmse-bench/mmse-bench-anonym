import os
import json
import random
from collections import Counter

def create_batch_files(input_path, batch_dir, batch_size=50, seed=42):
    """
    Create non-overlapping batch files from the input dataset for scalable evaluation.

    This function supports batch-wise generation of model responses, enabling
    efficient LLM-based evaluation (e.g., GPT-4o) under cost and scalability constraints.
    The dataset is randomly shuffled (with a fixed seed for reproducibility) and
    partitioned into fixed-size subsets, ensuring fair usage of data across experiments.

    Specifically, we construct non-overlapping evaluation batches by randomly
    shuffling the dataset and dividing it into fixed-size subsets of 50 root prompts,
    ensuring no sample overlap across batches while preserving the original data
    distribution.

    Args:
        input_path (str): Path to the input dataset file.
        batch_dir (str): Directory where batch files will be saved.
        batch_size (int, optional): Number of samples per batch. Default is 50.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        None
    """

    # ==============================
    # ✅ Step 1: Load Data
    # ==============================
    with open(input_path, "r") as f:
        data = json.load(f)
    print("Total samples:", len(data))

    # ==============================
    # ✅ Step 2: Shuffle
    # ==============================
    random.seed(42)  # for reproducibility
    random.shuffle(data)

    # ==============================
    # ✅ Step 3: Create Batches (size = 50)
    # ==============================
    def create_batches(data, batch_size):
        batches = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        
        return batches

    batches = create_batches(data, batch_size)

    print("\nTotal batches created:", len(batches))
    print("Example batch size:", len(batches[0]))
    print("Last batch size:", len(batches[-1]))

    # ==============================
    # ✅ Step 4: Save Batches
    # ==============================
    os.makedirs(batch_dir, exist_ok=True)

    for i, batch in enumerate(batches):
        file_path = os.path.join(batch_dir, f"batch_{i+1}.json")
        with open(file_path, "w") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)
        
        print(f"Saved batch {i+1} → {file_path}")

    # ==============================
    # ✅ Step 5: Verify No Overlap
    # ==============================
    def check_no_overlap(batches):
        seen_ids = set()
        
        for i, batch in enumerate(batches):
            for item in batch:
                uid = item["id"]
                
                if uid in seen_ids:
                    print(f"❌ Overlap detected in batch {i+1}")
                    return
                seen_ids.add(uid)
        
        print("\n✅ No overlap across all batches!")

    check_no_overlap(batches)

    # ==============================
    # ✅ Step 6: Basic Batch Analysis
    # ==============================
    def batch_stats(batch, batch_id):
        categories = [item["most_severe_category"] for item in batch]
        dist = Counter(categories)
        
        print(f"\n📊 Batch {batch_id} category distribution:")
        for k, v in dist.items():
            print(f"{k}: {v}")

    # Show stats for first 3 batches
    for i in range(min(3, len(batches))):
        batch_stats(batches[i], i+1)

    # ==============================
    # ✅ Step 7: Overall Dataset Stats
    # ==============================
    def dataset_stats(data):
        categories = [item["most_severe_category"] for item in data]
        dist = Counter(categories)
        
        print("\n📊 Full Dataset Distribution:")
        for k, v in dist.items():
            print(f"{k}: {v}")

    dataset_stats(data)

def check_duplicate(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)

    ids = [item["id"] for item in data]

    unique_ids = set(ids)

    print("Total IDs:", len(ids))
    print("Unique IDs:", len(unique_ids))

    if len(ids) == len(unique_ids):
        print("✅ All IDs are unique")
    else:
        print("❌ Duplicate IDs found")
        
    from collections import Counter

    id_counts = Counter(ids)

    duplicates = {k: v for k, v in id_counts.items() if v > 1}

    total_samples = 0
    if not duplicates:
        print("✅ No duplicate IDs found")
    else:
        print(f"❌ Found {len(duplicates)} duplicate IDs:\n")
        
        for k, v in duplicates.items():
            print(f"ID: {k} → Count: {v}")
            total_samples += v
    print("total_samples: ", total_samples)

# keep the unique ids only & store in a new file
def keep_unique(input_path, output_path):
    import json

    with open(input_path, "r") as f:
        data = json.load(f)

    print(f"Original dataset size: {len(data)}")

    # ==============================
    # ✅ Deduplicate by ID
    # ==============================
    seen_ids = set()
    unique_data = []

    for item in data:
        uid = item.get("id")
        
        if uid not in seen_ids:
            unique_data.append(item)
            seen_ids.add(uid)

    print(f"Unique dataset size: {len(unique_data)}")
    print(f"Duplicates removed: {len(data) - len(unique_data)}")

    # ==============================
    # ✅ Save output
    # ==============================
    with open(output_path, "w") as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Deduplicated file saved at:\n{OUTPUT_PATH}")

if __name__ == "__main__":
    INPUT_PATH = "translation/aegis_multilang.json"
    UNIQUE_PATH = "translation/aegis_multilang_unique.json"
    BATCH_DIR = "/mmse-bench-anonym/translation/data/aegis-batches"

    keep_unique(INPUT_PATH, UNIQUE_PATH)
    create_batch_files(UNIQUE_PATH, BATCH_DIR, batch_size=50, seed=42)
    
