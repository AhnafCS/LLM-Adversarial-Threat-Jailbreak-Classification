import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

HF_TOKEN = os.environ.get("HF_TOKEN", None)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLASSES = {
    "harmful_behavior": 4000,
    "prompt_injection": 4000,
    "toxicity": 4000,
    "linguistic": 4000,
    "jailbreak": 4000
}

collected_data = {cls: [] for cls in TARGET_CLASSES}
seen_prompts = set()

print("Streaming dataset from Hugging Face...")
dataset_stream = load_dataset(
    "Necent/llm-jailbreak-prompt-injection-dataset",
    token=HF_TOKEN,
    streaming=True
)

total_needed = sum(TARGET_CLASSES.values())
scanned = 0

for row in dataset_stream["train"]:
    scanned += 1
    lang = row.get("language", "")
    pt = row.get("prompt_type", "")
    prompt = row.get("prompt", "")
    
    # Filter conditions: English, recognized target class, non-empty, deduplicated
    if lang == "en" and pt in TARGET_CLASSES and prompt:
        prompt_clean = prompt.strip()
        if len(prompt_clean) > 10 and prompt_clean not in seen_prompts:
            if len(collected_data[pt]) < TARGET_CLASSES[pt]:
                seen_prompts.add(prompt_clean)
                collected_data[pt].append({
                    "prompt": prompt_clean,
                    "label": pt,
                    "source": row.get("source", "Unknown"),
                    "prompt_harmful": int(row.get("prompt_harmful", 1) or 1),
                    "prompt_adversarial": int(row.get("prompt_adversarial", 1) or 1)
                })
    
    current_total = sum(len(v) for v in collected_data.values())
    if scanned % 10000 == 0:
        print(f"Scanned {scanned} rows | Collected: {current_total}/{total_needed}")
        for cls, items in collected_data.items():
            print(f"  {cls}: {len(items)}/{TARGET_CLASSES[cls]}")
            
    if current_total >= total_needed:
        break

# Combine into single DataFrame
all_rows = []
for cls, items in collected_data.items():
    all_rows.extend(items)

df = pd.DataFrame(all_rows)
# Shuffle rows
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"\nFinal Dataset Shape: {df.shape}")
print("\nClass Distribution:")
print(df["label"].value_counts())

csv_path = os.path.join(OUTPUT_DIR, "llm_jailbreak_safety_20k.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"Saved balanced dataset to: {csv_path}")

# Create Train (70%), Val (15%), Test (15%) splits
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])

print(f"\nSplit Sizes:")
print(f"  Train Set: {train_df.shape[0]} rows (70%)")
print(f"  Val Set:   {val_df.shape[0]} rows (15%)")
print(f"  Test Set:  {test_df.shape[0]} rows (15%)")

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, encoding="utf-8")
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False, encoding="utf-8")
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, encoding="utf-8")
print("Saved train.csv, val.csv, test.csv successfully!")
