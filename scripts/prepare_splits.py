import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FIX 1: Robust Pathing (Works from root or scripts/ folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" 
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_parallel_corpus(src_lang, tgt_lang):
    # FIX 2: Removed "JRC-Acquis" from path, as files are directly in data/raw
    # We use glob to find files ending in .hu and .ro
    try:
        src_path = list(RAW_DIR.glob(f"*.{src_lang}"))[0]
        tgt_path = list(RAW_DIR.glob(f"*.{tgt_lang}"))[0]
    except IndexError:
        logging.error(f"Could not find .{src_lang} or .{tgt_lang} files in {RAW_DIR}")
        raise FileNotFoundError("Check if download_data.py ran successfully.")

    logging.info(f"Loading {src_path.name} and {tgt_path.name}...")
    
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_lines = [line.strip() for line in f]
        
    assert len(src_lines) == len(tgt_lines), "Source and Target lengths do not match!"
    return pd.DataFrame({"hu": src_lines, "ro": tgt_lines})

def prepare_splits():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_parallel_corpus("hu", "ro")
    initial_count = len(df)
    
    # 1. Deduplicate
    df.drop_duplicates(inplace=True)
    
    # 2. Advanced Cleaning (Remove empty or whitespace-only lines)
    # This is critical for NLLB, which can hallucinate on empty inputs
    df = df[df["hu"].str.strip().astype(bool)]
    df = df[df["ro"].str.strip().astype(bool)]
    
    cleaned_count = len(df)
    logging.info(f"Original: {initial_count} | Cleaned: {cleaned_count} | Removed: {initial_count - cleaned_count}")
    
    # 3. Split
    # 80% Train, 10% Validation, 10% Test
    # random_state=42 guarantees your team gets the exact same split
    train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)
    
    logging.info(f"Split Sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 4. Save
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    
    logging.info(f"SUCCESS! Splits saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    prepare_splits()