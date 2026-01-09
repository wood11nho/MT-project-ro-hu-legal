import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
from comet import download_model, load_from_checkpoint
from pathlib import Path
import logging
from tqdm import tqdm

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Robust Pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Language Codes
SRC_LANG = "hun_Latn"
TGT_LANG = "ron_Latn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logging.info(f"Loading test data from {TEST_FILE}...")
    try:
        # We limit to 100 for the 'Quick Test' tonight. 
        df = pd.read_csv(TEST_FILE).head(100)
    except FileNotFoundError:
        logging.error(f"Test file not found at {TEST_FILE}. Did you run prepare_splits.py?")
        return

    src_sentences = df["hu"].tolist()
    refs = df["ro"].tolist()

    # 2. Load Model & Tokenizer
    logging.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # 3. Translate
    logging.info("Translating...")
    hyps = []
    batch_size = 8 
    max_len = 512 

    for i in tqdm(range(0, len(src_sentences), batch_size)):
        batch = src_sentences[i:i+batch_size]
        
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_len
        ).to(DEVICE)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs, 
                # FIX: Use convert_tokens_to_ids instead of lang_code_to_id
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG), 
                max_length=max_len,
                num_beams=1, 
                early_stopping=True
            )
            
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        hyps.extend(decoded)

    # 4. Compute BLEU (SacreBLEU)
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    logging.info(f"--------------------------------------------------")
    logging.info(f"BASELINE BLEU SCORE: {bleu.score:.2f}")
    logging.info(f"--------------------------------------------------")

    # 5. Compute COMET
    logging.info("Downloading/Loading COMET model (this might take a moment)...")
    try:
        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)
        
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src_sentences, hyps, refs)]
        
        logging.info("Scoring with COMET...")
        # Reduce batch size for COMET on CPU to prevent freeze
        comet_score = comet_model.predict(data, batch_size=4, gpus=1 if torch.cuda.is_available() else 0)
        
        logging.info(f"--------------------------------------------------")
        logging.info(f"BASELINE COMET SCORE: {comet_score.system_score:.4f}")
        logging.info(f"--------------------------------------------------")
    except Exception as e:
        logging.error(f"COMET calculation failed: {e}")
        logging.warning("Continuing to save predictions despite COMET failure.")

    # 6. Save outputs
    output_path = OUTPUT_DIR / "baseline_predictions.csv"
    output_df = pd.DataFrame({"source": src_sentences, "reference": refs, "hypothesis": hyps})
    output_df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    evaluate()