import os
import sys
import time
import logging
import warnings
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Translation and Metrics libraries
import sacrebleu
from comet import download_model, load_from_checkpoint
from groq import Groq, RateLimitError, APIError, BadRequestError, AuthenticationError

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Filter out third-party library warnings (e.g., torchmetrics/pkg_resources)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Robust Pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

# Model Configuration
# UPDATED: Using Llama 3.3 70B (Versatile) as the previous 70B model was decommissioned.
# This model offers state-of-the-art performance for complex translation tasks.
MODEL_NAME = "llama-3.3-70b-versatile"
SRC_LANG = "Hungarian"
TGT_LANG = "Romanian"
TEMPERATURE = 0.1  # Low temperature for deterministic, faithful translations

# ------------------------------------------------------------------------------
# Core Translation Logic
# ------------------------------------------------------------------------------

def get_groq_client():
    """Initializes and returns the Groq client using environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
        sys.exit(1)
    return Groq(api_key=api_key)

def build_prompt(source_text):
    """
    Constructs a zero-shot prompt optimized for legal fidelity.
    We enforce a strict output format to avoid 'chatty' LLM behavior.
    """
    system_message = (
        f"You are a specialized legal translator for the European Union. "
        f"Translate the following {SRC_LANG} legislative text into {TGT_LANG}. "
        "Do not provide explanations, notes, or headers. "
        "Output ONLY the translated text."
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": source_text}
    ]

def translate_batch(client, texts, model, delay=0.5):
    """
    Translates a list of texts sequentially with robust error handling.
    """
    hypotheses = []
    
    for text in tqdm(texts, desc=f"Translating with {model}"):
        retries = 3
        translation = ""
        
        while retries > 0:
            try:
                chat_completion = client.chat.completions.create(
                    messages=build_prompt(text),
                    model=model,
                    temperature=TEMPERATURE,
                    max_tokens=512,
                    top_p=1,
                    stop=None,
                    stream=False,
                )
                translation = chat_completion.choices[0].message.content.strip()
                break  # Success, exit retry loop
            
            except RateLimitError as e:
                wait_time = 5
                logging.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                retries -= 1
                
            except (BadRequestError, AuthenticationError) as e:
                # Critical errors that retrying won't fix (e.g., Wrong Model Name, Bad API Key)
                logging.error(f"CRITICAL API ERROR: {e}")
                logging.error("Stopping execution to prevent further failures.")
                sys.exit(1)
                
            except APIError as e:
                # Generic API errors (server side issues)
                logging.error(f"Groq API Error: {e}")
                retries -= 1
                if retries == 0:
                    logging.error(f"Failed to translate segment: {text[:30]}...")
                    
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                retries -= 1
        
        # If translation failed after retries, append empty string to keep alignment
        hypotheses.append(translation)
        
        # Brief pause to be a polite API citizen
        time.sleep(delay)
        
    return hypotheses

# ------------------------------------------------------------------------------
# Main Evaluation Pipeline
# ------------------------------------------------------------------------------

def evaluate():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    logging.info(f"Loading test data from {TEST_FILE}...")
    try:
        # Limit to 100 samples for consistency with baseline
        df = pd.read_csv(TEST_FILE).head(100)
    except FileNotFoundError:
        logging.error(f"Test file not found at {TEST_FILE}. Ensure prepare_splits.py has been executed.")
        return

    src_sentences = df["hu"].tolist()
    refs = df["ro"].tolist()

    # 2. Initialize Client
    logging.info(f"Initializing Groq client for model: {MODEL_NAME}")
    client = get_groq_client()

    # 3. Translate
    logging.info("Starting translation process...")
    hyps = translate_batch(client, src_sentences, MODEL_NAME)

    # Validation: Ensure we have exactly as many predictions as inputs
    if len(hyps) != len(refs):
        logging.error(f"Mismatch in counts: {len(hyps)} hypotheses vs {len(refs)} references.")
        return

    # 4. Compute BLEU (SacreBLEU)
    logging.info("Computing BLEU score...")
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    logging.info(f"--------------------------------------------------")
    logging.info(f"LLM ({MODEL_NAME}) BLEU SCORE: {bleu.score:.2f}")
    logging.info(f"--------------------------------------------------")

    # 5. Compute COMET
    logging.info("Downloading/Loading COMET model...")
    try:
        comet_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_path)
        
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src_sentences, hyps, refs)]
        
        logging.info("Scoring with COMET...")
        # Automatically detect GPU for COMET
        gpus = 1 if torch.cuda.is_available() else 0
        comet_score = comet_model.predict(data, batch_size=4, gpus=gpus)
        
        logging.info(f"--------------------------------------------------")
        logging.info(f"LLM ({MODEL_NAME}) COMET SCORE: {comet_score.system_score:.4f}")
        logging.info(f"--------------------------------------------------")
        
    except Exception as e:
        logging.error(f"COMET calculation failed: {e}")
        logging.warning("Continuing to save predictions despite COMET failure.")

    # 6. Save outputs
    # Sanitize model name for filename (replace - with _)
    safe_model_name = MODEL_NAME.replace("-", "_")
    output_filename = f"llm_groq_predictions_{safe_model_name}.csv"
    output_path = OUTPUT_DIR / output_filename
    
    output_df = pd.DataFrame({
        "source": src_sentences, 
        "reference": refs, 
        "hypothesis": hyps
    })
    output_df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    evaluate()