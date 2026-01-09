import os
import requests
import zipfile
from pathlib import Path
import logging

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FIX 1: Robust Pathing
# This ensures 'data' is always found in the project root, no matter where you run the script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# FIX 2: Direct Object Storage URL
# The previous URL was a PHP wrapper that often blocks scripts or redirects. 
# This is the direct link to the file on the OPUS cloud storage.
# Note: OPUS sorts languages alphabetically, so 'hu' comes before 'ro'.
OPUS_DIRECT_URL = "https://object.pouta.csc.fi/OPUS-JRC-Acquis/v3.0/moses/hu-ro.txt.zip"

def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "hu-ro.txt.zip"
    
    # Check if exists to avoid redownloading
    if not zip_path.exists():
        logging.info(f"Downloading corpus from {OPUS_DIRECT_URL}...")
        try:
            # We add a User-Agent just in case, though the direct link usually doesn't need it.
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(OPUS_DIRECT_URL, stream=True, headers=headers)
            response.raise_for_status() # Raise error if download fails (404, 500, etc.)
            
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info("Download complete.")
            
        except Exception as e:
            logging.error(f"Failed to download: {e}")
            return # Stop here if download failed
    else:
        logging.info("Zip file already exists.")

    # Validation before unzipping
    if not zipfile.is_zipfile(zip_path):
        logging.error("The downloaded file is not a valid zip file. It might be a corrupt download or an HTML page.")
        return

    logging.info("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        logging.info(f"Data successfully ready in {DATA_DIR}")
        
        # Verify extraction
        extracted_files = list(DATA_DIR.glob("*"))
        logging.info(f"Files in data folder: {[f.name for f in extracted_files]}")
        
    except zipfile.BadZipFile:
        logging.error("Critical Error: The file is corrupted. Please delete it and try again.")

if __name__ == "__main__":
    download_and_extract()