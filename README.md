# Legal Domain Adaptation for Hungarian-Romanian Neural Machine Translation

## Project Overview
This project aims to improve Neural Machine Translation (NMT) performance for the Hungarian-Romanian language pair in the legal domain (JRC-Acquis corpus). We finetune the NLLB-200 model to address specific challenges such as entity hallucination and terminology mismatch.

## Current Status (Baseline)
We have established a zero-shot baseline using `facebook/nllb-200-distilled-600M`.
* **BLEU:** 8.26
* **COMET:** 0.6183

**Identified Issues in Baseline:**
* Failure to translate key legal entities (e.g., "Bizottság" left as is).
* Hallucination of institutions (e.g., "Investment Fund" -> "Parliament").

## Setup
1. `pip install -r requirements.txt`
2. `python scripts/download_data.py` (Downloads JRC-Acquis)
3. `python scripts/prepare_splits.py` (Creates Train/Val/Test splits)
4. `python scripts/evaluate_baseline.py` (Runs NLLB baseline)