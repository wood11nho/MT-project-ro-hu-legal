# Legal Domain Adaptation for Hungarian-Romanian Neural Machine Translation

## **Project Overview**

This project aims to improve **Neural Machine Translation (NMT)** performance for the **Hungarian–Romanian** language pair in the **legal domain**, using the **JRC-Acquis v3.0** corpus, one of the largest publicly available HU–RO legal parallel corpora.

The focus is on addressing **domain-specific challenges under limited parallel data conditions**, such as legal terminology errors, entity hallucination, and terminology mismatch.

---

## **Preprocessing**

Implemented in `scripts/prepare_splits.py`:

- duplicate sentence pair removal  
- empty / whitespace-only line removal  
- alignment noise filtering using source–target length ratio:  
  `0.5 ≤ |src_chars| / |tgt_chars| ≤ 2.0`  
- fixed sentence-level split: **80 / 10 / 10**

---

## **Models**

### **Multilingual Baseline**

- `facebook/nllb-200-distilled-600M` (zero-shot)

**Baseline performance:**
- BLEU: **8.26**
- COMET: **0.6183**

---

### **Bilingual Models**

- `Helsinki-NLP/opus-mt-hu-ro`  
- Marian-style Transformer sequence-to-sequence models

---

### **Pivot-Based Translation (HU → EN → RO)**

- `opus-mt-hu-en`  
- `opus-mt-en-ro`

Both stages are **adapted independently to the legal domain**.

---

## **Parameter-Efficient Fine-Tuning**

### **Method**

- **LoRA (Low-Rank Adaptation)** via Hugging Face **PEFT**

### **Configuration**

- target modules: `q_proj`, `v_proj`  
- rank (`r`): **8**  
- alpha: **16**  
- dropout: **0.05**

---

## **Training & Inference Setup**

- Framework: Hugging Face **Seq2SeqTrainer**  
- Precision: **FP16** when CUDA is available  

**Max sequence length:**
- 256 tokens (training)  
- 512 tokens (evaluation)  

**Decoding:**
- greedy (default)  
- beam search (beam = 4) for ablations  

- dynamic batch size reduction on OOM  
- checkpointing with retention of latest checkpoints only  

---

## **Evaluation**

### **Metrics**

- **BLEU** (SacreBLEU, standardized)  
- **COMET-DA** (`wmt22-comet-da`)  

### **Diagnostics**

- Hungarian diacritics leakage detection  
- rule-based legal terminology consistency checks  

Diagnostics are used **for analysis only**, not for model selection.

---

## **Observed Issues**

- failure to translate legally salient entities  
  *(e.g., “Bizottság” left untranslated)*  

- hallucinated institutions  
  *(e.g., “Investment Fund” → “Parliament”)*  

- source-language interference in output  

## Setup
1. `pip install -r requirements.txt`
2. `python scripts/download_data.py` (Downloads JRC-Acquis)
3. `python scripts/prepare_splits.py` (Creates Train/Val/Test splits)
4. `python scripts/evaluate_baseline.py` (Runs NLLB baseline)
