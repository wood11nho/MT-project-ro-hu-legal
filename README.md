# Legal Domain Adaptation for Hungarian-Romanian Neural Machine Translation

## **Project Overview**

This project aims to improve **Neural Machine Translation (NMT)** performance for the **Hungarian–Romanian** language pair in the **legal domain**, using the **JRC-Acquis v3.0** corpus.

The focus is on comparing **traditional encoder-decoder models** against **modern Generative LLMs** and addressing domain-specific challenges like terminology mismatch under limited resource conditions.

---

## **Key Results**

We achieved a **4.5x improvement** over the multilingual baseline.

| Model | Type | BLEU | COMET |
| --- | --- | --- | --- |
| **NLLB-200** | Multilingual Baseline | 8.26 | 0.6183 |
| **Llama 3.3 70B** | Zero-Shot LLM | 21.54 | **0.8842** |
| **Pivot + LoRA** | **Fine-Tuned + Glossary** | **38.10** | 0.8797 |

---

## **Preprocessing**

Implemented in `scripts/prepare_splits.py`:

* duplicate sentence pair removal
* empty / whitespace-only line removal
* alignment noise filtering using source–target length ratio:
`0.5 ≤ |src_chars| / |tgt_chars| ≤ 2.0`
* fixed sentence-level split: **80 / 10 / 10**

---

## **Models**

### **1. Multilingual Baseline**

* `facebook/nllb-200-distilled-600M`
* Zero-shot inference

### **2. Generative LLM (Zero-Shot)**

* **Model:** `llama-3.3-70b-versatile` (via Groq API)
* **Strategy:** System prompt engineering for legal specialization + deterministic decoding (temp=0.1).

### **3. Pivot-Based Translation (HU → EN → RO)**

* **Pipeline:** `opus-mt-hu-en` → `opus-mt-en-ro`
* **Adaptation:** Both stages fine-tuned independently on JRC-Acquis.
* **Inference:** Augmented with **glossary-based constrained decoding** to enforce strict legal terminology.

---

## **Parameter-Efficient Fine-Tuning**

### **Method**

* **LoRA (Low-Rank Adaptation)** via Hugging Face **PEFT**

### **Configuration**

* target modules: `q_proj`, `v_proj`
* rank (`r`): **8**
* alpha: **16**
* dropout: **0.05**

---

## **Training & Inference Setup**

* Framework: Hugging Face **Seq2SeqTrainer**
* Precision: **FP16** (CUDA)

**Max sequence length:**

* 256 tokens (training)
* 512 tokens (evaluation)

**Decoding:**

* greedy (default)
* **Constrained Beam Search** (for Pivot + Glossary)

---

## **Evaluation**

### **Metrics**

* **BLEU** (SacreBLEU) – Structural accuracy
* **COMET-DA** (`wmt22-comet-da`) – Semantic similarity

### **Diagnostics**

* Source language leakage detection (HU/EN retention)
* Rule-based legal terminology consistency checks

---

## **Observed Issues**

- failure to translate legally salient entities  
  *(e.g., “Bizottság” left untranslated)*  

- hallucinated institutions  
  *(e.g., “Investment Fund” → “Parliament”)*  

- source-language interference in output

---

## **Visualizations**

Automated analysis scripts generated:

* Sentence length distribution (Density plots)
* Terminology frequency analysis
* Source-Target length correlation checks

---

## Baseline Setup
1. `pip install -r requirements.txt`
2. `python scripts/download_data.py` (Downloads JRC-Acquis)
3. `python scripts/prepare_splits.py` (Creates Train/Val/Test splits)
4. `python scripts/evaluate_baseline.py` (Runs NLLB baseline)
