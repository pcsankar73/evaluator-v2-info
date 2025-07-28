# Evaluator v2 — Multi‑Label Human‑Centric Decision Model

**Model ID:** `pcsankar73s/EvaluatorModel`  
**License:** CC BY‑NC 4.0 (non‑commercial; approval required for inference)  
**Access:** 🔒 Gated — visible to all, usable only with explicit approval  
**Author:** Sankar Palamadai @ Simple Machine Mind (https://www.smsquared.ai)

---

## 💡 Overview

Evaluator v2 is a BERT‑based, multi‑task, multi‑label reasoning model that predicts:

- **Decision** (Yes / No / TBD)  
- **Emotion** (multi‑label emotion states)  
- **Value** (moral/cognitive categories)  

Trained on 181 000 curated examples, it’s optimized for structured decision‑making with built‑in interpretability.

---

## 🧠 High‑Level Architecture

- **Backbone:** `bert-base-uncased` (12‑layer Transformer)  
- **Heads:** Three separate classification heads for Decision, Emotion, and Value  
- **Training recipe:** Gradual unfreeze → full unfreeze; LR=1e‑5; batch size=32; early stopping (patience=2); threshold‑sweep  

---

## 🧰 Data Processing Modules

| **Included for Further Progress**      | **Cited (for Reference/Citation)**                |
|----------------------------------------|---------------------------------------------------|
| `process_semeval2017_local`            | `process_sentiment140`                            |
| `process_financial_phrasebank`         | `process_imdb`                                    |
| `process_tweeteval`                    | `process_multinli`                                |
| `process_goemotions`                   | `process_tweeteval_health`                        |
|                                        | `process_normbank_csv_concatenated`               |
|                                        | `process_mft_from_json`                           |
|                                        | `process_meld`                                    |
|                                        | `process_empathetic_dialogues`                    |
|                                        | `process_social_bias_frames`                      |
|                                        | `process_ethics_local`                            |
|                                        | `process_ethics_virtue`                           |

> The right column lists commented‑out processors—use them as citations if you leverage those datasets later.

---

## 📚 Datasets

1. **Custom Reasoning Dataset** (181 000 samples) — in‑house decision/emotion/value annotations  
2. **GoEmotions** (Kumar et al., 2021) — 58 000 Reddit comments, 27 fine‑grained emotion labels  
3. **HUMSET** (AI4Good & UN OCHA, 2024) — 47 000 humanitarian crisis reports, multi‑label tags  

---

## 📊 Performance

| Task                     | Macro F1 (%) | Accuracy (%) |
|--------------------------|--------------|--------------|
| Decision Classification  | **80.24**    | **82.35**    |
| Emotion Recognition      | 27.8         | 46.1         |
| Value Prediction         | 37.9         | 62.3         |

> **Validation (Epoch 3):** Train Loss 0.2887 → Val Loss 0.4448  
> **Checkpoint:** `best_model_epoch_3.pt`  
> **HF export path:** `hf_format/`

---

## 🙅 Limitations

- **Domain Shift:** May underperform on slang, typos, long docs.  
- **Token Window:** Inputs truncated to 128 tokens—longer texts require chunking.  
- **Auxiliary Heads:** Emotion/Value heads (F1 27.8%/37.9%) are for interpretability, not high‑stakes decisions.  
- **No Live Feedback Loop:** Not yet fine‑tuned on production logs; periodic retraining needed.  
- **Human Oversight Required:** Avoid use in medical, legal, or safety‑critical domains without expert review.

---

## 🔑 Access Rights (Invite‑Only)

When granted access, a collaborator can:  
1. **Download & Load** the model artifacts (`pytorch_model.bin`, configs, tokenizer).  
2. **Run Inference** locally with Hugging Face Transformers.  
3. **View & Propose Changes** to README (if write access granted).

They **cannot** see private training scripts, raw data, or other repos without invitation, nor publish the model without approval.

---

## 🚀 Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("pcsankar73s/EvaluatorModel")
model = AutoModelForSequenceClassification.from_pretrained("pcsankar73s/EvaluatorModel")

inputs = tokenizer(
    "Should we proceed with this contract?",
    return_tensors="pt",
    max_length=128,
    truncation=True
)
logits = model(**inputs).logits
# logits[:,0] = decision head, logits[:,1] = emotion head, logits[:,2] = value head

@misc{palamadai2025evaluator,
  author       = {Sankar Palamadai},
  title        = {Evaluator v2: Multi‑Label Decision Model (F1_target_85)},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/pcsankar73s/EvaluatorModel}},
  note         = {Simple Machine Mind}
}

@inproceedings{goemotions2021,
  title   = {GoEmotions: A Dataset of 58k Fine‐Grained Emotion Labels},
  author  = {Kumar, S. and others},
  journal = {ACL Findings},
  year    = {2021},
  url     = {https://github.com/google-research/google-research/tree/master/goemotions}
}

@article{Srinivasan2024HUMSET,
  title   = {HUMSET: A Multi‐Label Dataset for Humanitarian Text Classification},
  author  = {Srinivasan, A. and others},
  journal = {AI for Good Journal},
  year    = {2024},
  url     = {https://github.com/AI4Good/HUMSET}
}

@inproceedings{hovy2014financialphrasebank,
  title     = {Financial PhraseBank, a Sentiment Analysis Dataset for the Financial Domain},
  author    = {Malo, P. and others},
  booktitle = {LREC},
  year      = {2014},
  url       = {https://www.researchgate.net/publication/261305405_Financial_PhraseBank_v1}
}

@article{barbieri2020tweeteval,
  title   = {TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification},
  author  = {Barbieri, F. and others},
  journal = {EMNLP Findings},
  year    = {2020},
  url     = {https://aclanthology.org/2020.findings-emnlp.112}
}

@misc{ethics_local2025,
  title     = {Ethics Local Dataset for Value and Virtue Analysis},
  author    = {Simple Machine Mind},
  year      = {2025},
  note      = {Internal ethics/value dataset}
}

@misc{ethics_virtue2025,
  title     = {Ethics Virtue Dataset for Multi‑Label Value Classification},
  author    = {Simple Machine Mind},
  year      = {2025},
  note      = {Internal virtue ethics dataset}
}

This repo is private—invite‑only.
For credentials or collaboration, please reach out at 📧 sankar@smsquared.ai
