# 🛡️ LLM Adversarial Threat & Jailbreak Classification
### **CSE440: Natural Language Processing II — Research Project**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Ready-FF4B4B.svg)](https://streamlit.io/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)

---

## 1. Project Overview & Motivation

Large Language Models (LLMs) deployed in production applications are susceptible to adversarial prompt manipulation techniques designed to bypass safety guardrails. This research project constructs a complete multi-class NLP classification pipeline to detect and triage adversarial prompts across **5 core vulnerability classes**:

1. **Jailbreak (🟠):** Persona adoption and hypothetical roleplaying exploits designed to circumvent safety guardrails (e.g., DAN, fictional framing).
2. **Prompt Injection (🔴):** System instruction overrides and context hijacking aimed at extracting internal directives or executing unauthorized behavior.
3. **Harmful Behavior (🟣):** Direct solicitations for hazardous, illegal, or destructive instructions (e.g., cyberattacks, weapons, synthesis).
4. **Toxicity (🔴):** Explicit hate speech, harassment, slurs, profanity, and derogatory insults.
5. **Linguistic Evasion (🔵):** Subtle social engineering, linguistic euphemisms, and syntactic obfuscation.

---

## 2. Dataset Specifications

* **Source:** Curated and deduplicated from `Necent/llm-jailbreak-prompt-injection-dataset` (aggregating RedBench, SPML, SGToxicGuard, Do-Not-Answer, and AdvBench).
* **Volume:** 20,000 balanced English prompts (**4,000 samples per class**).
* **Partitions:** Stratified 70% Training (14,000), 15% Validation (3,000), and 15% Held-Out Testing (3,000).

| Threat Class | Samples | Avg Char Length | Avg Word Count | Primary Attack Vector |
|---|---|---|---|---|
| `harmful_behavior` | 4,000 | 74.2 | 12.6 | Direct harmful command execution |
| `jailbreak` | 4,000 | 703.1 | 112.3 | Long persona & storytelling contexts |
| `linguistic` | 4,000 | 117.6 | 21.1 | Semantic obfuscation & inquiry framing |
| `prompt_injection` | 4,000 | 158.4 | 24.3 | Directive override particles (`[System:]`, `Ignore`) |
| `toxicity` | 4,000 | 89.1 | 14.8 | Explicit abusive language |

---

## 3. Evaluated Model Architectures (30 Tuning Runs)

We evaluate **10 distinct model families** across **3 hyperparameter configurations each** ($10 \times 3 = 30$ recorded experiments) on the held-out test set ($N=3,000$):

| # | Model Architecture | Category | Key Hyperparameters | Test Accuracy | Test Macro F1 |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | **BERT Base (`bert-base-uncased`)** | **Transformer** | $\text{lr}=3\times 10^{-5}, \text{epochs}=3, \text{warmup}=0.1$ | **96.63%** | **96.64%** |
| 2 | **Soft-Voting Ensemble** | **Ensemble** | Weights: 5 (BERT) : 1 (BiLSTM) : 1 (LogReg) | **96.20%** | **96.22%** |
| 3 | **Logistic Regression** | Classical ML | $C=10.0, \text{L2 penalty}, \text{TF-IDF}=15\text{k}$ | 92.03% | 92.12% |
| 4 | **Bidirectional GRU** | Deep Learning | 2-Layer Bi-GRU (Units: $2\times 128$), Dropout=0.5 | 91.70% | 91.78% |
| 5 | **Bidirectional SimpleRNN** | Deep Learning | 2-Layer Bi-RNN (Units: $2\times 128$), Dropout=0.5 | 91.57% | 91.67% |
| 6 | **Random Forest** | Classical ML | $n=300 \text{ trees}, \text{max\_depth}=50$ | 91.47% | 91.62% |
| 7 | **Bidirectional LSTM** | Deep Learning | 2-Layer Bi-LSTM (Units: $2\times 128$), Dropout=0.5 | 91.47% | 91.55% |
| 8 | **GRU** | Deep Learning | 1-Layer GRU (Units: 128), Dropout=0.5 | 90.77% | 90.90% |
| 9 | **Naive Bayes (MultinomialNB)** | Classical ML | Smoothing $\alpha=0.1$, Sublinear TF-IDF | 90.03% | 90.12% |
| 10 | **SimpleRNN** | Deep Learning | 1-Layer SimpleRNN (Units: 128), Dropout=0.5 | 88.80% | 89.00% |
| 11 | **LSTM** | Deep Learning | 1-Layer LSTM (Units: 128), Dropout=0.5 | 87.67% | 87.92% |

---

## 4. Key Scientific Insights & Ablation Findings

* **Directive Stopword Preservation (+0.9% Macro F1 Gain):** Removing stopwords degrades adversarial classification because grammatical particles (`ignore`, `system`, `you`, `must`, `as`, `previous`) represent critical trigger signals for prompt injection and roleplay framing.
* **Context Length Impact:** Increasing padding length from 128 to 256 tokens significantly prevents information loss for jailbreak prompts (which average 112 words with a long tail).
* **Transformer Superiority:** Fine-tuned BERT Base captures deep bidirectional syntactic dependencies, drastically reducing false positives between subtle linguistic evasion and harmful behavior.

---

## 5. Repository Structure

```
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & benchmark reporting
├── .gitignore                  # Clean repository exclusions
├── data/
│   ├── llm_jailbreak_safety_20k.csv  # 20,000 Curated Prompt Dataset
│   ├── train.csv                     # 70% Training Split (14,000 samples)
│   ├── val.csv                       # 15% Validation Split (3,000 samples)
│   └── test.csv                      # 15% Test Split (3,000 samples)
├── hf_space/
│   ├── app.py                        # Gradio Web App for Hugging Face Spaces
│   ├── requirements.txt              # HF Spaces dependencies
│   └── README.md                     # HF Spaces metadata & configuration
└── results/
    ├── all_30_hyperparameter_tuning_runs.csv  # 30 logged validation experiments
    ├── master_test_benchmarks.csv             # Held-out test set metrics
    ├── label_mapping.json                     # Class label ID mappings
    └── ablation_study_results.csv             # Feature ablation metrics
```

---

## 6. How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

---

## 7. Hugging Face Spaces Deployment

The `hf_space/` directory contains a standalone **Gradio** web application configured to deploy the fine-tuned BERT Base model directly to Hugging Face Spaces for real-time public inference.
