---
title: LLM Adversarial Threat & Jailbreak Classifier
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.28.3
app_file: app.py
pinned: false
license: mit
short_description: Real-time adversarial prompt classification using fine-tuned BERT
---

# 🛡️ LLM Adversarial Threat & Jailbreak Classifier
### **CSE440: Natural Language Processing II — Research Project**

This space hosts a fine-tuned **BERT Base (`bert-base-uncased`)** sequence classification model trained on **20,000 curated adversarial prompts** spanning 5 key vulnerability categories:
1. **Jailbreak** (DAN persona exploits, fictional roleplay filters)
2. **Prompt Injection** (System instruction override, directive hijack)
3. **Harmful Behavior** (Direct illegal, cyber, or destructive requests)
4. **Toxicity** (Hate speech, profanity, harassment)
5. **Linguistic Evasion** (Subtle social engineering, syntactic obfuscation)

## Model Performance
- **Test Accuracy:** 96.63%
- **Test Macro F1:** 96.64%
- **Dataset Size:** 20,000 balanced English samples
- **Split:** 70% Train (14,000), 15% Validation (3,000), 15% Test (3,000)
