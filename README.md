# TrueNab: Evaluating LLMs for Arabic Fake News Detection

This repository contains the code and experimental setup for our research evaluating **Large Language Models (LLMs)** for detecting **fake news and disinformation in Arabic**. The study benchmarks multiple advanced models across two datasets and investigates the influence of zero-shot, few-shot, and chain-of-thought (CoT) prompting strategies.

---

## Overview

The rise of online misinformation presents a significant challenge, especially in **low-resource languages** like Arabic. This project explores how modern LLMs perform on the task of classifying Arabic fake news.

We evaluate several prominent models, including:

* Allam
* Noon
* Bloomz
* Llama-8B
* SILMA
* GPT-4o

across two benchmark datasets:

* **ANS**
* **ArAiEval**

The study investigates:

* Zero-shot classification
* Few-shot in-context learning
* Chain-of-thought (CoT) prompting

---

## Key Findings

* In-context learning improves the performance of several models
* Some models struggle due to precision–recall trade-offs, limiting their overall F1 performance.
* CoT prompting shows **dataset-dependent benefits**, performing well only in specific cases.
* Despite being top performers, the highest-scoring models achieve results that are **only slightly above a near-random baseline**, highlighting the inherent difficulty of Arabic fake news detection.

**Top-model performance:**

* **GPT-4o:** F1 = 0.647 on ArAiEval (16-shot)
* **SILMA:** F1 = 0.570 on ANS (zero-shot)

These trends demonstrate that **advanced Arabic LLMs and multilingual—still struggle** with the nuances of misinformation detection in Arabic text.

---

## Repository Structure

```
.
├── expirement/
│   ├── fake_news.py          
│   ├── prompt_cot.txt            
|   ├── prompt_shots.txt
|   ├── requirements.txt 
│   ├── script.bash         
│   └── datasets/               
└── README.md
```

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Datasets should be placed as follows:

```
expirement/datasets/ANS/train.csv
expirement/datasets/ANS/test.csv
expirement/datasets/ArAiEval/train.csv
expirement/datasets/ArAiEval/test.csv
```

Each dataset CSV must contain:

* `claim_s`
* `fake_flag`

---

## Running Experiments

### Zero-Shot (OpenAI API)

```bash
python run.py \
    --provider openai \
    --model gpt-4o-mini \
    --task ANS \
    --input test.csv \
    --no_shots 0
```

### Few-Shot (Local HuggingFace model)

```bash
python run.py \
    --provider hf-local \
    --model meta-llama/Llama-8b \
    --task ArAiEval \
    --no_shots 16
```

All predictions, raw outputs, and evaluation summaries will be saved automatically in:

```
expirement/predictions/
```

---

## Results

Each experiment generates:

* Parsed predictions
* Raw LLM outputs
* Parsing status
* Error indices
* Metrics (Accuracy, Precision, Recall, F1)

All results are stored under:

```
expirement/predictions/
```

---

## Abstract

The growing prevalence of fake news and disinformation in digital media poses a significant challenge, particularly in low-resource languages such as Arabic. To address this issue, we investigate the effectiveness of large language models (LLMs) in detecting fake news and disinformation in Arabic text. Various LLMs, including Allam, Noon, Bloomz, Llama8b, Silma, and GPT-4o, are benchmarked on two datasets: ANS and ArAiEval. Additionally, we explored the impact of zero-shot, few-shot, and chain-of-thought (CoT) reasoning on the models' performance across these datasets. Our experiments demonstrate that in-context learning benefits some models, including GPT-4o and SILMA, while others remain limited by precision–recall trade-offs. Moreover, the CoT prompting effect is highly dependent on the nature of the dataset. Across the two tasks, the top-performing systems—GPT-o with an F1-score of 0.647 on ArAiEval (16-shot) and ANS SILMA with 0.570 in a zero-shot setting—achieve only marginal gains over a near-random baseline, indicating that the benchmark remains challenging for advanced or even Arabic models. These findings contribute to computational linguistics and emphasize the practical applications of LLMs in maintaining information integrity.

---

## Code & Data Availability

All code required to reproduce the experiments is included in this repository.
