# SemEval-2026 Task 5

## Rating Plausibility of Word Senses in Ambiguous Narratives

## Overview

This repository contains our proposed solution for **SemEval 2026 – Task 5**, which focuses on **rating the plausibility of word senses in ambiguous narrative contexts**. The task evaluates a system’s ability to perform **fine-grained word sense disambiguation (WSD)** by leveraging **narrative coherence and contextual understanding**.

Given a short story containing a homonym, the goal is to predict how plausible a particular word sense is **from a human perspective**, based on the surrounding narrative. Our approach emphasizes **encoder-based semantic modeling**, **parameter-efficient fine-tuning**, and **ensemble learning** to robustly capture contextual cues and resolve ambiguity.

---

## Task Description

Each sample consists of a **5-sentence short story** from the **AmbiStory** dataset:

* **Pre-context (Sentences 1–3):** Establish narrative background
* **Ambiguous Sentence (Sentence 4):** Contains a homonym with multiple plausible senses
* **Ending (Sentence 5, optional):** Helps signal the intended sense

### Objective

Predict a **plausibility score between 1 and 5** for a given word sense, reflecting how well it fits within the narrative context.

---

## Evaluation Metrics

The task uses two official evaluation metrics:

* **Spearman Correlation**
  Measures rank correlation between predicted scores and human-annotated plausibility scores.

* **Accuracy within ±1 Standard Deviation**
  Measures how often model predictions fall within one standard deviation of the human mean score.

---

## Dataset

### AmbiStory

* 5 stories for trial evaluation (30 total samples)
* Full training and validation sets available
* Test set withheld for official evaluation



---

## Solution Overview

### Key Idea

We propose a **comprehensive suite of encoder-based modeling approaches**, ranging from classical baselines to advanced **multi-transformer semantic fusion ensembles**. Our experiments demonstrate that **deep contextual encoders**, when combined with **semantic embeddings and ensemble learning**, significantly outperform decoder-only LLMs and lexical baselines.

---

## Modeling Approaches

### 1. Baseline Models

* **TF-IDF + Logistic Regression**
* **Knowledge-Graph-based WSD**
* **Zero-shot prompting:** LLaMA-3.1 (8B), Qwen
* **Encoder–Decoder prompting:** T5-base (zero-shot & few-shot)

---

### 2. Encoder Fine-Tuning

Fully fine-tuned pretrained encoders with regression heads:

* **BERT-base**
* **RoBERTa-base**
* **DeBERTa-v3**

These models directly regress plausibility scores from narrative context.

---

### 3. Parameter-Efficient Fine-Tuning (LoRA)

* Applied to **RoBERTa**, **DeBERTa**, and **MiniLM**
* Experiments with:

  * Fully frozen backbones
  * Partial layer unfreezing
* Optimized using **Smooth L1 Loss**

---

### 4. Hybrid RoBERTa–Word2Vec Model

* Word sense keywords extracted from **WordNet**
* Context overlap scores computed using **Word2Vec**
* Lexical overlap features concatenated with RoBERTa embeddings

---

### 5. LoRA-Enhanced Stacked Ensemble

* Base models:

  * **DeBERTa-v3-large + RoBERTa-large**
* Additional **SentenceTransformer semantic embeddings**
* Two-level stacking using:

  * Gradient Boosting
  * Random Forest meta-learners

---

### 6. Multi-Transformer Semantic Fusion Ensemble (Best Model)

Our best-performing system combines:

* Independently fine-tuned:

  * **DeBERTa-v3-base**
  * **DeBERTa-v3-large**
  * **RoBERTa-large**
* Dense semantic representations from **Sentence Transformers**
* **Gradient Boosting Regressor** as a meta-learner for semantic fusion

This architecture captures **complementary contextual and semantic signals** across models.

---

## Results

### Baseline Results

| No. | Model                           | Spearman | Accuracy |
| --- | ------------------------------- | -------- | -------- |
| 1   | TF-IDF + Logistic Regression    | -0.0365  | 0.0391   |
| 2   | Knowledge Graph                 | 0.0620   | 0.4000   |
| 3   | Knowledge + Sentence Embeddings | 0.0920   | 0.4300   |
| 4   | BERT-base (baseline)            | 0.1718   | 0.5816   |
| 5   | RoBERTa-base (baseline)         | 0.4396   | 0.6701   |
| 6   | DeBERTa-v3-base (LoRA)          | 0.3830   | 0.6276   |
| 7   | MiniLM                          | 0.0100   | 0.5680   |
| 8   | LLaMA-3.1 (Zero-shot)           | 0.1072   | 0.4626   |
| 9   | Qwen (Zero-shot)                | 0.0086   | 0.5200   |
| 10  | T5-base (Zero-shot)             | 0.0479   | 0.4626   |
| 11  | T5-base (Few-shot)              | 0.0654   | 0.5425   |

---

### Experimental Results

| No. | Model                                          | Spearman   | Accuracy   |
| --- | ---------------------------------------------- | ---------- | ---------- |
| 1   | BERT-base (Fine-tuning)                        | 0.3132     | 0.6684     |
| 2   | RoBERTa-base (10 Epochs)                       | 0.5031     | 0.7211     |
| 3   | RoBERTa-base (20 Epochs)                       | 0.4887     | 0.7245     |
| 4   | RoBERTa + Word2Vec (Hybrid)                    | 0.4965     | 0.7065     |
| 5   | MiniLM + LoRA (Frozen)                         | 0.2200     | 0.6000     |
| 6   | MiniLM + LoRA (7 Layers Unfrozen)              | 0.2500     | 0.6300     |
| 7   | RoBERTa-base + LoRA (7 Layers Unfrozen)        | 0.4700     | 0.7000     |
| 8   | RoBERTa-base + LoRA (Frozen)                   | 0.2000     | 0.6000     |
| 9   | DeBERTa-v3-large + LoRA                        | 0.0312     | 0.5544     |
| 10  | RoBERTa-large + LoRA                           | 0.4597     | 0.6990     |
| 11  | LoRA-Enhanced Stacked Ensemble                 | 0.4644     | 0.7279     |
| 12  | DeBERTa-v3-base (Ensemble Component)           | 0.1560     | 0.5867     |
| 13  | DeBERTa-v3-large (Ensemble Component)          | **0.6570** | 0.7245     |
| 14  | RoBERTa-large (Ensemble Component)             | 0.5300     | 0.7262     |
| 15  | **Multi-Transformer Semantic Fusion Ensemble** | **0.6525** | **0.8044** |

---

### Key Observations

* Encoder-based models consistently outperform decoder-only LLMs
* Full fine-tuning generally surpasses LoRA unless carefully configured
* Larger models benefit substantially from ensemble learning
* Semantic embeddings significantly improve robustness under ambiguity

---

## Installation & Usage

```bash
git clone https://github.com/your-repo/semeval2026-task5.git
cd semeval2026-task5
pip install -r requirements.txt
```

---

## Contributors

**Team Members:**

* Areeba Munir
* Isra Rafique
* Maryam Arshad
* Noor-ul-Ain

**Advisor:**

* Dr. Mehwish Fatima

---

## Acknowledgements

We thank the **SemEval 2026 organizers** for releasing the AmbiStory dataset and enabling research on narrative-based word sense plausibility. Special appreciation goes to the **Prolific annotators** whose human judgments form the backbone of this task.


* Write a **short “Reproducibility” section**
* Optimize it for **leaderboard submission clarity**
