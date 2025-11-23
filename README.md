# SemEval 2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding - Solution

## Overview

This repository contains the proposed solution for the **SemEval 2026 Task 5**: *Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding*. 
The task challenges participants to predict the human-perceived plausibility of a word sense in a given narrative context, considering the nuances of word sense disambiguation
(WSD) and narrative coherence.

Our solution aims to address the ambiguity in word sense selection by leveraging contextual understanding from surrounding sentences in short stories. The dataset provided for
this task, **AmbiStory**, consists of 5-sentence short stories, with a homonym in the fourth sentence whose interpretation depends on the surrounding narrative. Participants 
are tasked with rating the plausibility of different senses of the homonym, based on how well they fit the context provided by the story.

## Task Description

In this task, the goal is to **predict the human-perceived plausibility** of a word sense by assigning a score between 1 and 5. The task evaluates the system's ability to
understand narrative context and resolve word sense ambiguity based on contextual clues.

### Input

* **Stories**: Each story consists of:

  * **Precontext**: The first three sentences ground the story and provide necessary background.
  * **Ambiguous Sentence**: The fourth sentence contains a homonym, with two widely different plausible interpretations based on the surrounding context.
  * **Ending**: Optionally, one of two endings that help indicate the intended sense of the homonym.

### Task Objective

* **Predict plausibility**: For each story, the model must assign a plausibility score to a word sense between 1 and 5, reflecting how plausible that sense is within the context of the entire story.

### Metrics

We evaluate the model using two primary metrics:

1. **Spearman Correlation**: Measures how closely the predicted plausibility scores correlate with the human-annotated average scores.

2. **Accuracy Within Standard Deviation**: This metric calculates the proportion of model predictions that fall within one standard deviation of the average human judgment, accounting for variability in the ratings.

## Data

### AmbiStory Dataset

The **AmbiStory dataset** includes:

* **5 stories** for initial evaluation, representing 30 samples in total.
* **Training and validation sets** are available for model development.
* **Test set** is not publicly available; the evaluation phase is scheduled for **January 10th, 2026 - January 31st, 2026**.

[Download the sample data here!](https://drive.google.com/drive/folders/1evACvNGyBPKr99R5Db-zo2sK4u2_a9i3)

[Download the training data here!](https://drive.google.com/drive/folders/1evACvNGyBPKr99R5Db-zo2sK4u2_a9i3)


## Solution Overview

### Approach



### Key Components


## Installation and Usage

To use this solution, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/semeval2026-task5.git
cd semeval2026-task5
pip install -r requirements.txt
```

### Running the Model



3. **Evaluate the Model**: 

## Evaluation


## Contribution

We acknowledge the following contributors for their work in developing this solution:

* **Team Members**: [Areeba Munir,Isra Rafique,Maryum Arshad,Noor ul ain]
* **Mentors/Advisors**: [Dr Mehwish Fatima]




## Acknowledgements

We would like to thank the SemEval 2026 organizers for providing the AmbiStory dataset and the opportunity to participate in this important task. Special thanks to the 
Prolific participants whose ratings form the foundation of this challenge.

---

Feel free to modify any parts of the README to align with your specific implementation details.
