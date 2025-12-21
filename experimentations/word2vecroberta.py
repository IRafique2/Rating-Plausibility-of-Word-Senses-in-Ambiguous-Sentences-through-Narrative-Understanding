import json
import torch
import torch.nn as nn
import numpy as np
import statistics
from datasets import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DefaultDataCollator
)
from scipy.stats import spearmanr
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')

STOPWORDS = set(stopwords.words("english"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("train.json", "r") as f:
    train_data = json.load(f)

with open("dev.json", "r") as f:
    dev_data = json.load(f)

print(f"Train size: {len(train_data)}")
print(f"Dev size: {len(dev_data)}")

def get_wordnet_sense_keywords(homonym, judged_meaning):
    keywords = set()
    for syn in wn.synsets(homonym):
        if judged_meaning.lower() in syn.definition().lower():
            for lemma in syn.lemmas():
                keywords.add(lemma.name().replace("_", " "))
            for w in word_tokenize(syn.definition().lower()):
                if w.isalpha() and w not in STOPWORDS:
                    keywords.add(w)
    return list(keywords)

def wordnet_context_score(context_text, sense_keywords):
    if not sense_keywords:
        return 0.0
    context_tokens = {
        w.lower()
        for w in word_tokenize(context_text)
        if w.isalpha() and w.lower() not in STOPWORDS
    }
    overlap = context_tokens.intersection(set(sense_keywords))
    return len(overlap) / len(sense_keywords)

all_homonyms = sorted(
    {v["homonym"] for v in train_data.values()} |
    {v["homonym"] for v in dev_data.values()}
)

homonym_to_id = {h: i for i, h in enumerate(all_homonyms)}
print(f"Total unique homonyms: {len(homonym_to_id)}")

NUM_HOMONYMS = len(homonym_to_id)

class RobertaHomonymRegressor(nn.Module):
    def __init__(self, model_name, num_homonyms, homonym_dim=32, dropout=0.2):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.homonym_embedding = nn.Embedding(num_homonyms, homonym_dim)
        self.attention = nn.Linear(hidden, 1)

        self.layer_norm = nn.LayerNorm(hidden + homonym_dim + 1)
        self.dropout = nn.Dropout(dropout)

        self.rating_head = nn.Sequential(
            nn.Linear(hidden + homonym_dim + 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden + homonym_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, homonym_ids, wn_score,
                labels=None, stdevs=None):

        outputs = self.encoder(input_ids, attention_mask=attention_mask)

        attn = torch.softmax(
            self.attention(outputs.last_hidden_state).squeeze(-1)
            + (1 - attention_mask.float()) * -1e9,
            dim=1
        )

        pooled = torch.sum(
            outputs.last_hidden_state * attn.unsqueeze(-1), dim=1
        )

        hom_emb = self.homonym_embedding(homonym_ids)
        wn_score = wn_score.unsqueeze(1)

        combined = torch.cat([pooled, hom_emb, wn_score], dim=1)
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        rating_pred = self.rating_head(combined).squeeze(-1)
        uncertainty_pred = self.uncertainty_head(combined).squeeze(-1)

        loss = None
        if labels is not None:
            rating_loss = nn.MSELoss()(rating_pred, labels)
            uncertainty_loss = nn.MSELoss()(uncertainty_pred, stdevs)
            loss = rating_loss + 0.3 * uncertainty_loss

        return {
            "loss": loss,
            "logits": rating_pred,
            "uncertainty": uncertainty_pred
        }

MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def format_example(sample):
    context = (
        sample["precontext"] + " " +
        sample["sentence"] + " " +
        sample.get("ending", "")
    )

    sense_keywords = get_wordnet_sense_keywords(
        sample["homonym"], sample["judged_meaning"]
    )

    wn_score = wordnet_context_score(context, sense_keywords)

    text = (
        f"Word: {sample['homonym']}\n"
        f"Meaning: {sample['judged_meaning']}\n"
        f"Story: {context}"
    )

    mean = (float(sample["average"]) - 1) / 4
    std = max(float(sample["stdev"]) / 2, 0.1)

    return {
        "text": text,
        "labels": mean,
        "stdevs": std,
        "wn_score": wn_score,
        "homonym_id": homonym_to_id[sample["homonym"]]
    }

def tokenize(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    enc["labels"] = batch["labels"]
    enc["stdevs"] = batch["stdevs"]
    enc["wn_score"] = batch["wn_score"]
    enc["homonym_ids"] = batch["homonym_id"]
    return enc

train_dataset = Dataset.from_list([format_example(v) for v in train_data.values()])
dev_dataset = Dataset.from_list([format_example(v) for v in dev_data.values()])

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
dev_dataset = dev_dataset.map(tokenize, batched=True, remove_columns=dev_dataset.column_names)

model = RobertaHomonymRegressor(
    MODEL_NAME, NUM_HOMONYMS
).to(DEVICE)

training_args = TrainingArguments(
    output_dir="./roberta_homonym",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=DefaultDataCollator()
)

trainer.train()

import numpy as np
from scipy.stats import spearmanr
import statistics

pred_array = np.array(preds.predictions).T
pred_values = np.clip(pred_array[:, 0] * 4 + 1, 1, 5)

pred_std = pred_array[:, 1] * 2

gold_values = [v["average"] for v in dev_data.values()]

corr, _ = spearmanr(pred_values, gold_values)
print(f"Spearman correlation: {corr:.4f}")

def within_std(pred, labels):
    """Check if prediction falls within the std of labels"""
    avg = statistics.mean(labels)
    stdev = statistics.stdev(labels)
    return (avg - stdev) <= pred <= (avg + stdev) or abs(avg - pred) < 1

correct = 0
for i, pred in enumerate(pred_values):
    labels = dev_data[str(i)]["choices"]
    if within_std(pred, labels):
        correct += 1

accuracy = correct / len(pred_values)
print(f"Accuracy within gold std: {accuracy:.4f} ({correct}/{len(pred_values)})")

errors = np.abs(pred_values - np.array(gold_values))

import matplotlib.pyplot as plt
import numpy as np

if "uncertainties" not in globals():
    uncertainties = errors

errors = np.abs(pred_values - np.array(gold_values))

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

scatter = axes[0, 0].scatter(
    gold_values,
    pred_values,
    c=uncertainties,
    cmap="viridis",
    alpha=0.6,
    edgecolors="black",
    linewidth=0.5
)

axes[0, 0].plot([1, 5], [1, 5], "r--", linewidth=2, label="Perfect Prediction")
axes[0, 0].set_xlabel("Gold Rating", fontsize=12, weight="bold")
axes[0, 0].set_ylabel("Predicted Rating", fontsize=12, weight="bold")
axes[0, 0].set_title("Predicted vs Gold Ratings (DEV)", fontsize=14, weight="bold")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 0], label="Uncertainty / Error")

axes[0, 1].hist(errors, bins=30, alpha=0.7, edgecolor="black")
axes[0, 1].axvline(errors.mean(), color="red", linestyle="--", linewidth=2,
                    label=f"Mean Error = {errors.mean():.3f}")
axes[0, 1].set_xlabel("Absolute Error", fontsize=12, weight="bold")
axes[0, 1].set_ylabel("Frequency", fontsize=12, weight="bold")
axes[0, 1].set_title("Error Distribution on DEV Set", fontsize=14, weight="bold")
axes[0, 1].legend()
axes[0, 1].grid(axis="y", alpha=0.3)


challenging_mask = dev_df["homonym"].isin(CHALLENGING_HOMONYMS).to_numpy()
challenging_errors = errors[challenging_mask]
other_errors = errors[~challenging_mask]

axes[1, 0].boxplot([challenging_errors, other_errors],
                   labels=["Challenging Homonyms", "Other Homonyms"])
axes[1, 0].set_ylabel("Absolute Error", fontsize=12, weight="bold")
axes[1, 0].set_title("Error by Homonym Difficulty", fontsize=14, weight="bold")
axes[1, 0].grid(alpha=0.3)

axes[1, 1].axis("off")
summary_text = f"""
DATASET SUMMARY

TRAIN:
• {len(train_data)} stories
• {len(set(v['homonym'] for v in train_data.values()))} homonyms

DEV:
• {len(dev_data)} stories

RESULTS (DEV):
• Spearman: {corr:.4f}
• MAE: {errors.mean():.4f}
"""
axes[1, 1].text(0.05, 0.5, summary_text, fontsize=11, family="monospace",
                verticalalignment="center")

plt.tight_layout()
plt.savefig("roberta_homonym_results.png", dpi=300)
plt.show()