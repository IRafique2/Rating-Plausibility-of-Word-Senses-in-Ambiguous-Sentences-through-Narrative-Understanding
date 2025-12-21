#import Dependencies
import os
import json
import random
import torch
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch.nn as nn 
from transformers import TrainingArguments
import statistics
from scipy.stats import spearmanr
import json
from transformers import DefaultDataCollator

#import data
with open("train.json", "r") as f:
    train_data = json.load(f)

with open("dev.json", "r") as f:
    dev_data = json.load(f)

# import models
MODEL_NAME="FacebookAI/roberta-large"
base_model = AutoModel.from_pretrained(MODEL_NAME)
# define a custom class
class MiniLMRegression(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model

        self.norm = nn.LayerNorm(model.config.hidden_size)

        self.regressor = nn.Linear(model.config.hidden_size, 1)

    @property
    def device(self):
        return self.encoder.device

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)

        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)
        pooled = self.norm(pooled)

        preds = self.regressor(pooled).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fn = nn.SmoothL1Loss(beta=0.5)  # better than MSE
            loss = loss_fn(preds, labels.float())

        return {"loss": loss, "logits": preds}


#call the model
model=MiniLMRegression(base_model)

#call tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

train_values = list(train_data.values())
labels = np.array([int(round(v['average'])) for v in train_values])

few_examples = []

for score in range(1, 6):
    candidates = [v for v, lbl in zip(train_values, labels) if lbl == score]
    if candidates:
        few_examples.append(random.choice(candidates))

while len(few_examples) < 3:
    few_examples.append(random.choice(train_values))



def unfreeze_last_n_layers(model, n_layers=7):
    """
    Unfreeze last n transformer layers + all LayerNorms
    """

    #  Freeze everything first (safety)
    for param in model.parameters():
        param.requires_grad = False

    #  Unfreeze LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    #  Unfreeze last N encoder layers

    # Corrected line for RoBERTa
    encoder_layers = model.base_model.encoder.encoder.layer

    for layer in encoder_layers[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    #  Unfreeze all LayerNorms (VERY IMPORTANT)
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = True

    print(f" Unfroze last {n_layers} layers + LayerNorms + LoRA")


model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
   r = 16,
lora_alpha = 32,

    target_modules=["query", "key", "value", "dense"], 
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION"  
)


model = get_peft_model(model, lora_config)

unfreeze_last_n_layers(model, n_layers=7)

model.print_trainable_parameters()

def format_train_example(sample):
    text = (
        sample['precontext'] + " "
        + sample['sentence'] + " "
        + sample.get('ending', '') + "\n"
        + "Target meaning: " + sample['judged_meaning']
    )
    label = (float(sample["average"]) - 1) / 4   # → [0,1]

    return {"text": text, "labels": label}

def tokenize(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    enc["labels"] = batch["labels"]
    return enc

# Format dev examples
dev_examples = [format_train_example(v) for v in dev_data.values()]

dev_dataset = Dataset.from_list(dev_examples)

# Tokenize dev dataset
tokenized_dev = dev_dataset.map(
    tokenize,
    batched=True,
    remove_columns=dev_dataset.column_names
)


training_args = TrainingArguments(
    output_dir="./tmp_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=20,
    learning_rate=5e-5,
    logging_steps=10,
    eval_strategy="steps",  # IMPORTANT: enable eval
    eval_steps=100,               # evaluate every N steps
    save_strategy="no",
    report_to="none",
    fp16=True,
    remove_unused_columns=False # <--- Add this line to prevent labels from being removed
)

r

data_collator = DefaultDataCollator()



# Format train examples
train_examples = [format_train_example(v) for v in train_data.values()]
train_dataset = Dataset.from_list(train_examples)

# Tokenize train dataset
tokenized_train = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=train_dataset.column_names
)
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    corr, _ = spearmanr(preds, labels)
    return {"spearman": corr}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    compute_metrics=compute_metrics,
    eval_dataset=tokenized_dev,  
    data_collator=data_collator
)



trainer.train()
print(" LoRA fine-tuning finished")



def get_standard_deviation(l):
    return statistics.stdev(l)

def get_average(l):
    return sum(l)/len(l)
def is_within_standard_deviation(prediction, labels):
    avg = get_average(labels)
    stdev = get_standard_deviation(labels)

    if (avg - stdev) < prediction < (avg + stdev):
        return True
    if abs(avg - prediction) < 1:
        return True
    return False

def spearman_evaluation_score(predictions_filepath, gold_data):
    gold_list = []
    pred_list = []

    with open(predictions_filepath, "r") as f:
        for line in f:
            line = json.loads(line)
            id_str = str(line["id"])
            gold_avg = get_average(gold_data[id_str]["choices"])
            gold_list.append(gold_avg)
            pred_list.append(line["prediction"])

    # ✅ ADD THESE DEBUG PRINTS
    print("Prediction variance:", np.var(pred_list))
    print("Prediction min/max:", min(pred_list), max(pred_list))
    print("Unique predictions:", sorted(set(pred_list))[:10])

    corr, value = spearmanr(pred_list, gold_list)
    print(f"Spearman Correlation (TEST set): {corr}")
    print(f"Spearman p-Value: {value}")

    return corr


def accuracy_within_standard_deviation_score(predictions_filepath, gold_data):
    correct_guesses = 0
    total = 0

    with open(predictions_filepath, "r") as f:
        for line in f:
            line = json.loads(line)
            labels = gold_data[str(line["id"])]
            if "choices" in labels: # Check if 'choices' key exists
                labels = labels["choices"]
            else:
                print(f"Warning: 'choices' key not found for ID {line['id']}. Skipping.")
                continue

            if is_within_standard_deviation(line["prediction"], labels):
                correct_guesses += 1
            total += 1

    acc = correct_guesses / total
    print(f"Accuracy within standard deviation (TEST set): {acc} ({correct_guesses}/{total})")
    return acc




# Direct evaluation using LoRA fine-tuned model (no prompting) on TEST set


direct_predictions = {}

for i, (id_str, sample) in enumerate(dev_data.items(), 1):
    # Format text like in training but without any prompting
    input_text = (
    sample['precontext'] + " "
    + sample['sentence'] + " "
    + sample.get('ending', '') + "\n"
    + "Target meaning: " + sample['judged_meaning']
    )

    # Tokenize the input text and prepare for model
    encoded_input = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt" # Return PyTorch tensors
    )

    # Move tensors to the model's device
    inputs = {k: v.to(model.device) for k, v in encoded_input.items() if k in ['input_ids', 'attention_mask']}

    with torch.no_grad():
      output = model(**inputs)
      pred = output["logits"].item()

    # Clamp prediction to 1-5
    pred = max(1, min(5, pred * 4 + 1)) # Revert scaling from [0,1] back to [1,5]
    direct_predictions[id_str] = pred

    print(f"[{i}/{len(dev_data)}] {id_str}: {pred}")

# Save predictions to JSONL file
direct_predictions_filepath = "results/direct_lora_test_predictions.json"

# Create the 'results' directory if it doesn't exist
os.makedirs(os.path.dirname(direct_predictions_filepath), exist_ok=True)

with open(direct_predictions_filepath, "w") as f:
    for id_str, pred in direct_predictions.items():
        f.write(json.dumps({"id": int(id_str), "prediction": pred}) + "\n")

print(f"Direct LoRA test predictions saved to {direct_predictions_filepath}")


# Evaluate using test set

corr = spearman_evaluation_score(direct_predictions_filepath, dev_data)
acc = accuracy_within_standard_deviation_score(direct_predictions_filepath, dev_data)

print("Direct LoRA test evaluation complete.")