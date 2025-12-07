
# Import Dependencies
"""

import os
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from scipy.stats import spearmanr
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Setup directories
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

"""# Load Data"""

with open("train.json", "r") as f:
    train_data = json.load(f)

with open("dev.json", "r") as f:
    dev_data = json.load(f)

"""# Load Model and tokenizer"""

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

print("Model loaded in 4-bit successfully!")

"""# Prompt Templte creation"""

def build_zero_shot_prompt(sample):
    return f"""
{sample['precontext']} **{sample['sentence']}** {sample.get('ending','')}
In this context, how plausible is it that the meaning of the word "{sample['homonym']}" is "{sample['judged_meaning']}"?
Return only the numbered score (1,2,3,4,5)
"""

"""# Prediction"""

def get_model_prediction(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract first integer 1-5 from model output
    for token in decoded.split():
        if token.isdigit() and 1 <= int(token) <= 5:
            return int(token)

    # fallback
    return 3

def get_average(l):
    return sum(l)/len(l)

def get_standard_deviation(l):
    return statistics.stdev(l)

def is_within_standard_deviation(pred, labels):
    avg = get_average(labels)
    stdev = get_standard_deviation(labels)
    if (avg - stdev) <= pred <= (avg + stdev):
        return True
    if abs(avg - pred) < 1:
        return True
    return False

def evaluate(predictions, gold_data):
    gold_list, pred_list = [], []
    correct, total = 0, 0
    errors = []

    for id_str, pred in predictions.items():
        labels = gold_data[id_str]['choices']
        avg = get_average(labels)
        gold_list.append(avg)
        pred_list.append(pred)

        if is_within_standard_deviation(pred, labels):
            correct += 1
        else:
            errors.append((id_str, pred, avg))
        total += 1

    # Spearman correlation
    corr, p_val = spearmanr(pred_list, gold_list)
    accuracy = correct / total

    # MAE and MSE
    mae = mean_absolute_error(gold_list, pred_list)
    mse = mean_squared_error(gold_list, pred_list)

    print(f"Spearman correlation: {corr:.4f} (p={p_val:.4f})")
    print(f"Accuracy within SD: {accuracy:.4f} ({correct}/{total})")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    return {
        'spearman': corr,
        'accuracy_within_std': accuracy,
        'mae': mae,
        'mse': mse,
        'errors': errors
    }

def plot_corrections(errors, save_path=None, metrics=None):
    """
    Plots predicted vs gold averages for errors only.

    errors: list of tuples (id_str, pred, gold_avg)
    save_path: optional path to save the figure
    metrics: optional dict with spearman, accuracy, mae, mse to show in summary
    """
    if not errors:
        print("No errors to plot!")
        return

    preds = [e[1] for e in errors]
    golds = [e[2] for e in errors]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    # 1. Predictions vs Gold (errors only)
    scatter = axes[0].scatter(golds, preds, alpha=0.7, edgecolors='black', s=50)
    axes[0].plot([min(golds), max(golds)], [min(golds), max(golds)], 'r--', label='Perfect')
    axes[0].set_xlabel("Gold Average")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Prediction vs Gold (Errors Only)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Absolute Error Distribution
    abs_errors = [abs(p - g) for p, g in zip(preds, golds)]
    axes[1].hist(abs_errors, bins=20, color='#4C72B0', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(abs_errors), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(abs_errors):.3f}')
    axes[1].set_xlabel("Absolute Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Error Distribution")
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # Optional metrics display
    if metrics:
        metric_text = f"""
        Spearman: {metrics.get('spearman', 0):.4f}
        Accuracy (within SD): {metrics.get('accuracy_within_std', 0):.4f}
        MAE: {metrics.get('mae', 0):.4f}
        MSE: {metrics.get('mse', 0):.4f}
        """
        plt.figtext(0.5, -0.05, metric_text, ha='center', fontsize=11, family='monospace')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

zero_shot_predictions = {}

for i, (id_str, sample) in enumerate(dev_data.items(), 1):
    prompt = build_zero_shot_prompt(sample)
    pred = get_model_prediction(prompt)
    zero_shot_predictions[id_str] = pred

    print(f"[{i}/{len(dev_data)}] Sample {id_str} done, prediction: {pred} at {datetime.now().strftime('%H:%M:%S')}")

# Use the updated evaluate function
results = evaluate(zero_shot_predictions, dev_data)

# Plot errors/corrections
plot_corrections(results['errors'])

# Save predictions
with open("results/zero_shot_results.json", "w") as f:
    json.dump(zero_shot_predictions, f)

print("Zero-shot predictions complete! Results saved to results/zero_shot_results.json")
print(f"Spearman: {results['spearman']:.4f}, Accuracy: {results['accuracy_within_std']:.4f}, MAE: {results['mae']:.4f}, MSE: {results['mse']:.4f}")

