#!/usr/bin/env python
# =============================================================================
# 03a_bert_finetune_gpu.py
# -----------------------------------------------------------------------------
# Fine-tunes bert-base-uncased on the GossipCop++ 60/20/20 split for the TFM
# "Detecting Misinformation in the Age of LLMs" and writes the two files that
# the R/Quarto evaluation (analysis/03b_bert_eval.qmd) reads back:
#
#     bert_test_predictions.csv   (one row per held-out TEST document)
#     bert_training_log.csv       (per-epoch validation metrics)
#
# This is the SAME code as the (eval:false) Python chunks in analysis/03b_bert_eval.qmd,
# packaged as a script so it can run on a free GPU (Google Colab / Kaggle).
#
# ----------------------------- HOW TO RUN ------------------------------------
# 1. In R (locally), run the BERT modelling-data chunk of analysis/03b_bert_eval.qmd. It writes gossip_bert_data.csv.
# 2. Open Google Colab -> Runtime -> Change runtime type -> GPU (T4 is enough).
# 3. Upload gossip_bert_data.csv and this script (left sidebar -> Files), then:
#        !pip -q install "transformers>=4.44" "datasets>=2.19" \
#                        "accelerate>=0.30" scikit-learn pandas
#        !python 03a_bert_finetune_gpu.py
#    (or just paste the body of this file into a cell).
# 4. Download bert_test_predictions.csv and bert_training_log.csv from the
#    Files panel into the repo's artifacts/ folder (next to the committed
#    bert_test_predictions.csv / bert_training_log.csv).
# 5. Back in R, run analysis/03b_bert_eval.qmd. It will find the files and evaluate.
#
# Recipe (proposal / Devlin et al. 2019): AdamW, lr 2e-5, batch 16, 3 epochs,
# linear warmup over the first 10% of steps, inputs truncated to 512 tokens,
# best checkpoint chosen on validation macro-F1.
# =============================================================================

import os
import random
import inspect

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from transformers.utils import logging as hf_logging
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
INPUT_CSV   = "gossip_bert_data.csv"          # written by the R eval notebook
PRED_CSV    = "bert_test_predictions.csv"     # consumed by the R eval notebook
LOG_CSV     = "bert_training_log.csv"         # consumed by the R eval notebook
MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 512
SEED        = 2026                            # same seed as the rest of the TFM

hf_logging.set_verbosity_error()              # quiet the ">512 tokens" notes

# --------------------------------------------------------------------------- #
# Determinism + device
# --------------------------------------------------------------------------- #
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
print("Torch:", torch.__version__, "| device:", device)
if device == "cpu":
    print("WARNING: no GPU detected. Fine-tuning bert-base on ~8.4k docs for "
          "3 epochs will take HOURS on CPU. Use a Colab/Kaggle GPU runtime.")

# --------------------------------------------------------------------------- #
# Load the shared split data (built in R so the comparison stays fair)
# --------------------------------------------------------------------------- #
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"'{INPUT_CSV}' not found in {os.getcwd()}. Run the BERT modelling-data chunk of analysis/03b_bert_eval.qmd "
        "in R to create it, then upload it next to this script."
    )

label2id = {"real": 0, "fake": 1}             # 'fake' is class 1 -> prob_fake = P(class 1)
id2label = {0: "real", 1: "fake"}

df = pd.read_csv(INPUT_CSV)

# Safety: the R notebook guarantees these, but guard anyway.
df = df.dropna(subset=["doc_text"]).copy()
df = df[df["doc_text"].astype(str).str.strip() != ""].copy()
df["labels"] = df["label"].map(label2id).astype(int)

df_train = df[df["split"] == "train"].reset_index(drop=True)
df_val   = df[df["split"] == "validation"].reset_index(drop=True)
df_test  = df[df["split"] == "test"].reset_index(drop=True)
print(f"train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")
assert len(df_train) and len(df_val) and len(df_test), \
    "One of the splits is empty. Check the 'split' column in gossip_bert_data.csv."

# --------------------------------------------------------------------------- #
# Tokenise + build HF datasets
# --------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(batch):
    return tokenizer(batch["doc_text"], truncation=True, max_length=MAX_LEN)


def to_dataset(frame):
    return (
        Dataset.from_pandas(frame[["doc_text", "labels"]], preserve_index=False)
        .map(tokenize_fn, batched=True, remove_columns=["doc_text"])
    )


train_ds = to_dataset(df_train)
val_ds   = to_dataset(df_val)
test_ds  = to_dataset(df_test)


# Untruncated token length (incl. [CLS]/[SEP]) -> truncation flag for the truncation sensitivity analysis
def token_lengths(texts):
    return [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]


df_test["n_tokens"]  = token_lengths(df_test["doc_text"].tolist())
df_test["truncated"] = df_test["n_tokens"] > MAX_LEN

for name, frame in [("train", df_train), ("val", df_val), ("test", df_test)]:
    lens = (df_test["n_tokens"].to_numpy() if name == "test"
            else np.asarray(token_lengths(frame["doc_text"].tolist())))
    print(f"{name:5s}: median tokens={int(np.median(lens))}, "
          f">512 = {(lens > MAX_LEN).mean():.1%}")

# --------------------------------------------------------------------------- #
# Model + metrics
# --------------------------------------------------------------------------- #
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy":       accuracy_score(labels, preds),
        "f1_macro":       f1_score(labels, preds, average="macro"),
        "f1_fake":        f1_score(labels, preds, pos_label=1, average="binary"),
        "precision_fake": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_fake":    recall_score(labels, preds, pos_label=1, zero_division=0),
    }


# --------------------------------------------------------------------------- #
# TrainingArguments — version-robust (eval_strategy vs evaluation_strategy)
# --------------------------------------------------------------------------- #
_ta_params = inspect.signature(TrainingArguments.__init__).parameters
_eval_key  = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"

args_kwargs = dict(
    output_dir="bert_gossipcop_ckpt",
    num_train_epochs=3,                  # proposal recipe (Devlin et al., 2019)
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,                    # linear warmup over first 10% of steps
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",    # checkpoint selected on validation macro-F1
    greater_is_better=True,
    seed=SEED,
    fp16=torch.cuda.is_available(),      # mixed precision on GPU for speed
    report_to="none",                    # no wandb/tensorboard prompts on Colab
)
args_kwargs[_eval_key]       = "epoch"
args_kwargs["save_strategy"] = "epoch"   # must match eval strategy for best-ckpt reload
training_args = TrainingArguments(**args_kwargs)

# --------------------------------------------------------------------------- #
# Trainer — version-robust (processing_class vs tokenizer)
# --------------------------------------------------------------------------- #
trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
_tr_params = inspect.signature(Trainer.__init__).parameters
if "processing_class" in _tr_params:           # transformers >= ~4.46
    trainer_kwargs["processing_class"] = tokenizer
else:                                           # older transformers
    trainer_kwargs["tokenizer"] = tokenizer
trainer = Trainer(**trainer_kwargs)

# --------------------------------------------------------------------------- #
# Fine-tune
# --------------------------------------------------------------------------- #
trainer.train()
print("Best validation f1_macro:", trainer.state.best_metric)


# --------------------------------------------------------------------------- #
# Score the held-out TEST split (row order preserved == df_test order)
# --------------------------------------------------------------------------- #
def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


pred_out  = trainer.predict(test_ds)
logits    = pred_out.predictions
prob_fake = softmax_np(logits)[:, label2id["fake"]]
pred_id   = logits.argmax(axis=1)
assert len(pred_id) == len(df_test), "Prediction/row-count mismatch on test set."

predictions = pd.DataFrame({
    "doc_id":    df_test["doc_id"].values,
    "subclass":  df_test["subclass"].values,
    "truth":     df_test["label"].values,         # "fake"/"real"
    "pred":      [id2label[i] for i in pred_id],
    "prob_fake": prob_fake,
    "n_tokens":  df_test["n_tokens"].values,
    "truncated": df_test["truncated"].values,
})
predictions.to_csv(PRED_CSV, index=False)
pd.DataFrame(trainer.state.log_history).to_csv(LOG_CSV, index=False)

print("\nDONE.")
print(f"  Wrote {PRED_CSV}  ({len(predictions)} test rows)")
print(f"  Wrote {LOG_CSV}")
print("Download BOTH files into the repo's artifacts/ folder, then run analysis/03b_bert_eval.qmd in R.")
