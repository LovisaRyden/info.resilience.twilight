import json, os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score

# --- EDIT THESE TWO PATHS IF NEEDED ---
RUN_DIR = r".\finetuned_multitask_roberta_20250812_033718"        # either run is fine for test_ids
CSV_FT  = r".\comments_subsample_finetfinish_ft_hybrid.csv"


# --------------------------------------

print("USING:", RUN_DIR, "CSV:", CSV_FT)


# Labels (must match what you trained on)
LABELS = ["Spam_flag","irony","distrust","trust","hostility",
          "concept_misinfo","believe_misinfo","Accuse_misinfo",
          "cogni_dis_manual","discrim_manual"]

# Load run config (for thresholds & sanity)
with open(os.path.join(RUN_DIR, "label_config.json"), "r", encoding="utf-8") as fh:
    cfg = json.load(fh)
assert cfg["label_names"] == LABELS, "Label order mismatch with run config."

# Load predictions CSV
df = pd.read_csv(CSV_FT)

# Identify the test rows using saved IDs (robust to id/index differences)
test_ids_path = os.path.join(RUN_DIR, "test_ids.txt")
test_ids = [x.strip() for x in open(test_ids_path, "r", encoding="utf-8").read().splitlines() if x.strip()]

# Prefer matching on comment_id if present; else fall back to index
if "comment_id" in df.columns:
    df["comment_id"] = df["comment_id"].astype(str)
    dft = df[df["comment_id"].isin(test_ids)].copy()
else:
    df["_row_id"] = df.index.astype(str)
    dft = df[df["_row_id"].isin(test_ids)].copy()

print(f"Test rows found: {len(dft)}")

# --- MULTI-LABEL: metrics using tuned thresholds (already applied in *_ft_pred) ---
missing_pred_cols = [c for c in LABELS if f"{c}_ft_pred" not in dft.columns]
if missing_pred_cols:
    raise ValueError(f"Missing prediction columns in CSV: {missing_pred_cols}")

y_true = dft[LABELS].astype(int).values
y_pred = dft[[f"{c}_ft_pred" for c in LABELS]].astype(int).values

print("\n=== Multi-label (tuned thresholds) ===")
print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0, digits=3))
p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
print({"ml_f1_micro": round(f_micro, 3), "ml_f1_macro": round(f_macro, 3)})

# --- SENTIMENT: macro-F1 overall and on irony subset ---
if "sentiment" not in dft.columns or "sent_pred" not in dft.columns:
    raise ValueError("CSV must include 'sentiment' (gold) and 'sent_pred' (model).")

y_s_true = dft["sentiment"].astype(int).values   # -1/0/1 gold
y_s_pred = dft["sent_pred"].astype(int).values   # -1/0/1 model

sent_macro = f1_score(y_s_true, y_s_pred, average="macro")
print("\n=== Sentiment (macro-F1 overall) ===", round(sent_macro, 3))

mask_irony = dft["irony"].astype(int).values == 1
if mask_irony.any():
    sent_macro_irony = f1_score(y_s_true[mask_irony], y_s_pred[mask_irony], average="macro")
    print("Sentiment macro-F1 on irony subset:", round(sent_macro_irony, 3))
else:
    sent_macro_irony = None
    print("No irony positives in test; report N/A for subset.")

# --- Save a compact metrics file for the thesis repo ---
out = {
    "ml_f1_micro": float(f_micro),
    "ml_f1_macro": float(f_macro),
    "sent_macro_f1": float(sent_macro),
    "sent_macro_f1_irony": (None if sent_macro_irony is None else float(sent_macro_irony)),
    "num_test_rows": int(len(dft)),
}
with open(os.path.join(RUN_DIR, "test_metrics_from_csv.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("\nWrote:", os.path.join(RUN_DIR, "test_metrics_from_csv.json"))

# --- Quick supports table (helps interpret zeros) ---
supports = dft[LABELS].sum().astype(int)
print("\nLabel supports in TEST split:")
print(supports.to_string())
