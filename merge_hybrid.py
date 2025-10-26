import pandas as pd
import json
from pathlib import Path

# --- INPUTS (edit if your filenames differ) ---
BASE_CSV   = Path("comments_subsample_finetfinish_ft.csv")          # from BASE encoder run
IRONY_CSV  = Path("comments_subsample_finetfinish_ft_irony.csv")    # from IRONY encoder run
OUT_CSV    = Path("comments_subsample_finetfinish_ft_hybrid.csv")

ID_COL = "comment_id"
LABELS = ["Spam_flag","irony","distrust","trust","hostility",
          "concept_misinfo","believe_misinfo","Accuse_misinfo",
          "cogni_dis_manual","discrim_manual"]

# Strategy:
# - For 'irony' and all sentiment columns: take from IRONY_CSV
# - For all other labels: take from BASE_CSV

# ---------------------------------------------

db = pd.read_csv(BASE_CSV)
di = pd.read_csv(IRONY_CSV)

# Align on comment_id (fallback to index if needed)
if ID_COL in db.columns and ID_COL in di.columns:
    db[ID_COL] = db[ID_COL].astype(str)
    di[ID_COL] = di[ID_COL].astype(str)
    merged = db.merge(di, on=ID_COL, suffixes=("_base", "_irony"), how="inner")
else:
    db["_row_id"] = db.index.astype(str)
    di["_row_id"] = di.index.astype(str)
    merged = db.merge(di, on="_row_id", suffixes=("_base", "_irony"), how="inner")

# Start from BASE original columns
out = db.copy()

# Copy non-irony labels from BASE
for lab in LABELS:
    if lab == "irony":
        continue
    out[f"{lab}_ft_prob"] = merged[f"{lab}_ft_prob_base"]
    out[f"{lab}_ft_pred"] = merged[f"{lab}_ft_pred_base"]

# Copy IRONY label from IRONY model
out["irony_ft_prob"] = merged["irony_ft_prob_irony"]
out["irony_ft_pred"] = merged["irony_ft_pred_irony"]

# Copy SENTIMENT from IRONY model (better overall + on irony subset)
for col in ["sent_neg_prob","sent_neu_prob","sent_pos_prob","sent_pred"]:
    out[col] = merged[f"{col}_irony"]

# Optional: provenance metadata
meta = {
    "hybrid_sources": {
        "irony": "irony_model",
        "sentiment": "irony_model",
        "other_labels": "base_model"
    },
    "base_csv": str(BASE_CSV),
    "irony_csv": str(IRONY_CSV),
}
OUT_CSV.with_name("prediction_meta_hybrid.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

out.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV)
