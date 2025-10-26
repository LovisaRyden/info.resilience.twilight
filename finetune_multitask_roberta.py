# ==========================================
# File: finetune_multitask_roberta_complement.py
# Purpose: Train a multi-task RoBERTa with
#   - Head A: multi-label (sigmoid) for binary labels
#   - Head B: 3-class sentiment (softmax)
# Aligned with the old pipeline (shared preprocessing & length policy),
# saves per-label thresholds (tuned on validation) for the multi-label head,
# logs sentiment metrics also on the irony subset.
# ==========================================

import os, re, json, random, math
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score

from transformers import (
    AutoTokenizer,
)
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers import AutoConfig

# ----------------------------
# CONFIG — adjust as needed
# ----------------------------
CSV_PATH = "comments_subsample_finetfinish.csv"
TEXT_COL_RAW = "text"
TITLE_COL = "video_title"
ID_COL = "comment_id"
SENTIMENT_COL = "sentiment"   # values in {-1, 0, 1}

# Canonical label names (binary, 0/1). Sentiment is trained via a separate softmax head.
BINARY_LABEL_COLUMNS = [
    "Spam_flag", "irony", "distrust", "trust", "hostility",
    "concept_misinfo", "believe_misinfo", "Accuse_misinfo",
    "cogni_dis_manual", "discrim_manual"
]

# Preprocessing flags mirroring old pipeline
USE_TITLE_PREFIX = True          # prepend [Video title: ...]
CLEAN_LIKE_OLD = True            # lowercase, strip URLs/@handles, collapse whitespace
MAX_LENGTH = 320                 # align with long-text policy (reduced truncation)
BASE_MODEL = "cardiffnlp/twitter-roberta-base-irony"  # irony-tuned encoder  # robust for short social text

# Training settings
BATCH_SIZE = 8
EPOCHS = 3  # slightly shorter run; pair with lower LR
LEARNING_RATE = 2e-5  # lowered for stability when warm-starting / more conservative fine-tuning
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42
REPORT_STEPS = 50

# Multi-task loss weights
LAMBDA_SENT = 1.0
LAMBDA_MULTI = 0.7
# Extra weight for sentiment loss on ironic samples (1.0 = no extra, 0.5 => +50%)
IRONY_SENT_EXTRA = 0.5  # sentiment loss is multiplied by (1 + IRONY_SENT_EXTRA) for irony==1

OUTPUT_DIR = f"./finetuned_multitask_roberta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize column names that varied in prior scripts."""
    rename_map = {}
    if "Cogni_dis_manual" in df.columns and "cogni_dis_manual" not in df.columns:
        rename_map["Cogni_dis_manual"] = "cogni_dis_manual"
    if "discrim_ manual" in df.columns and "discrim_manual" not in df.columns:
        rename_map["discrim_ manual"] = "discrim_manual"
    return df.rename(columns=rename_map)

_url = re.compile(r"https?://\S+|www\.\S+")
_at  = re.compile(r"@\w+")
_ws  = re.compile(r"\s+")


def do_old_cleaning(t: str) -> str:
    """Minimal, safe cleaning aligned with earlier pipeline."""
    if not isinstance(t, str):
        t = "" if t is None else str(t)
    t = t.lower()
    t = _url.sub(" ", t)
    t = _at.sub(" ", t)
    t = _ws.sub(" ", t).strip()
    return t


def preprocess_row(row: pd.Series, use_title_prefix: bool = USE_TITLE_PREFIX, clean_like_old: bool = CLEAN_LIKE_OLD) -> str:
    t = row.get(TEXT_COL_RAW, "")
    if clean_like_old:
        t = do_old_cleaning(t)
    if use_title_prefix:
        title = row.get(TITLE_COL, "")
        if isinstance(title, str) and title.strip():
            t = f"[Video title: {title}] " + t
    return t


def build_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    missing = [c for c in BINARY_LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected binary label columns: {missing}")
    y_multi = df[BINARY_LABEL_COLUMNS].astype(float).fillna(0.0).values  # (N, L)

    if SENTIMENT_COL not in df.columns:
        raise ValueError(f"Sentiment column '{SENTIMENT_COL}' not found.")
    # Map -1,0,1 -> 0,1,2 (neg, neu, pos); fill NaN as neutral conservatively
    sent = df[SENTIMENT_COL].astype(float).fillna(0.0)
    sent_map = { -1.0: 0, 0.0: 1, 1.0: 2 }
    y_sent = sent.map(sent_map).astype(int).values  # (N,)

    return y_multi, y_sent, list(BINARY_LABEL_COLUMNS)


def split_train_val_test(df: pd.DataFrame, y_multi: np.ndarray, y_sent: np.ndarray,
                         val_size: float = 0.15, test_size: float = 0.15, seed: int = SEED):
    # Multi-label stratification is non-trivial; proxy uses both tasks.
    # Combine (any positive in multi) with sentiment class into 6 buckets
    any_pos = (y_multi.sum(axis=1) > 0).astype(int)
    strat_proxy = (any_pos * 3 + y_sent).astype(int)

    df_tr, df_tmp, yM_tr, yM_tmp, yS_tr, yS_tmp, sp_tr, sp_tmp = train_test_split(
        df, y_multi, y_sent, strat_proxy,
        test_size=(val_size + test_size), random_state=seed, stratify=strat_proxy
    )
    rel_test = test_size / (val_size + test_size)
    df_va, df_te, yM_va, yM_te, yS_va, yS_te = train_test_split(
        df_tmp, yM_tmp, yS_tmp, test_size=rel_test, random_state=seed, stratify=sp_tmp
    )
    return (df_tr, yM_tr, yS_tr), (df_va, yM_va, yS_va), (df_te, yM_te, yS_te)


class CommentDataset(Dataset):
    def __init__(self, texts: List[str], y_multi: np.ndarray, y_sent: np.ndarray, irony_idx: int, tokenizer, max_len: int):
        self.texts = texts
        self.y_multi = y_multi
        self.y_sent = y_sent
        self.irony_idx = irony_idx
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.texts[idx] if self.texts[idx] is not None else ""
        enc = self.tok(
            t,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels_multi"] = torch.tensor(self.y_multi[idx], dtype=torch.float)
        item["labels_sent"]  = torch.tensor(self.y_sent[idx], dtype=torch.long)
        # irony label for weighting sentiment loss
        item["irony_label"]   = torch.tensor(self.y_multi[idx, self.irony_idx], dtype=torch.float)
        return item


class MultiTaskRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        hidden = config.hidden_size
        dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)

        # Heads
        self.num_labels_multi = getattr(config, "num_labels_multi", 10)
        self.num_labels_sent  = getattr(config, "num_labels_sent", 3)

        self.classifier_multi = nn.Linear(hidden, self.num_labels_multi)
        self.classifier_sent  = nn.Linear(hidden, self.num_labels_sent)

        # Loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='none')   # we'll reduce manually
        self.ce  = nn.CrossEntropyLoss(reduction='none')    # per-sample

        # Loss weights
        self.lambda_sent = getattr(config, "lambda_sent", LAMBDA_SENT)
        self.lambda_multi = getattr(config, "lambda_multi", LAMBDA_MULTI)
        self.irony_sent_extra = getattr(config, "irony_sent_extra", IRONY_SENT_EXTRA)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None,
                labels_multi=None, labels_sent=None, irony_label=None, **kwargs):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **{k:v for k,v in kwargs.items() if k in ["token_type_ids"]})
        cls = out.last_hidden_state[:, 0, :]  # [CLS]
        cls = self.dropout(cls)

        logits_multi = self.classifier_multi(cls)
        logits_sent  = self.classifier_sent(cls)

        loss = None
        if labels_multi is not None and labels_sent is not None:
            # Multi-label loss
            multi_per_elem = self.bce(logits_multi, labels_multi)  # (B, L)
            multi_loss = multi_per_elem.mean(dim=1).mean()         # mean over labels then batch

            # Sentiment loss with extra weight on ironic samples
            sent_per_sample = self.ce(logits_sent, labels_sent)    # (B,)
            if irony_label is not None:
                weights = 1.0 + self.irony_sent_extra * irony_label
                sent_loss = (sent_per_sample * weights).mean()
            else:
                sent_loss = sent_per_sample.mean()

            loss = self.lambda_multi * multi_loss + self.lambda_sent * sent_loss

        return {
            "loss": loss,
            "logits": (logits_multi, logits_sent),  # tuple for Trainer.predict
        }


# Index of the irony label in BINARY_LABEL_COLUMNS
IRONY_IDX = BINARY_LABEL_COLUMNS.index("irony")


def compute_metrics_builder(label_names: List[str]):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # unpack
        if isinstance(preds, (tuple, list)):
            logits_multi, logits_sent = preds
        else:
            # if concatenated, split (not expected)
            raise ValueError("Predictions are not a tuple; unexpected format.")

        # labels may be dict or tuple depending on HF version
        if isinstance(labels, dict):
            y_multi = labels["labels_multi"]
            y_sent  = labels["labels_sent"]
        elif isinstance(labels, (tuple, list)):
            y_multi, y_sent = labels
        else:
            raise ValueError("Labels format not supported.")

        # Multi-label metrics at 0.5
        probs_multi = 1 / (1 + np.exp(-logits_multi))
        preds_multi = (probs_multi >= 0.5).astype(int)
        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
            y_multi, preds_multi, average="micro", zero_division=0
        )
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_multi, preds_multi, average="macro", zero_division=0
        )

        # Sentiment metrics (overall + irony subset)
        probs_sent = torch.softmax(torch.tensor(logits_sent), dim=1).numpy()
        preds_sent = probs_sent.argmax(axis=1)
        prec_s, rec_s, f1_s, _ = precision_recall_fscore_support(
            y_sent, preds_sent, average="macro", zero_division=0
        )
        # irony subset
        irony_mask = y_multi[:, IRONY_IDX] == 1
        if irony_mask.sum() > 0:
            prec_si, rec_si, f1_si, _ = precision_recall_fscore_support(
                y_sent[irony_mask], preds_sent[irony_mask], average="macro", zero_division=0
            )
        else:
            f1_si = 0.0

        return {
            "ml_precision_micro": prec_micro,
            "ml_recall_micro": rec_micro,
            "ml_f1_micro": f1_micro,
            "ml_precision_macro": prec_macro,
            "ml_recall_macro": rec_macro,
            "ml_f1_macro": f1_macro,
            "sent_macro_f1": f1_s,
            "sent_macro_f1_irony": f1_si,
        }
    return compute_metrics


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> List[float]:
    """Grid-search per-label thresholds to maximize F1 on validation."""
    thresholds = []
    grid = np.linspace(0.05, 0.95, 19)
    for j in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in grid:
            pred = (y_prob[:, j] >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true[:, j], pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(float(best_t))
    return thresholds


# ----------------------------
# Main
# ----------------------------

def to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels_multi=batch["labels_multi"],
            labels_sent=batch["labels_sent"],
            irony_label=batch["irony_label"],
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))

def predict_dataloader(model, loader, device):
    model.eval()
    logits_multi_list, logits_sent_list = [], []
    y_multi_list, y_sent_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            lm, ls = out["logits"]
            logits_multi_list.append(lm.cpu().numpy())
            logits_sent_list.append(ls.cpu().numpy())
            y_multi_list.append(batch["labels_multi"].cpu().numpy())
            y_sent_list.append(batch["labels_sent"].cpu().numpy())
    import numpy as np
    return (
        np.vstack(logits_multi_list),
        np.vstack(logits_sent_list),
        np.vstack(y_multi_list),
        np.hstack(y_sent_list),
    )

def train_main():
    set_seed(SEED)
    safe_mkdir(OUTPUT_DIR)

    print("Loading data…")
    df = pd.read_csv(CSV_PATH)
    df = normalize_columns(df)

    print("Preprocessing text (mirrors old pipeline)…")
    df["text_proc"] = df.apply(preprocess_row, axis=1)

    print("Building labels…")
    y_multi, y_sent, label_names = build_labels(df)

    # Save class frequencies for Methods
    class_freq = {name: float(df[name].sum()) for name in label_names}
    with open(os.path.join(OUTPUT_DIR, "class_frequencies.json"), "w", encoding="utf-8") as fh:
        json.dump(class_freq, fh, indent=2, ensure_ascii=False)

    # Split
    (df_tr, yM_tr, yS_tr), (df_va, yM_va, yS_va), (df_te, yM_te, yS_te) = split_train_val_test(df, y_multi, y_sent)
    print(f"Train/Val/Test: {len(df_tr)}/{len(df_va)}/{len(df_te)}")

    # Save split IDs for reproducibility
    def save_ids(name, frame):
        p = os.path.join(OUTPUT_DIR, f"{name}_ids.txt")
        with open(p, "w", encoding="utf-8") as fh:
            for x in frame.get(ID_COL, frame.index).tolist():
                fh.write(str(x) + "\n")
    save_ids("train", df_tr)
    save_ids("val", df_va)
    save_ids("test", df_te)

    # Tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    ds_train = CommentDataset(df_tr["text_proc"].tolist(), yM_tr, yS_tr, IRONY_IDX, tokenizer, MAX_LENGTH)
    ds_val   = CommentDataset(df_va["text_proc"].tolist(), yM_va, yS_va, IRONY_IDX, tokenizer, MAX_LENGTH)
    ds_test  = CommentDataset(df_te["text_proc"].tolist(), yM_te, yS_te, IRONY_IDX, tokenizer, MAX_LENGTH)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

    # Model config and init
    base_cfg = AutoConfig.from_pretrained(BASE_MODEL)
    base_cfg.num_labels_multi = len(label_names)
    base_cfg.num_labels_sent  = 3
    base_cfg.lambda_sent      = LAMBDA_SENT
    base_cfg.lambda_multi     = LAMBDA_MULTI
    base_cfg.irony_sent_extra = IRONY_SENT_EXTRA

    model = MultiTaskRoberta.from_pretrained(BASE_MODEL, config=base_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("Training…")
    for epoch in range(EPOCHS):
        loss_epoch = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {loss_epoch:.4f}")
    print("Training complete.")

    # Evaluate on test (default thresholds for reporting)
    print("Evaluating on test set (default thresholds)…")
    logits_multi_te, logits_sent_te, yM_te, yS_te = predict_dataloader(model, test_loader, device)
    yM_te_probs = 1 / (1 + np.exp(-logits_multi_te))
    yS_te_probs = torch.softmax(torch.tensor(logits_sent_te), dim=1).numpy()
    preds_multi_default = (yM_te_probs >= 0.5).astype(int)
    tm = {
        "ml_test_f1_micro_default": precision_recall_fscore_support(yM_te, preds_multi_default, average="micro", zero_division=0)[2],
        "ml_test_f1_macro_default": precision_recall_fscore_support(yM_te, preds_multi_default, average="macro", zero_division=0)[2],
        "sent_test_macro_f1": precision_recall_fscore_support(yS_te, yS_te_probs.argmax(1), average="macro", zero_division=0)[2],
    }
    with open(os.path.join(OUTPUT_DIR, "test_metrics_default.json"), "w") as fh:
        json.dump(tm, fh, indent=2)

    # Tune per-label thresholds on validation
    print("Tuning per-label thresholds on validation…")
    logits_multi_va, logits_sent_va, yM_va, yS_va = predict_dataloader(model, val_loader, device)
    yM_va_probs = 1 / (1 + np.exp(-logits_multi_va))
    thresholds = tune_thresholds(yM_va, yM_va_probs)
    print("Per-label thresholds:", dict(zip(label_names, thresholds)))

    # Save model + config
    print("Saving model and artifacts…")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    label_cfg = {
        "label_names": label_names,
        "thresholds": thresholds,
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "preprocessing": {
            "use_title_prefix": USE_TITLE_PREFIX,
            "clean_like_old": CLEAN_LIKE_OLD
        },
        "seed": SEED,
        "trained_on": CSV_PATH,
        "loss_weights": {"lambda_sent": LAMBDA_SENT, "lambda_multi": LAMBDA_MULTI, "irony_sent_extra": IRONY_SENT_EXTRA},
        "model_type": "MultiTaskRoberta",
        "timestamp": datetime.now().isoformat(timespec='seconds')
    }
    with open(os.path.join(OUTPUT_DIR, "label_config.json"), "w", encoding="utf-8") as fh:
        json.dump(label_cfg, fh, indent=2, ensure_ascii=False)

    # Extra: classification report on test using tuned thresholds (appendix-ready)
    preds_multi_tuned = (yM_te_probs >= np.array(thresholds)[None, :]).astype(int)
    rep = classification_report(yM_te, preds_multi_tuned, target_names=label_names, zero_division=0, digits=3)
    with open(os.path.join(OUTPUT_DIR, "test_report_multilabel_tuned.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep)

    print("\n----- Test report (multi-label, tuned thresholds) -----\n", rep)
    print("Done. Artifacts in:", OUTPUT_DIR)





# ==========================================
# File: predict_multitask_complement.py
# Purpose: Inference for the multi-task model. Mirrors old preprocessing,
# supports sliding-window via tokenizer overflow, and appends columns:
#   - For each binary label: <label>_ft_prob, <label>_ft_pred
#   - For sentiment: sent_neg_prob, sent_neu_prob, sent_pos_prob, sent_pred (-1/0/1)
# ==========================================

import os, re, json, argparse
import numpy as np
import pandas as pd
import torch
from typing import List
from transformers import AutoTokenizer, AutoConfig
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

# Match training preprocessing
_url = re.compile(r"https?://\S+|www\.\S+")
_at  = re.compile(r"@\w+")
_ws  = re.compile(r"\s+")

def do_old_cleaning(t: str) -> str:
    if not isinstance(t, str):
        t = "" if t is None else str(t)
    t = t.lower()
    t = _url.sub(" ", t)
    t = _at.sub(" ", t)
    t = _ws.sub(" ", t).strip()
    return t


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "Cogni_dis_manual" in df.columns and "cogni_dis_manual" not in df.columns:
        rename_map["Cogni_dis_manual"] = "cogni_dis_manual"
    if "discrim_ manual" in df.columns and "discrim_manual" not in df.columns:
        rename_map["discrim_ manual"] = "discrim_manual"
    return df.rename(columns=rename_map)


def preprocess_text(text, title, use_title_prefix, clean_like_old):
    t = text
    if clean_like_old:
        t = do_old_cleaning(t)
    if use_title_prefix and isinstance(title, str) and title.strip():
        t = f"[Video title: {title}] " + t
    return t


# where the duplicate used to be

def load_model(model_dir: str):
    cfg = AutoConfig.from_pretrained(model_dir)
    # ensure attributes exist
    if not hasattr(cfg, "num_labels_multi"): cfg.num_labels_multi = 10
    if not hasattr(cfg, "num_labels_sent"):  cfg.num_labels_sent  = 3
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = MultiTaskRoberta.from_pretrained(model_dir, config=cfg)
    with open(os.path.join(model_dir, "label_config.json"), "r", encoding="utf-8") as fh:
        label_cfg = json.load(fh)
    return tok, mdl, label_cfg


def predict_rows(df: pd.DataFrame, model_dir: str, text_col="text", title_col="video_title",
                 slide=True, stride_frac=0.5, batch_size=32, device=None):
    tok, mdl, cfg = load_model(model_dir)
    mdl.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)

    use_title_prefix = cfg["preprocessing"]["use_title_prefix"]
    clean_like_old   = cfg["preprocessing"]["clean_like_old"]
    max_len = cfg.get("max_length", 320)
    label_names = cfg["label_names"]
    thresholds = cfg.get("thresholds", [0.5]*len(label_names))

    texts = [
        preprocess_text(df.iloc[i][text_col], df.iloc[i].get(title_col, ""),
                        use_title_prefix, clean_like_old)
        for i in range(len(df))
    ]

    probs_multi_all = np.zeros((len(df), len(label_names)), dtype="float32")
    probs_sent_all  = np.zeros((len(df), 3), dtype="float32")

    enc_kwargs = dict(
        truncation=True, max_length=max_len, padding=True,
        return_overflowing_tokens=True if slide else False,
        stride=int(max_len * stride_frac) if slide else 0,
        return_tensors="pt"
    )

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk_texts = texts[start:start+batch_size]
            enc = tok(chunk_texts, **enc_kwargs)
            if slide and "overflow_to_sample_mapping" in enc:
                mapping = enc["overflow_to_sample_mapping"].tolist()
            else:
                mapping = list(range(len(chunk_texts)))

            model_inputs = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
            outputs = mdl(**model_inputs)
            logits_multi, logits_sent = outputs["logits"]

            probs_multi = torch.sigmoid(logits_multi).cpu().numpy()
            probs_sent = torch.softmax(logits_sent, dim=1).cpu().numpy()

            buckets_multi, buckets_sent = {}, {}
            for i_local, p in zip(mapping, probs_multi):
                buckets_multi.setdefault(i_local, []).append(p)
            for i_local, p in zip(mapping, probs_sent):
                buckets_sent.setdefault(i_local, []).append(p)

            for i_local, plist in buckets_multi.items():
                i_global = start + i_local
                probs_multi_all[i_global, :] = np.mean(plist, axis=0)
            for i_local, plist in buckets_sent.items():
                i_global = start + i_local
                probs_sent_all[i_global, :] = np.mean(plist, axis=0)

    preds_multi = (probs_multi_all >= np.array(thresholds)[None, :]).astype(int)
    preds_sent_class = probs_sent_all.argmax(axis=1)
    # map 0/1/2 back to -1/0/1
    sent_map_back = {0: -1, 1: 0, 2: 1}
    preds_sent_signed = np.vectorize(sent_map_back.get)(preds_sent_class)

    return probs_multi_all, preds_multi, probs_sent_all, preds_sent_signed, label_names, thresholds


def predict_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--title_col", default="video_title")
    ap.add_argument("--slide", action="store_true", help="Enable sliding-window inference")
    ap.add_argument("--no-slide", dest="slide", action="store_false")
    ap.set_defaults(slide=True)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    df = normalize_columns(df)

    p_multi, y_multi, p_sent, y_sent, labels, thresholds = predict_rows(
        df, args.model_dir, text_col=args.text_col, title_col=args.title_col,
        slide=args.slide, batch_size=args.batch_size
    )

    # Append columns without touching existing ones
    for j, name in enumerate(labels):
        df[f"{name}_ft_prob"] = p_multi[:, j]
        df[f"{name}_ft_pred"] = y_multi[:, j]

    df["sent_neg_prob"] = p_sent[:, 0]
    df["sent_neu_prob"] = p_sent[:, 1]
    df["sent_pos_prob"] = p_sent[:, 2]
    df["sent_pred"] = y_sent

    # Metadata for auditability
    meta = {
        "model_dir": args.model_dir,
        "labels": labels,
        "thresholds": thresholds,
        "slide": args.slide,
    }
    meta_path = os.path.join(os.path.dirname(args.output_csv) or ".", "prediction_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    df.to_csv(args.output_csv, index=False)
    print("Wrote:", args.output_csv)


# --- Hybrid prediction combining two models (BASE + IRONY) ---

def predict_hybrid_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir_base", required=True)
    ap.add_argument("--model_dir_irony", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--title_col", default="video_title")
    ap.add_argument("--slide", action="store_true", help="Enable sliding-window inference")
    ap.add_argument("--no-slide", dest="slide", action="store_false")
    ap.set_defaults(slide=True)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    df = normalize_columns(df)

    # BASE predictions (all labels except irony)
    pM_b, yM_b, pS_b, yS_b, labels_b, thr_b = predict_rows(
        df, args.model_dir_base, text_col=args.text_col, title_col=args.title_col,
        slide=args.slide, batch_size=args.batch_size
    )

    # IRONY-model predictions (irony label + sentiment)
    pM_i, yM_i, pS_i, yS_i, labels_i, thr_i = predict_rows(
        df, args.model_dir_irony, text_col=args.text_col, title_col=args.title_col,
        slide=args.slide, batch_size=args.batch_size
    )

    assert labels_b == labels_i, "Label order mismatch between models"
    labels = labels_b

    # Start from BASE then override irony + sentiment with IRONY-model
    pM = pM_b.copy(); yM = yM_b.copy()
    try:
        j_irony = labels.index("irony")
    except ValueError:
        raise ValueError("'irony' not found in label list")

    pM[:, j_irony] = pM_i[:, j_irony]
    yM[:, j_irony] = yM_i[:, j_irony]

    # Sentiment entirely from IRONY model
    pS = pS_i
    yS = yS_i

    # Append merged outputs to df
    for j, name in enumerate(labels):
        df[f"{name}_ft_prob"] = pM[:, j]
        df[f"{name}_ft_pred"] = yM[:, j]

    df["sent_neg_prob"] = pS[:, 0]
    df["sent_neu_prob"] = pS[:, 1]
    df["sent_pos_prob"] = pS[:, 2]
    df["sent_pred"] = yS

    # Provenance metadata
    meta = {
        "mode": "hybrid",
        "model_dir_base": args.model_dir_base,
        "model_dir_irony": args.model_dir_irony,
        "labels": labels,
        "thresholds_base": thr_b,
        "thresholds_irony": thr_i,
        "slide": args.slide,
    }
    meta_path = os.path.join(os.path.dirname(args.output_csv) or ".", "prediction_meta_hybrid.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    df.to_csv(args.output_csv, index=False)
    print("Wrote:", args.output_csv)


if __name__ == "__main__":
    import sys
    argv = " ".join(sys.argv)
    # Hybrid mode takes precedence when both model dirs are supplied
    if "--model_dir_base" in argv and "--model_dir_irony" in argv:
        predict_hybrid_main()
    elif "--model_dir" in argv and "--input_csv" in argv and "--output_csv" in argv:
        predict_main()
    else:
        train_main()
