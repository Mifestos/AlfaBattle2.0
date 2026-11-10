"""
Обучение мультимодального трансформера.
"""
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from v01_baseline_gbdt.train import create_features, build_dataset
from v02_deep_learning.dataset import MultimodalDataset
from v02_deep_learning.model import MultimodalTransformer

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = PROJECT_ROOT / "archive" / "alfabattle2_train_target.csv"
TRX_DIR = PROJECT_ROOT / "archive" / "alfabattle2_train_transactions_contest" / "train_transactions_contest"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "v02_dl"
ARTIFACTS_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
MAX_LEN = 30
D_MODEL = 32
NUM_LAYERS = 1
MAX_EPOCHS = 8
PATIENCE = 2
LR = 3e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 5.0
NUM_WORKERS = 0
USE_AMP = DEVICE.type == "cuda"
SAMPLE_SIZE = None

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, optimizer, crit_main, crit_aux, scaler):
    model.train()
    for seq, tab, y, aux_y, _ in loader:
        seq, tab, y, aux_y = seq.to(DEVICE), tab.to(DEVICE), y.to(DEVICE), aux_y.to(DEVICE)
        y, aux_y = y.unsqueeze(1), aux_y.unsqueeze(1)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits_main, logits_aux = model(seq, tab)
            loss = crit_main(logits_main, y) + 0.1 * crit_aux(logits_aux, aux_y)
            
        if not torch.isfinite(loss): continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        for param in model.parameters():
            if param.grad is not None: param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

@torch.no_grad()
def predict_all(model, dataset, batch_size=1024):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    all_preds = []
    for seq, tab, _, _, _ in loader:
        seq, tab = seq.to(DEVICE), tab.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits_main, _ = model(seq, tab)
        probs = torch.sigmoid(logits_main).cpu().float().numpy().flatten()
        all_preds.extend(np.nan_to_num(probs, nan=0.5))
    return np.array(all_preds, dtype=np.float32)

def main() -> None:
    set_seed(SEED)
    print(f"[INFO] Устройство: {DEVICE}, AMP: {USE_AMP}")

    target = pd.read_csv(TARGET_PATH)
    trx_files = sorted(TRX_DIR.glob("*.parquet"))

    print("[1/3] Генерация табличных признаков...")
    features, _ = create_features(trx_files)
    ds, feat_cols, _ = build_dataset(target, features)
    tabular_cols = [c for c in feat_cols if c != "mcc_risk_mean"]
    tabular_df = ds[["app_id"] + tabular_cols].copy()

    print(f"[2/3] Создание MultimodalDataset...")
    dataset = MultimodalDataset(trx_files, target, tabular_df, max_len=MAX_LEN)
    
    train_idx = np.arange(len(dataset))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    final_oof_preds = np.zeros(len(dataset), dtype=np.float32)
    preds_count = np.zeros(len(dataset), dtype=np.float32)

    print(f"[3/3] Кросс-валидация (Transformer)...")
    for fold, (tr_idx_sub, va_idx_sub) in enumerate(skf.split(train_idx, dataset.labels[train_idx]), 1):
        print(f"\n  Фолд {fold}/{N_SPLITS}")
        tr_idx, va_idx = train_idx[tr_idx_sub], train_idx[va_idx_sub]

        train_loader = DataLoader(Subset(dataset, tr_idx), batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
        val_loader = DataLoader(Subset(dataset, va_idx), batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=(DEVICE.type == "cuda"))

        model = MultimodalTransformer(tab_input_dim=len(tabular_cols), d_model=D_MODEL, nhead=4, num_layers=NUM_LAYERS, dropout=0.15).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        
        crit_main = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))
        crit_aux = nn.MSELoss()
        scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)

        best_auc, best_state, patience_counter = 0.0, deepcopy(model.state_dict()), 0

        for epoch in range(MAX_EPOCHS):
            train_epoch(model, train_loader, optimizer, crit_main, crit_aux, scaler)
            val_preds = predict_all(model, Subset(dataset, va_idx))
            auc = roc_auc_score(dataset.labels[va_idx], val_preds)
            scheduler.step()
            
            print(f"    Эпоха {epoch + 1}: val_AUC = {auc:.5f} (mean={val_preds.mean():.3f})")

            if auc > best_auc:
                best_auc, best_state, patience_counter = auc, deepcopy(model.state_dict()), 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"    Early stopping")
                    break

        print("    Генерация OOF для всех клиентов...")
        model.load_state_dict(best_state)
        fold_preds_all = predict_all(model, dataset)
        final_oof_preds += fold_preds_all
        preds_count += 1
        
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    final_oof_preds /= np.maximum(preds_count, 1)
    print(f"\nOOF AUC (на всей выборке): {roc_auc_score(dataset.labels, final_oof_preds):.5f}")

    pd.DataFrame({
        "app_id": dataset.app_ids, "dl_oof_pred": final_oof_preds, "flag": dataset.labels,
    }).to_parquet(ARTIFACTS_DIR / "oof_dl.parquet", index=False)
    
    torch.save(best_state, ARTIFACTS_DIR / "transformer_model.pth")
    print(f"Сохранено в {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()