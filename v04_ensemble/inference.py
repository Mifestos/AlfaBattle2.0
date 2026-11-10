"""
v04: Inference (Генерация submission.csv).
"""
import json
import warnings
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from v01_baseline_gbdt.train import create_features, build_dataset, _client_mcc_risk
from v02_deep_learning.dataset import MultimodalDataset
from v02_deep_learning.model import MultimodalTransformer

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_TARGET_PATH = PROJECT_ROOT / "archive" / "alfabattle2_test_target_contest.csv"
TEST_TRX_DIR = PROJECT_ROOT / "archive" / "alfabattle2_test_transactions_contest" / "test_transactions_contest"

ARTIFACTS_V1 = PROJECT_ROOT / "artifacts" / "v01_lgb"
ARTIFACTS_V2 = PROJECT_ROOT / "artifacts" / "v02_dl"
ARTIFACTS_CB = PROJECT_ROOT / "artifacts" / "v03_cb"
FINAL_DIR = PROJECT_ROOT / "artifacts" / "v04_ensemble"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
MAX_LEN = 30
D_MODEL = 32
NUM_LAYERS = 1


def main():
    print("[1/4] Загрузка весов и маппингов...")
    with open(FINAL_DIR / "best_weights.json", "r") as f:
        weights = json.load(f)
    
    mcc_stats = pd.read_parquet(ARTIFACTS_V1 / "mcc_stats.parquet").rename(columns={"mcc_risk": "mcc_category_risk"})
    product_stats = pd.read_parquet(ARTIFACTS_V1 / "product_stats.parquet")
    with open(ARTIFACTS_V1 / "feat_cols.json", "r") as f:
        lgb_feat_cols = json.load(f)

    print("[2/4] Генерация признаков для теста...")
    test_target = pd.read_csv(TEST_TARGET_PATH)
    test_trx_files = sorted(TEST_TRX_DIR.glob("*.parquet"))
    
    features, all_mcc_test = create_features(test_trx_files)
    
  
    ds_dl, dl_feat_cols, _ = build_dataset(test_target, features)
    tabular_cols = [c for c in dl_feat_cols if c != "mcc_risk_mean"]
    tabular_df = ds_dl[["app_id"] + tabular_cols].copy()
    

    ds_lgb = ds_dl.copy()
    y_mean = 0.0276
    

    test_mcc_risk = _client_mcc_risk(all_mcc_test, mcc_stats, y_mean).reset_index()
    ds_lgb = ds_lgb.merge(test_mcc_risk, on="app_id", how="left")
    
    ds_lgb = ds_lgb.merge(product_stats, on="product", how="left")
    
    for col in ["mcc_risk_mean", "mcc_risk_max", "mcc_risk_min", "mcc_risk_std", "product_risk"]:
        ds_lgb[col] = ds_lgb[col].fillna(y_mean)

    print("[3/4] Предсказания моделей...")
    
 
    lgb_model = lgb.Booster(model_file=str(ARTIFACTS_V1 / "lgb_model.txt"))
    pred_lgb = lgb_model.predict(ds_lgb[lgb_feat_cols])


    dl_dataset = MultimodalDataset(test_trx_files, test_target, tabular_df, max_len=MAX_LEN)
    dl_loader = DataLoader(dl_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    dl_model = MultimodalTransformer(tab_input_dim=len(tabular_cols), d_model=D_MODEL, num_layers=NUM_LAYERS).to(DEVICE)
    dl_model.load_state_dict(torch.load(ARTIFACTS_V2 / "transformer_model.pth", map_location=DEVICE))
    dl_model.eval()
    
    pred_dl = []
    with torch.no_grad():
        for seq, tab, _, _, _ in dl_loader:
            seq, tab = seq.to(DEVICE), tab.to(DEVICE)
            logits, _ = dl_model(seq, tab)
            pred_dl.extend(torch.sigmoid(logits).cpu().numpy().flatten())
    pred_dl = np.array(pred_dl)


    ds_lgb["dl_oof_pred"] = pred_dl


    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(str(ARTIFACTS_CB / "catboost_model.cbm"))
    cb_feat_cols = cb_model.feature_names_
    pred_cb = cb_model.predict_proba(ds_lgb[cb_feat_cols])[:, 1]

    print("[4/4] Блендинг и сохранение...")
    final_pred = (
        weights["lgb_weight"] * pred_lgb +
        weights["cb_weight"] * pred_cb +
        weights["dl_weight"] * pred_dl
    )

    submission = pd.DataFrame({"app_id": ds_lgb["app_id"], "flag": final_pred})
    submission.to_csv(PROJECT_ROOT / "submission.csv", index=False)
    print(f"Готово! Файл submission.csv сохранен.")


if __name__ == "__main__":
    main()