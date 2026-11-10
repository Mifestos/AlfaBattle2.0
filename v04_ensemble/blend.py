"""
v04: Финальный Блендинг (3-Way Blending).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_V1 = PROJECT_ROOT / "artifacts" / "v01_lgb"
ARTIFACTS_V2 = PROJECT_ROOT / "artifacts" / "v02_dl"
ARTIFACTS_CB = PROJECT_ROOT / "artifacts" / "v03_cb"
FINAL_DIR = PROJECT_ROOT / "artifacts" / "v04_ensemble"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("[1/3] Загрузка OOF предсказаний...")
    lgb = pd.read_parquet(ARTIFACTS_V1 / "oof.parquet")[["app_id", "oof_pred", "flag"]]
    lgb.rename(columns={"oof_pred": "lgb_pred"}, inplace=True)
    
    dl = pd.read_parquet(ARTIFACTS_V2 / "oof_dl.parquet")[["app_id", "dl_oof_pred"]]
    cb = pd.read_parquet(ARTIFACTS_CB / "oof_cb.parquet")[["app_id", "cb_oof_pred"]]

    print("[2/3] Объединение данных и перевод в NumPy...")
    df = lgb.merge(dl, on="app_id").merge(cb, on="app_id")
    
  
    y_true = df["flag"].values
    lgb_preds = df["lgb_pred"].values
    dl_preds = df["dl_oof_pred"].values
    cb_preds = df["cb_oof_pred"].values

    print("[3/3] Поиск оптимальных весов...")
    
    best_auc = 0.0
    best_weights = (0.0, 0.0, 0.0)

   
    for w_lgb in np.arange(0.0, 1.01, 0.05):
        for w_cb in np.arange(0.0, 1.01 - w_lgb, 0.05):
            w_dl = 1.0 - w_lgb - w_cb
            blend = w_lgb * lgb_preds + w_dl * dl_preds + w_cb * cb_preds
            auc = roc_auc_score(y_true, blend)
            if auc > best_auc:
                best_auc = auc
                best_weights = (w_lgb, w_dl, w_cb)

   
    w_lgb_opt, w_dl_opt, w_cb_opt = best_weights
    for w_lgb in np.arange(max(0, w_lgb_opt - 0.05), min(1, w_lgb_opt + 0.05), 0.01):
        for w_cb in np.arange(max(0, w_cb_opt - 0.05), min(1 - w_lgb, w_cb_opt + 0.05), 0.01):
            w_dl = 1.0 - w_lgb - w_cb
            if w_dl < 0:
                continue
            blend = w_lgb * lgb_preds + w_dl * dl_preds + w_cb * cb_preds
            auc = roc_auc_score(y_true, blend)
            if auc > best_auc:
                best_auc = auc
                best_weights = (w_lgb, w_dl, w_cb)

    
    print("\n" + "=" * 50)
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ АНСАМБЛЯ")
    print("=" * 50)
    print(f"Вес LightGBM:    {best_weights[0]:.2f}")
    print(f"Вес Transformer: {best_weights[1]:.2f}")
    print(f"Вес CatBoost:    {best_weights[2]:.2f}")
    print(f"\nМаксимальный OOF AUC (3-Way Blend): {best_auc:.5f}")
    print("=" * 50)

    weights_dict = {
        "lgb_weight": float(best_weights[0]),
        "dl_weight": float(best_weights[1]),
        "cb_weight": float(best_weights[2]),
        "oof_auc": float(best_auc)
    }
    with open(FINAL_DIR / "best_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=4)
        
    df["final_pred"] = (
        best_weights[0] * df["lgb_pred"] + 
        best_weights[1] * df["dl_oof_pred"] + 
        best_weights[2] * df["cb_oof_pred"]
    )
    df[["app_id", "final_pred", "flag"]].to_parquet(FINAL_DIR / "final_oof.parquet", index=False)
    print(f"Веса и предсказания сохранены в {FINAL_DIR}")


if __name__ == "__main__":
    main()