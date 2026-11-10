"""
v03: CatBoost Model.
"""
import warnings
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from v01_baseline_gbdt.train import (
    create_features, build_dataset, N_SPLITS, SEED, SMOOTHING,
    _fit_cat_risk_stats, _client_mcc_risk
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = PROJECT_ROOT / "archive" / "alfabattle2_train_target.csv"
TRX_DIR = PROJECT_ROOT / "archive" / "alfabattle2_train_transactions_contest" / "train_transactions_contest"
ARTIFACTS_V2 = PROJECT_ROOT / "artifacts" / "v02_dl"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "v03_cb"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("[1/3] Загрузка данных и признаков...")
    target = pd.read_csv(TARGET_PATH)
    trx_files = sorted(TRX_DIR.glob("*.parquet"))
    features, all_mcc = create_features(trx_files)

    dl_oof = pd.read_parquet(ARTIFACTS_V2 / "oof_dl.parquet")[["app_id", "dl_oof_pred"]]
    features = features.merge(dl_oof, on="app_id", how="left")
    features["dl_oof_pred"] = features["dl_oof_pred"].fillna(features["dl_oof_pred"].median())

    ds, feat_cols, y = build_dataset(target, features)
    X, ds_app_ids = ds[feat_cols], ds["app_id"]

    print("[2/3] Кросс-валидация (CatBoost)...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype=np.float32)
    fold_aucs = []
    
    y_map = pd.Series(y.values, index=ds_app_ids.values)
    global_mean = float(y.mean())
    X_features = X.drop(columns=["product"], errors="ignore")
    best_model = None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_features, y), 1):
        print(f"  фолд {fold}/{N_SPLITS}")
        tr_app_ids = ds_app_ids.iloc[tr_idx].values
        va_app_ids = ds_app_ids.iloc[va_idx].values

        tr_mcc = all_mcc[all_mcc["app_id"].isin(tr_app_ids)]
        tr_mcc_full = tr_mcc.merge(pd.DataFrame({'app_id': ds_app_ids, 'flag': y}), on='app_id', how='left')
        mcc_stats = tr_mcc_full.groupby('mcc_category').agg(sum_flag=('flag', 'sum'), count=('app_id', 'count')).reset_index()
        mcc_stats['mcc_category_risk'] = (mcc_stats['sum_flag'] + global_mean * SMOOTHING) / (mcc_stats['count'] + SMOOTHING)
        
        tr_risk = _client_mcc_risk(all_mcc[all_mcc["app_id"].isin(tr_app_ids)], mcc_stats[['mcc_category', 'mcc_category_risk']], global_mean)
        va_risk = _client_mcc_risk(all_mcc[all_mcc["app_id"].isin(va_app_ids)], mcc_stats[['mcc_category', 'mcc_category_risk']], global_mean)

        tr_product = pd.DataFrame({'app_id': ds_app_ids.iloc[tr_idx], 'product': X['product'].iloc[tr_idx]})
        product_stats = _fit_cat_risk_stats(tr_product, y_map, 'product', global_mean)
        
        tr_prod_risk = X['product'].iloc[tr_idx].map(product_stats.set_index('product')['product_risk']).fillna(global_mean).values
        va_prod_risk = X['product'].iloc[va_idx].map(product_stats.set_index('product')['product_risk']).fillna(global_mean).values

        X_tr, X_va = X_features.iloc[tr_idx].copy(), X_features.iloc[va_idx].copy()
        
        for col in ["mcc_risk_mean", "mcc_risk_max", "mcc_risk_min", "mcc_risk_std"]:
            X_tr[col] = ds_app_ids.iloc[tr_idx].map(tr_risk[col]).fillna(global_mean).values
            X_va[col] = ds_app_ids.iloc[va_idx].map(va_risk[col]).fillna(global_mean).values
            
        X_tr["product_risk"] = tr_prod_risk
        X_va["product_risk"] = va_prod_risk

        model = cb.CatBoostClassifier(
            iterations=2500, learning_rate=0.03, depth=7, l2_leaf_reg=5,
            eval_metric='AUC', early_stopping_rounds=100, random_seed=SEED,
            verbose=0, scale_pos_weight=35,
        )

        model.fit(X_tr, y.iloc[tr_idx], eval_set=(X_va, y.iloc[va_idx]), use_best_model=True)
        
        pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred
        auc = roc_auc_score(y.iloc[va_idx], pred)
        fold_aucs.append(auc)
        best_model = model 

    print(f"\n  CV AUC (CatBoost): {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")
    print(f"  OOF AUC (CatBoost): {roc_auc_score(y, oof):.5f}")

    pd.DataFrame({"app_id": ds_app_ids, "cb_oof_pred": oof, "flag": y}).to_parquet(ARTIFACTS_DIR / "oof_cb.parquet", index=False)
    best_model.save_model(str(ARTIFACTS_DIR / "catboost_model.cbm"))
    print(f"[3/3] Сохранено в {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()