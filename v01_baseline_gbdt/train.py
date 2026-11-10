"""
v01: Baseline GBDT.
LightGBM с OOF Target Encoding для MCC и Product.
"""
import gc
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = PROJECT_ROOT / "archive" / "alfabattle2_train_target.csv"
TRX_DIR = PROJECT_ROOT / "archive" / "alfabattle2_train_transactions_contest" / "train_transactions_contest"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "v01_lgb"
ARTIFACTS_DIR.mkdir(exist_ok=True)

WINDOWS = (7, 30, 90)
TOP_MCC_K = 5
N_SPLITS = 5
SEED = 42
SMOOTHING = 10
EARLY_STOPPING_ROUNDS = 150
FINAL_ITER_MULTIPLIER = 1.1

LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "num_leaves": 127, "min_data_in_leaf": 300, "learning_rate": 0.015,
    "feature_fraction": 0.65, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l1": 1.0, "lambda_l2": 1.0, "scale_pos_weight": 35,
    "feature_pre_filter": False, "verbose": -1, "random_state": SEED, "n_jobs": -1,
}

def _aggregate_one_file(trx: pd.DataFrame) -> pd.DataFrame:
    trx = trx.copy()
    trx["is_night"] = ((trx["hour"] >= 22) | (trx["hour"] <= 6)).astype(np.int8)
    trx["is_weekend"] = trx["day_of_week"].isin([5, 6]).astype(np.int8)
    trx["is_ecommerce"] = (trx["ecommerce_flag"] == 1).astype(np.int8)
    trx["amnt_sq"] = trx["amnt"].astype(np.float64) ** 2
    trx["is_large"] = (trx["amnt"] > 10000).astype(np.int8)

    base = trx.groupby("app_id").agg(
        total_amnt=("amnt", "sum"), count_trx=("amnt", "count"), max_amnt=("amnt", "max"),
        min_amnt=("amnt", "min"), sum_sq_amnt=("amnt_sq", "sum"), recency_days=("days_before", "min"),
        unique_days=("days_before", "nunique"), night_count=("is_night", "sum"),
        weekend_count=("is_weekend", "sum"), ecommerce_count=("is_ecommerce", "sum"),
        large_trx_count=("is_large", "sum"), hour_mean=("hour", "mean"), hour_std=("hour", "std"),
        mcc_1_count=("mcc_category", lambda x: (x == 1).sum()),
        mcc_2_count=("mcc_category", lambda x: (x == 2).sum()),
        mcc_4_count=("mcc_category", lambda x: (x == 4).sum()),
        mcc_5_count=("mcc_category", lambda x: (x == 5).sum()),
    )

    for w in WINDOWS:
        base[f"amnt_sum_{w}d"] = 0.0
        base[f"count_{w}d"] = 0
        base[f"sum_sq_amnt_{w}d"] = 0.0
        base[f"max_amnt_{w}d"] = 0.0
        sub = trx.loc[trx["days_before"] <= w]
        if not sub.empty:
            wagg = sub.groupby("app_id").agg(
                amnt_sum=("amnt", "sum"), count=("amnt", "count"),
                sum_sq=("amnt_sq", "sum"), max_val=("amnt", "max"),
            )
            base.loc[wagg.index, f"amnt_sum_{w}d"] = wagg["amnt_sum"]
            base.loc[wagg.index, f"count_{w}d"] = wagg["count"]
            base.loc[wagg.index, f"sum_sq_amnt_{w}d"] = wagg["sum_sq"]
            base.loc[wagg.index, f"max_amnt_{w}d"] = wagg["max_val"]
    return base

def _combine_files(per_file: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(per_file, axis=0)
    agg_dict = {}
    for col in combined.columns:
        if col.startswith("max_amnt") or col == "unique_days": agg_dict[col] = "max"
        elif col == "min_amnt" or col == "recency_days": agg_dict[col] = "min"
        else: agg_dict[col] = "sum"
    return combined.groupby(level="app_id").agg(agg_dict)

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    n = df["count_trx"].clip(lower=1)
    df["amnt_mean"] = df["total_amnt"] / n
    df["amnt_max_to_mean"] = df["max_amnt"] / df["amnt_mean"].clip(lower=1)
    var = df["sum_sq_amnt"] / n - df["amnt_mean"] ** 2
    df["amnt_std"] = np.sqrt(var.clip(lower=0))
    df["amnt_cv"] = df["amnt_std"] / df["amnt_mean"].clip(lower=1)

    for w in WINDOWS:
        cnt = df[f"count_{w}d"].clip(lower=1)
        df[f"amnt_mean_{w}d"] = df[f"amnt_sum_{w}d"] / cnt
        mean_w = df[f"amnt_mean_{w}d"]
        var_w = df[f"sum_sq_amnt_{w}d"] / cnt - mean_w ** 2
        df[f"amnt_std_{w}d"] = np.sqrt(var_w.clip(lower=0))
        if "total_amnt" in df.columns:
            df[f"share_{w}d"] = df[f"amnt_sum_{w}d"] / df["total_amnt"].clip(lower=1)

    df["ecommerce_share"] = df["ecommerce_count"] / n
    df["night_share"] = df["night_count"] / n
    df["weekend_share"] = df["weekend_count"] / n
    df["large_trx_share"] = df["large_trx_count"] / n

    if {"count_30d", "count_90d"} <= set(df.columns):
        df["activity_trend_30d"] = df["count_30d"] / (df["count_90d"] / 3).clip(lower=1)
    if {"amnt_mean_30d", "amnt_mean"} <= set(df.columns):
        df["ticket_trend_30d"] = df["amnt_mean_30d"] / df["amnt_mean"].clip(lower=1)

    df["spend_per_day"] = df["total_amnt"] / df["unique_days"].clip(lower=1)
    df["trx_per_day"] = df["count_trx"] / df["unique_days"].clip(lower=1)
    
    if {"amnt_sum_7d", "amnt_sum_30d"} <= set(df.columns):
        df["speed_of_spending"] = df["amnt_sum_7d"] / (df["amnt_sum_30d"] / 4 + 1)
    return df

def _compute_mcc_features(all_mcc: pd.DataFrame) -> pd.DataFrame:
    df = all_mcc.copy()
    df["total"] = df.groupby("app_id")["cnt"].transform("sum")
    df["prob"] = df["cnt"] / df["total"]
    df["entropy_contrib"] = -df["prob"] * np.log(df["prob"] + 1e-12)

    agg = df.groupby("app_id").agg(unique_mcc=("mcc_category", "nunique"), mcc_entropy=("entropy_contrib", "sum"))

    df = df.sort_values(["app_id", "cnt"], ascending=[True, False])
    df["rank"] = df.groupby("app_id").cumcount() + 1
    top = df[df["rank"] <= TOP_MCC_K].copy()
    top["share"] = top["cnt"] / top["total"]

    pivot = top.pivot_table(index="app_id", columns="rank", values="share", fill_value=0.0)
    pivot.columns = [f"top_mcc_share_{int(c)}" for c in pivot.columns]
    return agg.join(pivot, how="left").fillna(0.0)

def create_features(trx_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_file_agg, per_file_mcc = [], []
    t_start = time.time()
    for i, path in enumerate(trx_files):
        trx = pd.read_parquet(path)
        per_file_agg.append(_aggregate_one_file(trx))
        per_file_mcc.append(trx.groupby(["app_id", "mcc_category"]).size().reset_index(name="cnt"))
        del trx

        if (i + 1) % 10 == 0 or (i + 1) == len(trx_files):
            print(f"  файлов: {i+1}/{len(trx_files)} ({(time.time() - t_start)/60:.1f} мин)")
        if (i + 1) % 5 == 0:
            gc.collect()

    df = _combine_files(per_file_agg)
    df = _add_derived_features(df)
    all_mcc = pd.concat(per_file_mcc, ignore_index=True)
    all_mcc = all_mcc.groupby(["app_id", "mcc_category"], as_index=False)["cnt"].sum()

    mcc_feats = _compute_mcc_features(all_mcc)
    df = df.join(mcc_feats, how="left")
    df = df.replace([np.inf, -np.inf], np.nan).reset_index()
    return df, all_mcc

def build_dataset(target_df: pd.DataFrame, features_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.Series | None]:
    """Мержит признаки с таргетом и оставляет product для OOF TE."""
    ds = target_df.merge(features_df, on="app_id", how="left")
    
    if "product" not in ds.columns:
        ds["product"] = -1
        
    ds = ds.replace([np.inf, -np.inf], np.nan).fillna(0)
    drop_cols = {"app_id", "flag"} 
    feat_cols = [c for c in ds.columns if c not in drop_cols]
    
    y = ds["flag"] if "flag" in ds.columns else None
    return ds, feat_cols, y

def _fit_cat_risk_stats(train_df: pd.DataFrame, y_map: pd.Series, cat_col: str, global_mean: float) -> pd.DataFrame:
    tr = train_df.copy()
    tr["flag"] = tr["app_id"].map(y_map)
    stats = tr.groupby(cat_col).agg(sum_flag=("flag", "sum"), count=("app_id", "count")).reset_index()
    stats[f"{cat_col}_risk"] = (stats["sum_flag"] + global_mean * SMOOTHING) / (stats["count"] + SMOOTHING)
    return stats[[cat_col, f"{cat_col}_risk"]]

def _client_mcc_risk(all_mcc: pd.DataFrame, mcc_stats: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    merged = all_mcc.merge(mcc_stats, on="mcc_category", how="left")
    merged["mcc_category_risk"] = merged["mcc_category_risk"].fillna(global_mean)
    merged["weighted_risk"] = merged["mcc_category_risk"] * merged["cnt"]
    agg = merged.groupby("app_id", sort=False).agg(
        weighted_risk=("weighted_risk", "sum"), total_cnt=("cnt", "sum"),
        mcc_risk_max=("mcc_category_risk", "max"), mcc_risk_min=("mcc_category_risk", "min"),
        mcc_risk_std=("mcc_category_risk", "std"),
    )
    agg["mcc_risk_mean"] = agg["weighted_risk"] / agg["total_cnt"]
    return agg[["mcc_risk_mean", "mcc_risk_max", "mcc_risk_min", "mcc_risk_std"]]

def train_cv(X, y, all_mcc, ds_app_ids):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype=np.float32)
    models, fold_aucs, best_iters = [], [], []
    global_mean = float(y.mean())
    y_map = pd.Series(y.values, index=ds_app_ids.values)
    X_features = X.drop(columns=["product"], errors="ignore")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_features, y), 1):
        print(f"  фолд {fold}/{N_SPLITS}")
        tr_app_ids = ds_app_ids.iloc[tr_idx].values
        va_app_ids = ds_app_ids.iloc[va_idx].values

        tr_mcc = all_mcc[all_mcc["app_id"].isin(tr_app_ids)]
        tr_mcc_full = tr_mcc.merge(pd.DataFrame({'app_id': ds_app_ids, 'flag': y}), on='app_id', how='left')
        mcc_stats = tr_mcc_full.groupby('mcc_category').agg(sum_flag=('flag', 'sum'), count=('app_id', 'count')).reset_index()
        mcc_stats['mcc_category_risk'] = (mcc_stats['sum_flag'] + global_mean * SMOOTHING) / (mcc_stats['count'] + SMOOTHING)
        mcc_stats = mcc_stats[['mcc_category', 'mcc_category_risk']]

        tr_risk = _client_mcc_risk(all_mcc[all_mcc["app_id"].isin(tr_app_ids)], mcc_stats, global_mean)
        va_risk = _client_mcc_risk(all_mcc[all_mcc["app_id"].isin(va_app_ids)], mcc_stats, global_mean)

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

        dtrain = lgb.Dataset(X_tr, y.iloc[tr_idx], params={"feature_pre_filter": False})
        dval = lgb.Dataset(X_va, y.iloc[va_idx], reference=dtrain)

        model = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=0)]
        )

        pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof[va_idx] = pred
        auc = roc_auc_score(y.iloc[va_idx], pred)
        fold_aucs.append(auc)
        best_iters.append(model.best_iteration)
        models.append(model)
        print(f"    AUC={auc:.5f}  best_iter={model.best_iteration}")

    print(f"\n  CV AUC: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")
    print(f"  OOF AUC: {roc_auc_score(y, oof):.5f}")
    return models, oof, best_iters, X_features.columns.tolist()

def main() -> None:
    print("[1/6] загрузка таргета...")
    target = pd.read_csv(TARGET_PATH)
    print("\n[2/6] генерация признаков...")
    trx_files = sorted(TRX_DIR.glob("*.parquet"))
    features, all_mcc = create_features(trx_files)
    features.to_parquet(ARTIFACTS_DIR / "features.parquet", index=False)

    print("\n[3/6] сборка датасета...")
    ds, feat_cols, y = build_dataset(target, features)
    X, ds_app_ids = ds[feat_cols], ds["app_id"]
    print(f"      X={X.shape}, n_features={len(feat_cols)}")

    print("\n[4/6] кросс-валидация (OOF target encoding)...")
    models, oof, best_iters, actual_feat_cols = train_cv(X, y, all_mcc, ds_app_ids)
    
    print("\n[5/6] обучение финальной модели на 100% данных...")
    y_map = pd.Series(y.values, index=ds_app_ids.values)
    
    all_mcc_full = all_mcc.merge(pd.DataFrame({"app_id": ds_app_ids, "flag": y}), on="app_id", how="left")
    mcc_stats_full = all_mcc_full.groupby("mcc_category").agg(sum_flag=("flag", "sum"), count=("app_id", "count")).reset_index()
    mcc_stats_full["mcc_risk"] = (mcc_stats_full["sum_flag"] + y.mean() * SMOOTHING) / (mcc_stats_full["count"] + SMOOTHING)
    
    final_mcc_risk = _client_mcc_risk(all_mcc, mcc_stats_full[['mcc_category', 'mcc_risk']].rename(columns={'mcc_risk': 'mcc_category_risk'}), float(y.mean()))
    
    full_product = pd.DataFrame({'app_id': ds_app_ids, 'product': X['product']})
    product_stats_full = _fit_cat_risk_stats(full_product, y_map, 'product', float(y.mean()))

    X_final = X.drop(columns=["product"], errors="ignore").copy()
    for col in ["mcc_risk_mean", "mcc_risk_max", "mcc_risk_min", "mcc_risk_std"]:
        X_final[col] = ds_app_ids.map(final_mcc_risk[col]).fillna(y.mean()).values
    X_final["product_risk"] = X['product'].map(product_stats_full.set_index('product')['product_risk']).fillna(y.mean()).values

    final_rounds = int(np.mean(best_iters) * FINAL_ITER_MULTIPLIER)
    final_model = lgb.train(LGB_PARAMS, lgb.Dataset(X_final, y), num_boost_round=final_rounds)
    
    print("\n[6/6] сохранение артефактов...")
    final_model.save_model(str(ARTIFACTS_DIR / "lgb_model.txt"))
    pd.DataFrame({"app_id": ds_app_ids, "oof_pred": oof, "flag": y}).to_parquet(ARTIFACTS_DIR / "oof.parquet")
    
    with open(ARTIFACTS_DIR / "feat_cols.json", "w") as f:
        json.dump(actual_feat_cols + ["mcc_risk_mean", "mcc_risk_max", "mcc_risk_min", "mcc_risk_std", "product_risk"], f)
        
    mcc_stats_full[['mcc_category', 'mcc_risk']].to_parquet(ARTIFACTS_DIR / "mcc_stats.parquet", index=False)
    product_stats_full.to_parquet(ARTIFACTS_DIR / "product_stats.parquet", index=False)
        
    print(f"Готово. Артефакты в {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()