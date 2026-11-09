"""
Alfa Battle 2.0 — предсказание вероятности дефолта клиента.

Пайплайн:
    1. Чтение parquet-файлов с транзакциями пофайлово (экономия памяти).
    2. Генерация признаков: суммы, счётчики, окна 7/30/90 дней, тренды,
       MCC-разнообразие, риск по MCC-категориям.
    3. 5-фолдовая StratifiedKFold кросс-валидация с out-of-fold target
       encoding (без утечки данных).
    4. Финальная модель LightGBM на 100% данных.

Метрика: ROC AUC.
Результат: CV AUC ≈ 0.7517 ± 0.0037 (OOF без утечек).

Артефакты сохраняются в ``artifacts/``:
    * lgb_model.txt           — обученный бустер LightGBM
    * features.parquet        — агрегированные признаки по клиентам
    * oof.parquet             — out-of-fold предсказания
    * feature_importance.csv  — важность признаков (gain)
"""

import gc
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

TARGET_PATH = Path("archive/alfabattle2_train_target.csv")
TRX_DIR = Path("archive/alfabattle2_train_transactions_contest/train_transactions_contest")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

WINDOWS = (7, 30, 90)
TOP_MCC_K = 5
N_SPLITS = 5
SEED = 42
SMOOTHING = 10

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "scale_pos_weight": 35,
    "feature_pre_filter": False,
    "verbose": -1,
    "random_state": SEED,
    "n_jobs": -1,
}


# ============================================================
# ГЕНЕРАЦИЯ ПРИЗНАКОВ — АГРЕГАЦИЯ ПО ФАЙЛУ
# ============================================================

def _aggregate_one_file(trx: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет аддитивные агрегаты по клиенту для одного parquet-файла.

    Все возвращаемые колонки сконструированы так, чтобы их можно было
    безопасно объединить между файлами через ``sum`` / ``max`` / ``min``
    (см. :func:`_combine_files`).

    Параметры
    ----------
    trx : pd.DataFrame
        Сырые транзакции с колонками ``app_id, amnt, hour, day_of_week,
        days_before, ecommerce_flag, mcc_category``.

    Возвращает
    -------
    pd.DataFrame
        Индексирован по ``app_id``. Содержит суммы, счётчики, окна
        (7/30/90 дней), а также структурные признаки (unique_days,
        счётчики по конкретным MCC).
    """
    trx = trx.copy()

    trx["is_night"] = ((trx["hour"] >= 22) | (trx["hour"] <= 6)).astype(np.int8)
    trx["is_weekend"] = trx["day_of_week"].isin([5, 6]).astype(np.int8)
    trx["is_ecommerce"] = (trx["ecommerce_flag"] == 1).astype(np.int8)
    trx["amnt_sq"] = trx["amnt"].astype(np.float64) ** 2

    base = trx.groupby("app_id").agg(
        total_amnt=("amnt", "sum"),
        count_trx=("amnt", "count"),
        max_amnt=("amnt", "max"),
        sum_sq_amnt=("amnt_sq", "sum"),
        recency_days=("days_before", "min"),
        unique_days=("days_before", "nunique"),
        night_count=("is_night", "sum"),
        weekend_count=("is_weekend", "sum"),
        ecommerce_count=("is_ecommerce", "sum"),
        mcc_2_count=("mcc_category", lambda x: (x == 2).sum()),
        mcc_4_count=("mcc_category", lambda x: (x == 4).sum()),
    )

    for w in WINDOWS:
        base[f"amnt_sum_{w}d"] = 0.0
        base[f"count_{w}d"] = 0
        sub = trx.loc[trx["days_before"] <= w]
        if not sub.empty:
            wagg = sub.groupby("app_id").agg(
                **{f"amnt_sum_{w}d": ("amnt", "sum"),
                   f"count_{w}d":    ("amnt", "count")}
            )
            base.loc[wagg.index, f"amnt_sum_{w}d"] = wagg[f"amnt_sum_{w}d"]
            base.loc[wagg.index, f"count_{w}d"]    = wagg[f"count_{w}d"]

    return base


def _combine_files(per_file: list[pd.DataFrame]) -> pd.DataFrame:
    """Объединяет пофайловые агрегаты в одну таблицу по клиенту.

    Использует ``sum`` для аддитивных величин, ``max`` для максимумов и
    количества уникальных дней (во избежание двойного счёта при
    пересечении дней между файлами), ``min`` для recency.

    Параметры
    ----------
    per_file : list[pd.DataFrame]
        Результаты :func:`_aggregate_one_file`, все индексированы по ``app_id``.

    Возвращает
    -------
    pd.DataFrame
        Единый фрейм агрегатов по клиенту, индексирован по ``app_id``.
    """
    combined = pd.concat(per_file, axis=0)
    agg_dict = {}
    for col in combined.columns:
        if col == "max_amnt":
            agg_dict[col] = "max"
        elif col == "recency_days":
            agg_dict[col] = "min"
        elif col == "unique_days":
            agg_dict[col] = "max"
        else:
            agg_dict[col] = "sum"
    return combined.groupby(level="app_id").agg(agg_dict)


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки: средние, std (через параллельную формулу
    дисперсии), доли, тренды.

    Стандартное отклонение считается как ``sqrt(E[X²] − E[X]²)`` — это
    корректно при пофайловой агрегации (обычное ``std`` по кускам не
    усредняется).

    Параметры
    ----------
    df : pd.DataFrame
        Объединённые агрегаты из :func:`_combine_files`.

    Возвращает
    -------
    pd.DataFrame
        Тот же фрейм с добавленными производными колонками.
    """
    n = df["count_trx"].clip(lower=1)
    df["amnt_mean"] = df["total_amnt"] / n

    var = df["sum_sq_amnt"] / n - df["amnt_mean"] ** 2
    df["amnt_std"] = np.sqrt(var.clip(lower=0))

    for w in WINDOWS:
        cnt = df[f"count_{w}d"].clip(lower=1)
        df[f"amnt_mean_{w}d"] = df[f"amnt_sum_{w}d"] / cnt

    df["ecommerce_share"] = df["ecommerce_count"] / n
    df["night_share"]     = df["night_count"]     / n
    df["weekend_share"]   = df["weekend_count"]   / n

    if {"count_30d", "count_90d"} <= set(df.columns):
        df["activity_trend_30d"] = df["count_30d"] / (df["count_90d"] / 3).clip(lower=1)
    if {"amnt_mean_30d", "amnt_mean"} <= set(df.columns):
        df["ticket_trend_30d"] = df["amnt_mean_30d"] / df["amnt_mean"].clip(lower=1)

    df["spend_per_day"] = df["total_amnt"] / df["unique_days"].clip(lower=1)
    return df


def _compute_mcc_features(all_mcc: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет признаки разнообразия MCC: количество уникальных, энтропию,
    доли топ-K категорий.

    Энтропия — скалярная мера разнообразия: высокая у клиентов, чьи траты
    распределены по многим категориям, низкая — у клиентов с концентрацией
    в одной-двух категориях.

    Параметры
    ----------
    all_mcc : pd.DataFrame
        Long-формат с колонками ``app_id, mcc_category, cnt``.

    Возвращает
    -------
    pd.DataFrame
        Индексирован по ``app_id``, колонки: ``unique_mcc, mcc_entropy,
        top_mcc_share_1..K``.
    """
    df = all_mcc.copy()
    df["total"] = df.groupby("app_id")["cnt"].transform("sum")
    df["prob"] = df["cnt"] / df["total"]
    df["entropy_contrib"] = -df["prob"] * np.log(df["prob"] + 1e-12)

    agg = df.groupby("app_id").agg(
        unique_mcc=("mcc_category", "nunique"),
        mcc_entropy=("entropy_contrib", "sum"),
    )

    df = df.sort_values(["app_id", "cnt"], ascending=[True, False])
    df["rank"] = df.groupby("app_id").cumcount() + 1
    top = df[df["rank"] <= TOP_MCC_K].copy()
    top["share"] = top["cnt"] / top["total"]

    pivot = top.pivot_table(
        index="app_id", columns="rank", values="share", fill_value=0.0
    )
    pivot.columns = [f"top_mcc_share_{int(c)}" for c in pivot.columns]

    return agg.join(pivot, how="left").fillna(0.0)


def create_features(trx_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Запускает полный пайплайн генерации признаков по всем parquet-файлам.

    Файлы обрабатываются по одному, чтобы ограничить пиковое потребление
    памяти; пофайловые агрегаты объединяются в конце.

    Параметры
    ----------
    trx_files : list[Path]
        Пути к ``*.parquet`` файлам транзакций.

    Возвращает
    -------
    features_df : pd.DataFrame
        Матрица признаков по клиентам (одна строка на ``app_id``).
    all_mcc : pd.DataFrame
        Long-формат ``(app_id, mcc_category, cnt)`` — нужен позже для
        OOF target encoding.
    """
    per_file_agg, per_file_mcc = [], []
    t_start = time.time()

    for i, path in enumerate(trx_files):
        trx = pd.read_parquet(path)
        per_file_agg.append(_aggregate_one_file(trx))
        per_file_mcc.append(
            trx.groupby(["app_id", "mcc_category"]).size().reset_index(name="cnt")
        )
        del trx

        if (i + 1) % 10 == 0 or (i + 1) == len(trx_files):
            elapsed = (time.time() - t_start) / 60
            print(f"  файлов: {i+1}/{len(trx_files)}  ({elapsed:.1f} мин)")
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


# ============================================================
# СБОРКА ДАТАСЕТА
# ============================================================

def build_dataset(
    target_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Мержит признаки с таргетом, добавляет флаги наличия транзакций.

    Параметры
    ----------
    target_df : pd.DataFrame
        Таргет с колонками ``app_id, product, flag``.
    features_df : pd.DataFrame
        Результат :func:`create_features`.

    Возвращает
    -------
    ds : pd.DataFrame
        Объединённый датасет с ``app_id``, ``flag`` и всеми признаками.
    feat_cols : list[str]
        Канонический упорядоченный список имён признаков.
    y : pd.Series
        Бинарный таргет, выровненный по ``ds``.
    """
    ds = target_df.merge(features_df, on="app_id", how="left")

    ds["has_transactions"] = ds["count_trx"].fillna(0).gt(0).astype(np.int8)
    for w in WINDOWS:
        if f"count_{w}d" in ds.columns:
            ds[f"has_{w}d"] = ds[f"count_{w}d"].fillna(0).gt(0).astype(np.int8)

    if "product" in ds.columns:
        ds["product_encoded"] = ds["product"].astype("category").cat.codes

    ds = ds.replace([np.inf, -np.inf], np.nan).fillna(0)

    drop_cols = {"app_id", "product", "flag"}
    feat_cols = [c for c in ds.columns if c not in drop_cols]

    return ds, feat_cols, ds["flag"]


# ============================================================
# ОБУЧЕНИЕ — CV С OOF TARGET ENCODING
# ============================================================

def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    all_mcc: pd.DataFrame,
    ds_app_ids: pd.Series,
) -> tuple[list[lgb.Booster], np.ndarray, list[int]]:
    """Stratified K-fold CV с target encoding без утечки данных.

    Для каждого фолда маппинг ``MCC → риск`` обучается **только на
    тренировочной части** и применяется к валидационной. Это исключает
    попадание таргета валидации в признак через кодирование.

    Параметры
    ----------
    X : pd.DataFrame
        Матрица признаков (одна строка на клиента), без колонки ``app_id``.
    y : pd.Series
        Бинарный таргет, выровненный по ``X``.
    all_mcc : pd.DataFrame
        Long-формат ``(app_id, mcc_category, cnt)``.
    ds_app_ids : pd.Series
        ``app_id``, выровненный по ``X`` (тот же индекс), используется
        для джойна кодирования.

    Возвращает
    -------
    models : list[lgb.Booster]
        Обученные бустеры (по одному на фолд).
    oof : np.ndarray
        Склеенные out-of-fold предсказания.
    best_iters : list[int]
        Лучшая итерация по каждому фолду (для расчёта длины финальной
        модели).
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype=np.float32)
    models, fold_aucs, best_iters = [], [], []

    global_mean = y.mean()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        print(f"  фолд {fold}/{N_SPLITS}")

        tr_app_ids = ds_app_ids.iloc[tr_idx]
        va_app_ids = ds_app_ids.iloc[va_idx]

        
        tr_mcc = all_mcc[all_mcc["app_id"].isin(tr_app_ids)].copy()
        tr_mcc = tr_mcc.merge(
            pd.DataFrame({"app_id": ds_app_ids, "flag": y}),
            on="app_id", how="left",
        )
        mcc_stats = tr_mcc.groupby("mcc_category").agg(
            sum_flag=("flag", "sum"),
            count=("app_id", "count"),
        ).reset_index()
        mcc_stats["mcc_risk"] = (
            (mcc_stats["sum_flag"] + global_mean * SMOOTHING)
            / (mcc_stats["count"] + SMOOTHING)
        )

        
        def _client_risk(app_ids_subset: pd.Series) -> pd.Series:
            temp = all_mcc[all_mcc["app_id"].isin(app_ids_subset)].copy()
            temp = temp.merge(
                mcc_stats[["mcc_category", "mcc_risk"]],
                on="mcc_category", how="left",
            )
            temp["mcc_risk"] = temp["mcc_risk"].fillna(global_mean)
            temp["weighted_risk"] = temp["mcc_risk"] * temp["cnt"]
            agg = temp.groupby("app_id").agg(
                weighted_risk=("weighted_risk", "sum"),
                total_cnt=("cnt", "sum"),
            )
            agg["mcc_risk_mean"] = agg["weighted_risk"] / agg["total_cnt"]
            return agg["mcc_risk_mean"]

        tr_risk_series = _client_risk(tr_app_ids)
        va_risk_series = _client_risk(va_app_ids)

        
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        X_tr["mcc_risk_mean"] = (
            ds_app_ids.iloc[tr_idx].map(tr_risk_series).fillna(global_mean).values
        )
        X_va["mcc_risk_mean"] = (
            ds_app_ids.iloc[va_idx].map(va_risk_series).fillna(global_mean).values
        )

        dtrain = lgb.Dataset(X_tr, y.iloc[tr_idx], params={"feature_pre_filter": False})
        dval = lgb.Dataset(X_va, y.iloc[va_idx], reference=dtrain)

        model = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
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
    return models, oof, best_iters


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """End-to-end пайплайн: признаки → CV → финальная модель → артефакты."""
    print("[1/6] загрузка таргета...")
    target = pd.read_csv(TARGET_PATH)
    print(f"      shape={target.shape}, default_rate={target['flag'].mean():.4f}")

    print("\n[2/6] генерация признаков...")
    trx_files = sorted(TRX_DIR.glob("*.parquet"))
    features, all_mcc = create_features(trx_files)
    features.to_parquet(ARTIFACTS_DIR / "features.parquet", index=False)

    print("\n[3/6] сборка датасета...")
    ds, feat_cols, y = build_dataset(target, features)
    X = ds[feat_cols]
    ds_app_ids = ds["app_id"]
    print(f"      X={X.shape}, n_features={len(feat_cols)}")

    print("\n[4/6] кросс-валидация (OOF target encoding)...")
    models, oof, best_iters = train_cv(X, y, all_mcc, ds_app_ids)

    print("\n[5/6] обучение финальной модели на 100% данных...")
    all_mcc_full = all_mcc.merge(
        pd.DataFrame({"app_id": ds_app_ids, "flag": y}),
        on="app_id", how="left",
    )
    mcc_stats_full = all_mcc_full.groupby("mcc_category").agg(
        sum_flag=("flag", "sum"),
        count=("app_id", "count"),
    ).reset_index()
    mcc_stats_full["mcc_risk"] = (
        (mcc_stats_full["sum_flag"] + y.mean() * SMOOTHING)
        / (mcc_stats_full["count"] + SMOOTHING)
    )

    all_mcc_full = all_mcc_full.merge(
        mcc_stats_full[["mcc_category", "mcc_risk"]],
        on="mcc_category", how="left",
    )
    all_mcc_full["weighted_risk"] = all_mcc_full["mcc_risk"] * all_mcc_full["cnt"]

    final_mcc_feat = all_mcc_full.groupby("app_id").agg(
        weighted_risk=("weighted_risk", "sum"),
        total_cnt=("cnt", "sum"),
    )
    final_mcc_feat["mcc_risk_mean"] = (
        final_mcc_feat["weighted_risk"] / final_mcc_feat["total_cnt"]
    )

    
    X_final = X.copy()
    X_final["mcc_risk_mean"] = (
        ds_app_ids.map(final_mcc_feat["mcc_risk_mean"]).fillna(y.mean()).values
    )

    final_rounds = int(np.mean(best_iters) * 1.1)
    dtrain_full = lgb.Dataset(X_final, y, params={"feature_pre_filter": False})
    final_model = lgb.train(LGB_PARAMS, dtrain_full, num_boost_round=final_rounds)

    print("\n[6/6] сохранение артефактов...")
    final_model.save_model(str(ARTIFACTS_DIR / "lgb_model.txt"))
    pd.DataFrame({
        "app_id": ds_app_ids, "oof_pred": oof, "flag": y,
    }).to_parquet(ARTIFACTS_DIR / "oof.parquet", index=False)

    imp = pd.DataFrame({
        "feature": final_model.feature_name(),
        "gain":    final_model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    imp.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)

    print("\nтоп-15 признаков по gain:")
    print(imp.head(15).to_string(index=False))
    print(f"\nготово. артефакты в {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()