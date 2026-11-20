import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import skew
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw" / "archive"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def extract_partition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Извлечение полного набора табличных и временных признаков для клиентов внутри партиции."""
    df = df.sort_values(["app_id", "transaction_number"]).reset_index(drop=True)
    g = df.groupby("app_id", sort=False)

    # 1. Базовые и квантильные статистики по сумме (amnt)
    feats = g["amnt"].agg(
        trx_count="count",
        amnt_sum="sum",
        amnt_mean="mean",
        amnt_std="std",
        amnt_min="min",
        amnt_max="max",
        amnt_median="median",
    )
    feats["amnt_std"] = feats["amnt_std"].fillna(0.0)
    feats["amnt_q10"] = g["amnt"].quantile(0.10)
    feats["amnt_q25"] = g["amnt"].quantile(0.25)
    feats["amnt_q75"] = g["amnt"].quantile(0.75)
    feats["amnt_q90"] = g["amnt"].quantile(0.90)
    feats["amnt_skew"] = g["amnt"].skew().fillna(0.0)

    # Производные отношения по суммам
    feats["amnt_iqr"] = feats["amnt_q75"] - feats["amnt_q25"]
    feats["amnt_range"] = feats["amnt_max"] - feats["amnt_min"]
    feats["amnt_cv"] = feats["amnt_std"] / (feats["amnt_mean"].abs() + 1e-5)
    feats["amnt_max_to_mean"] = feats["amnt_max"] / (feats["amnt_mean"].abs() + 1e-5)
    feats["amnt_median_to_mean"] = feats["amnt_median"] / (feats["amnt_mean"].abs() + 1e-5)

    # 2. Доходы vs Расходы (income_flag)
    if "income_flag" in df.columns:
        inc = df[df["income_flag"] == 1].groupby("app_id", sort=False)["amnt"]
        exp = df[df["income_flag"] == 0].groupby("app_id", sort=False)["amnt"]

        feats["inc_count"] = inc.count()
        feats["inc_sum"] = inc.sum()
        feats["inc_mean"] = inc.mean()
        feats["exp_count"] = exp.count()
        feats["exp_sum"] = exp.sum()
        feats["exp_mean"] = exp.mean()

        for c in ["inc_count", "inc_sum", "inc_mean", "exp_count", "exp_sum", "exp_mean"]:
            feats[c] = feats[c].fillna(0.0)

        feats["inc_count_ratio"] = feats["inc_count"] / (feats["trx_count"] + 1e-5)
        feats["inc_to_exp_count"] = feats["inc_count"] / (feats["exp_count"] + 1.0)
        feats["inc_to_exp_sum"] = feats["inc_sum"] / (feats["exp_sum"] + 1e-5)

    # 3. Разнообразие категорий (Nunique & Diversity)
    cat_cols = [
        "mcc", "mcc_category", "operation_type", "operation_type_group",
        "operation_kind", "card_type", "currency", "payment_system", "city", "country"
    ]
    for col in cat_cols:
        if col in df.columns:
            feats[f"{col}_nunique"] = g[col].nunique()
            feats[f"{col}_diversity"] = feats[f"{col}_nunique"] / (feats["trx_count"] + 1e-5)

    # 4. Флаги и доли (ecommerce, валюта, страна)
    if "ecommerce_flag" in df.columns:
        feats["ecom_sum"] = g["ecommerce_flag"].sum()
        feats["ecom_ratio"] = feats["ecom_sum"] / (feats["trx_count"] + 1e-5)

    if "currency" in df.columns:
        df["non_main_curr"] = (df["currency"] != 1).astype(np.float32)
        feats["non_main_curr_ratio"] = df.groupby("app_id", sort=False)["non_main_curr"].mean()

    if "country" in df.columns:
        df["non_main_country"] = (df["country"] != 1).astype(np.float32)
        feats["non_main_country_ratio"] = df.groupby("app_id", sort=False)["non_main_country"].mean()

    # 5. Временные окна (Recency & Velocity) по days_before
    if "days_before" in df.columns:
        feats["days_before_min"] = g["days_before"].min()
        feats["days_before_max"] = g["days_before"].max()
        feats["days_before_mean"] = g["days_before"].mean()
        feats["days_before_std"] = g["days_before"].std().fillna(0.0)
        feats["days_span"] = (feats["days_before_max"] - feats["days_before_min"]).clip(lower=1)
        feats["trx_per_day"] = feats["trx_count"] / feats["days_span"]
        feats["amnt_per_day"] = feats["amnt_sum"] / feats["days_span"]

        for window in (7, 14, 30, 60):
            sub = df[df["days_before"] <= window].groupby("app_id", sort=False)["amnt"]
            cnt_col = f"trx_cnt_last_{window}d"
            sum_col = f"amnt_sum_last_{window}d"
            mean_col = f"amnt_mean_last_{window}d"

            feats[cnt_col] = sub.count().reindex(feats.index, fill_value=0.0)
            feats[sum_col] = sub.sum().reindex(feats.index, fill_value=0.0)
            feats[mean_col] = sub.mean().reindex(feats.index, fill_value=0.0)

            feats[f"trx_ratio_last_{window}d"] = feats[cnt_col] / (feats["trx_count"] + 1e-5)
            feats[f"amnt_ratio_last_{window}d"] = feats[sum_col] / (feats["amnt_sum"].abs() + 1e-5)

        feats["velocity_14_to_60"] = (feats["trx_cnt_last_14d"] / 14.0) / (
            (feats["trx_cnt_last_60d"] - feats["trx_cnt_last_14d"]) / 46.0 + 1e-4
        )

    # 6. Интервалы между транзакциями (hour_diff) и время суток
    if "hour_diff" in df.columns:
        valid_hd = df[df["hour_diff"] >= 0]
        hd_g = valid_hd.groupby("app_id", sort=False)["hour_diff"]
        feats["hour_diff_mean"] = hd_g.mean().reindex(feats.index, fill_value=-1.0)
        feats["hour_diff_std"] = hd_g.std().reindex(feats.index, fill_value=-1.0)
        feats["hour_diff_median"] = hd_g.median().reindex(feats.index, fill_value=-1.0)
        feats["hour_diff_max"] = hd_g.max().reindex(feats.index, fill_value=-1.0)

        df["is_burst_trx"] = ((df["hour_diff"] >= 0) & (df["hour_diff"] <= 1)).astype(np.float32)
        feats["burst_trx_ratio"] = df.groupby("app_id", sort=False)["is_burst_trx"].mean()

    if "hour" in df.columns:
        df["is_night"] = df["hour"].isin((0, 1, 2, 3, 4, 5)).astype(np.float32)
        feats["night_trx_ratio"] = df.groupby("app_id", sort=False)["is_night"].mean()
        feats["hour_mean"] = g["hour"].mean()
        feats["hour_std"] = g["hour"].std().fillna(0.0)

    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin((6, 7)).astype(np.float32)
        feats["weekend_trx_ratio"] = df.groupby("app_id", sort=False)["is_weekend"].mean()

    # 7. Свежий тренд: последние 10 транзакций vs вся история
    last_10 = df.groupby("app_id", sort=False).tail(10)
    l10_g = last_10.groupby("app_id", sort=False)
    feats["last10_amnt_mean"] = l10_g["amnt"].mean()
    feats["last10_amnt_std"] = l10_g["amnt"].std().fillna(0.0)
    feats["last10_to_all_amnt_mean"] = feats["last10_amnt_mean"] / (feats["amnt_mean"].abs() + 1e-5)
    feats["last10_mcc_nunique"] = l10_g["mcc"].nunique()

    return feats.reset_index()


def collect_sequences(transactions_dir: Path, cols_to_collect: list[str]) -> dict[str, pd.Series]:
    """Быстро собирает текстовые последовательности кодов без медленного groupby.apply."""
    files = sorted(transactions_dir.rglob("*.parquet"))
    seq_chunks = {col: [] for col in cols_to_collect}

    for i, file in enumerate(files, 1):
        schema_cols = set(pq.read_schema(file).names)
        available = [c for c in cols_to_collect if c in schema_cols]
        if not available:
            continue

        read_cols = ["app_id", "transaction_number"] + available
        df = pd.read_parquet(file, columns=read_cols)
        df = df.sort_values(["app_id", "transaction_number"], kind="mergesort")

        app_ids = df["app_id"].to_numpy()
        change_idx = np.flatnonzero(app_ids[1:] != app_ids[:-1]) + 1
        split_indices = np.concatenate((np.array([0], dtype=np.int64), change_idx, np.array([len(app_ids)], dtype=np.int64)))
        unique_apps = app_ids[split_indices[:-1]]

        for col in available:
            vals = df[col].astype(str).tolist()
            joined = [
                " ".join(vals[split_indices[k] : split_indices[k + 1]])
                for k in range(len(unique_apps))
            ]
            seq_chunks[col].append(pd.Series(joined, index=unique_apps))

        if i % 25 == 0 or i == len(files):
            logger.info(f"  Сбор последовательностей ({transactions_dir.name}): {i}/{len(files)}")

    result = {}
    for col, chunks in seq_chunks.items():
        if chunks:
            combined = pd.concat(chunks)
            if not combined.index.is_unique:
                grouped_dict = {}
                for k, v in zip(combined.index, combined.values):
                    if k in grouped_dict:
                        grouped_dict[k] += " " + v
                    else:
                        grouped_dict[k] = v
                combined = pd.Series(grouped_dict)
            result[col] = combined
    return result


def add_tfidf_svd(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_seqs: dict[str, pd.Series],
    test_seqs: dict[str, pd.Series],
    configs: list[tuple[str, int, tuple[int, int]]],
):
    """Обучает TF-IDF + SVD строго на train и применяет к train и test."""
    new_train_parts = [train_df]
    new_test_parts = [test_df]

    for col, n_comp, ngram in configs:
        if col not in train_seqs or col not in test_seqs:
            logger.warning(f"Последовательность для '{col}' не найдена, пропускаем.")
            continue

        logger.info(f"TF-IDF + SVD ({n_comp} комп., ngram={ngram}) по '{col}'...")
        tr_corpus = train_df["app_id"].map(train_seqs[col]).fillna("empty")
        te_corpus = test_df["app_id"].map(test_seqs[col]).fillna("empty")

        vec = TfidfVectorizer(
            ngram_range=ngram,
            min_df=5,
            max_features=15000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        svd = TruncatedSVD(n_components=n_comp, random_state=42)

        tr_svd = svd.fit_transform(vec.fit_transform(tr_corpus))
        te_svd = svd.transform(vec.transform(te_corpus))

        col_names = [f"{col}_tfidf_svd_{i}" for i in range(n_comp)]
        new_train_parts.append(pd.DataFrame(tr_svd.astype(np.float32), columns=col_names, index=train_df.index))
        new_test_parts.append(pd.DataFrame(te_svd.astype(np.float32), columns=col_names, index=test_df.index))

    return pd.concat(new_train_parts, axis=1), pd.concat(new_test_parts, axis=1)


def process_all_partitions(transactions_dir: Path) -> pd.DataFrame:
    files = sorted(transactions_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Parquet файлы не найдены в {transactions_dir}")

    logger.info(f"Обработка {len(files)} партиций из {transactions_dir.name}...")
    parts = []

    for i, file in enumerate(files, 1):
        df = pd.read_parquet(file)
        drop_meta = [c for c in df.columns if c.startswith("__")]
        if drop_meta:
            df = df.drop(columns=drop_meta)

        part_feats = extract_partition_features(df)
        parts.append(part_feats)

        if i % 10 == 0 or i == len(files):
            logger.info(f"  Прогресс: {i}/{len(files)}")

    full_feats = pd.concat(parts, ignore_index=True)

    if not full_feats["app_id"].is_unique:
        logger.warning("Обнаружены пересечения app_id между партициями, агрегируем...")
        full_feats = full_feats.groupby("app_id", as_index=False).mean()

    for c in full_feats.select_dtypes(include=["float64"]).columns:
        full_feats[c] = full_feats[c].astype(np.float32)

    return full_feats


def main():
    train_dir = RAW_DIR / "alfabattle2_train_transactions_contest"
    test_dir = RAW_DIR / "alfabattle2_test_transactions_contest"

    logger.info("=== ЭТАП 1: Табличные и временные фичи (Train) ===")
    train_feats = process_all_partitions(train_dir)
    train_target = pd.read_csv(RAW_DIR / "alfabattle2_train_target.csv")
    train_data = train_target.merge(train_feats, on="app_id", how="left")

    logger.info("=== ЭТАП 2: Табличные и временные фичи (Test) ===")
    test_feats = process_all_partitions(test_dir)
    test_target = pd.read_csv(RAW_DIR / "alfabattle2_test_target_contest.csv")
    test_data = test_target.merge(test_feats, on="app_id", how="left")

    logger.info("=== ЭТАП 3: Последовательные эмбеддинги TF-IDF + SVD ===")
    seq_cols = ["mcc", "operation_type", "mcc_category"]
    train_seqs = collect_sequences(train_dir, seq_cols)
    test_seqs = collect_sequences(test_dir, seq_cols)

    tfidf_configs = [
        ("mcc", 24, (1, 2)),
        ("operation_type", 8, (1, 2)),
        ("mcc_category", 8, (1, 2)),
    ]
    train_data, test_data = add_tfidf_svd(train_data, test_data, train_seqs, test_seqs, tfidf_configs)

    for old_csv in ["train_v2.csv", "test_v2.csv"]:
        old_path = PROCESSED_DIR / old_csv
        if old_path.exists():
            old_path.unlink()

    train_data.to_parquet(PROCESSED_DIR / "train_v2.parquet", index=False)
    test_data.to_parquet(PROCESSED_DIR / "test_v2.parquet", index=False)
    logger.info(f"Готово! Train v2: {train_data.shape}, Test v2: {test_data.shape}")


if __name__ == "__main__":
    main()
