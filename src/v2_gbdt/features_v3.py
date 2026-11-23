import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

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
    """Извлечение расширенного набора поведенческих, финансовых и временных признаков v3 без фрагментации."""
    df = df.sort_values(["app_id", "transaction_number"]).reset_index(drop=True)
    g = df.groupby("app_id", sort=False)

    # 1. Базовые статистики по сумме (amnt)
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

    c: dict[str, pd.Series] = {}
    trx_cnt = feats["trx_count"]
    amnt_sum_abs = feats["amnt_sum"].abs() + 1e-5
    amnt_mean_abs = feats["amnt_mean"].abs() + 1e-5

    c["amnt_q10"] = g["amnt"].quantile(0.10)
    c["amnt_q25"] = g["amnt"].quantile(0.25)
    c["amnt_q75"] = g["amnt"].quantile(0.75)
    c["amnt_q90"] = g["amnt"].quantile(0.90)
    c["amnt_skew"] = g["amnt"].skew().fillna(0.0)

    c["amnt_iqr"] = c["amnt_q75"] - c["amnt_q25"]
    c["amnt_range"] = feats["amnt_max"] - feats["amnt_min"]
    c["amnt_cv"] = feats["amnt_std"] / amnt_mean_abs
    c["amnt_max_to_mean"] = feats["amnt_max"] / amnt_mean_abs
    c["amnt_median_to_mean"] = feats["amnt_median"] / amnt_mean_abs
    c["amnt_q90_to_q10"] = c["amnt_q90"] / (c["amnt_q10"].abs() + 1e-5)

    # 2. Денежный поток: Поступления vs Списания (вся история + свежие окна)
    if "income_flag" in df.columns:
        inc_df = df[df["income_flag"] == 1]
        exp_df = df[df["income_flag"] == 0]

        inc_g = inc_df.groupby("app_id", sort=False)["amnt"]
        exp_g = exp_df.groupby("app_id", sort=False)["amnt"]

        c["inc_count"] = inc_g.count().reindex(feats.index, fill_value=0.0)
        c["inc_sum"] = inc_g.sum().reindex(feats.index, fill_value=0.0)
        c["inc_mean"] = inc_g.mean().reindex(feats.index, fill_value=0.0)
        c["inc_max"] = inc_g.max().reindex(feats.index, fill_value=0.0)

        c["exp_count"] = exp_g.count().reindex(feats.index, fill_value=0.0)
        c["exp_sum"] = exp_g.sum().reindex(feats.index, fill_value=0.0)
        c["exp_mean"] = exp_g.mean().reindex(feats.index, fill_value=0.0)
        c["exp_max"] = exp_g.max().reindex(feats.index, fill_value=0.0)

        c["net_cashflow_sum"] = c["inc_sum"] - c["exp_sum"]
        c["inc_count_ratio"] = c["inc_count"] / (trx_cnt + 1e-5)
        c["inc_to_exp_count"] = c["inc_count"] / (c["exp_count"] + 1.0)
        c["inc_to_exp_sum"] = c["inc_sum"] / (c["exp_sum"] + 1e-5)
        c["inc_to_exp_mean"] = c["inc_mean"] / (c["exp_mean"] + 1e-5)

        if "days_before" in df.columns:
            for w in (14, 30):
                inc_w = inc_df[inc_df["days_before"] <= w].groupby("app_id", sort=False)["amnt"]
                exp_w = exp_df[exp_df["days_before"] <= w].groupby("app_id", sort=False)["amnt"]

                c[f"inc_sum_{w}d"] = inc_w.sum().reindex(feats.index, fill_value=0.0)
                c[f"inc_cnt_{w}d"] = inc_w.count().reindex(feats.index, fill_value=0.0)
                c[f"exp_sum_{w}d"] = exp_w.sum().reindex(feats.index, fill_value=0.0)
                c[f"exp_cnt_{w}d"] = exp_w.count().reindex(feats.index, fill_value=0.0)

                c[f"net_flow_{w}d"] = c[f"inc_sum_{w}d"] - c[f"exp_sum_{w}d"]
                c[f"inc_to_exp_sum_{w}d"] = c[f"inc_sum_{w}d"] / (c[f"exp_sum_{w}d"] + 1e-5)

            c["exp_accel_14_to_30"] = (c["exp_sum_14d"] / 14.0) / (
                (c["exp_sum_30d"] - c["exp_sum_14d"]) / 16.0 + 1e-4
            )

    # 3. Разнообразие и концентрация категорий
    cat_cols = (
        "mcc", "mcc_category", "operation_type", "operation_type_group",
        "operation_kind", "card_type", "currency", "payment_system", "city", "country"
    )
    for col in cat_cols:
        if col in df.columns:
            nunique_s = g[col].nunique()
            c[f"{col}_nunique"] = nunique_s
            c[f"{col}_diversity"] = nunique_s / (trx_cnt + 1e-5)

    for col in ("mcc", "mcc_category", "operation_type"):
        if col in df.columns:
            top_counts = df.groupby(["app_id", col], sort=False).size().groupby(level=0, sort=False).max()
            c[f"{col}_top1_share"] = top_counts.reindex(feats.index, fill_value=0.0) / (trx_cnt + 1e-5)

    # 4. Флаги и доли (ecommerce, валюта, страна) + динамика за 30 дней
    if "ecommerce_flag" in df.columns:
        c["ecom_sum"] = g["ecommerce_flag"].sum()
        c["ecom_ratio"] = c["ecom_sum"] / (trx_cnt + 1e-5)
        if "days_before" in df.columns:
            ecom_30 = df[df["days_before"] <= 30].groupby("app_id", sort=False)["ecommerce_flag"].mean()
            c["ecom_ratio_30d"] = ecom_30.reindex(feats.index, fill_value=0.0)
            c["ecom_ratio_diff_30d"] = c["ecom_ratio_30d"] - c["ecom_ratio"]

    if "currency" in df.columns:
        df["non_main_curr"] = (df["currency"] != 1).astype(np.float32)
        c["non_main_curr_ratio"] = df.groupby("app_id", sort=False)["non_main_curr"].mean()

    if "country" in df.columns:
        df["non_main_country"] = (df["country"] != 1).astype(np.float32)
        c["non_main_country_ratio"] = df.groupby("app_id", sort=False)["non_main_country"].mean()

    # 5. Временные окна (Recency & Velocity) по days_before
    if "days_before" in df.columns:
        c["days_before_min"] = g["days_before"].min()
        c["days_before_max"] = g["days_before"].max()
        c["days_before_mean"] = g["days_before"].mean()
        c["days_before_std"] = g["days_before"].std().fillna(0.0)
        c["days_span"] = (c["days_before_max"] - c["days_before_min"]).clip(lower=1)
        c["trx_per_day"] = trx_cnt / c["days_span"]
        c["amnt_per_day"] = feats["amnt_sum"] / c["days_span"]

        c["days_before_median"] = g["days_before"].median()
        c["recency_center_ratio"] = (c["days_before_median"] - c["days_before_min"]) / c["days_span"]

        for window in (7, 14, 30, 60):
            sub = df[df["days_before"] <= window].groupby("app_id", sort=False)["amnt"]
            cnt_col = f"trx_cnt_last_{window}d"
            sum_col = f"amnt_sum_last_{window}d"
            mean_col = f"amnt_mean_last_{window}d"

            c[cnt_col] = sub.count().reindex(feats.index, fill_value=0.0)
            c[sum_col] = sub.sum().reindex(feats.index, fill_value=0.0)
            c[mean_col] = sub.mean().reindex(feats.index, fill_value=0.0)

            c[f"trx_ratio_last_{window}d"] = c[cnt_col] / (trx_cnt + 1e-5)
            c[f"amnt_ratio_last_{window}d"] = c[sum_col] / amnt_sum_abs
            c[f"amnt_mean_ratio_{window}d_to_all"] = c[mean_col] / amnt_mean_abs

        c["velocity_7_to_30"] = (c["trx_cnt_last_7d"] / 7.0) / (
            (c["trx_cnt_last_30d"] - c["trx_cnt_last_7d"]) / 23.0 + 1e-4
        )
        c["velocity_14_to_60"] = (c["trx_cnt_last_14d"] / 14.0) / (
            (c["trx_cnt_last_60d"] - c["trx_cnt_last_14d"]) / 46.0 + 1e-4
        )

    # 6. Ритм транзакций (hour_diff) и циркадные паттерны
    if "hour_diff" in df.columns:
        valid_hd = df[df["hour_diff"] >= 0]
        hd_g = valid_hd.groupby("app_id", sort=False)["hour_diff"]
        c["hour_diff_mean"] = hd_g.mean().reindex(feats.index, fill_value=-1.0)
        c["hour_diff_std"] = hd_g.std().reindex(feats.index, fill_value=-1.0)
        c["hour_diff_median"] = hd_g.median().reindex(feats.index, fill_value=-1.0)
        c["hour_diff_q75"] = hd_g.quantile(0.75).reindex(feats.index, fill_value=-1.0)
        c["hour_diff_max"] = hd_g.max().reindex(feats.index, fill_value=-1.0)
        c["hour_diff_cv"] = c["hour_diff_std"] / (c["hour_diff_mean"].abs() + 1e-4)

        df["is_burst_trx"] = ((df["hour_diff"] >= 0) & (df["hour_diff"] <= 1)).astype(np.float32)
        c["burst_trx_ratio"] = df.groupby("app_id", sort=False)["is_burst_trx"].mean()

        df["is_long_pause"] = (df["hour_diff"] >= 168).astype(np.float32)
        c["long_pause_ratio"] = df.groupby("app_id", sort=False)["is_long_pause"].mean()

    if "hour" in df.columns:
        df["is_night"] = df["hour"].isin((0, 1, 2, 3, 4, 5)).astype(np.float32)
        c["night_trx_ratio"] = df.groupby("app_id", sort=False)["is_night"].mean()
        c["hour_mean"] = g["hour"].mean()
        c["hour_std"] = g["hour"].std().fillna(0.0)

    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin((6, 7)).astype(np.float32)
        c["weekend_trx_ratio"] = df.groupby("app_id", sort=False)["is_weekend"].mean()

    # 7. Свежий поведенческий сдвиг: последние 10 и 25 транзакций vs вся история
    for k in (10, 25):
        last_k = df.groupby("app_id", sort=False).tail(k)
        lk_g = last_k.groupby("app_id", sort=False)
        c[f"last{k}_amnt_mean"] = lk_g["amnt"].mean()
        c[f"last{k}_amnt_std"] = lk_g["amnt"].std().fillna(0.0)
        c[f"last{k}_to_all_amnt_mean"] = c[f"last{k}_amnt_mean"] / amnt_mean_abs
        if "mcc" in df.columns:
            c[f"last{k}_mcc_nunique"] = lk_g["mcc"].nunique()
        if "income_flag" in df.columns:
            c[f"last{k}_inc_ratio"] = lk_g["income_flag"].mean()

    out = pd.concat([feats, pd.DataFrame(c, index=feats.index)], axis=1)
    return out.reset_index()


def collect_sequences(
    transactions_dir: Path,
    cols_to_collect: list[str],
    tail_n: int | None = None,
) -> dict[str, pd.Series]:
    """Собирает текстовые последовательности кодов (по всей истории или по последним tail_n операциям)."""
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

        if tail_n is not None:
            df = df.groupby("app_id", sort=False).tail(tail_n)

        app_ids = df["app_id"].to_numpy()
        change_idx = np.flatnonzero(app_ids[1:] != app_ids[:-1]) + 1
        split_indices = np.concatenate(
            (np.zeros(1, dtype=np.int64), change_idx, np.array([len(app_ids)], dtype=np.int64))
        )
        unique_apps = app_ids[split_indices[:-1]]

        for col in available:
            vals = df[col].astype(str).tolist()
            joined = [
                " ".join(vals[split_indices[k] : split_indices[k + 1]])
                for k in range(len(unique_apps))
            ]
            seq_chunks[col].append(pd.Series(joined, index=unique_apps))

        if i % 25 == 0 or i == len(files):
            tag = f"last_{tail_n}" if tail_n else "full"
            logger.info(f"  Сбор последовательностей [{tag}] ({transactions_dir.name}): {i}/{len(files)}")

    result = {}
    for col, chunks in seq_chunks.items():
        if chunks:
            combined = pd.concat(chunks)
            assert combined.index.is_unique, f"Обнаружены дубликаты app_id в последовательностях по {col}!"
            result[col] = combined
    return result


def add_tfidf_svd(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_seqs: dict[str, pd.Series],
    test_seqs: dict[str, pd.Series],
    configs: list[tuple[str, int, tuple[int, int]]],
    prefix: str = "",
):
    """Обучает TF-IDF + SVD строго на train и применяет к train и test без утечек."""
    new_train_parts = [train_df]
    new_test_parts = [test_df]

    for col, n_comp, ngram in configs:
        if col not in train_seqs or col not in test_seqs:
            logger.warning(f"Последовательность для '{col}' не найдена, пропускаем.")
            continue

        tag = f"{prefix}_{col}" if prefix else col
        logger.info(f"TF-IDF + SVD ({n_comp} комп., ngram={ngram}) по '{tag}'...")
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

        col_names = [f"{tag}_tfidf_svd_{i}" for i in range(n_comp)]
        new_train_parts.append(pd.DataFrame(tr_svd.astype(np.float32), columns=col_names, index=train_df.index))
        new_test_parts.append(pd.DataFrame(te_svd.astype(np.float32), columns=col_names, index=test_df.index))

    return pd.concat(new_train_parts, axis=1), pd.concat(new_test_parts, axis=1)


def add_product_group_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Относительные признаки клиента в разрезе кредитного продукта (статистики строго по train)."""
    if "product" not in train_df.columns:
        return train_df, test_df

    logger.info("Генерация групповых признаков относительно кредитного продукта (product)...")
    target_cols = [
        "amnt_sum", "amnt_mean", "trx_count", "days_before_min",
        "inc_to_exp_sum", "exp_sum_30d", "trx_per_day"
    ]
    target_cols = [c for c in target_cols if c in train_df.columns]

    tr_new = {}
    te_new = {}

    for col in target_cols:
        stats = train_df.groupby("product")[col].agg(["mean", "std"])
        tr_mean = train_df["product"].map(stats["mean"]).astype(np.float32)
        tr_std = train_df["product"].map(stats["std"]).fillna(1.0).replace(0, 1.0).astype(np.float32)

        te_mean = test_df["product"].map(stats["mean"]).astype(np.float32)
        te_std = test_df["product"].map(stats["std"]).fillna(1.0).replace(0, 1.0).astype(np.float32)

        tr_new[f"{col}_to_prod_mean"] = (train_df[col] / (tr_mean.abs() + 1e-5)).astype(np.float32)
        te_new[f"{col}_to_prod_mean"] = (test_df[col] / (te_mean.abs() + 1e-5)).astype(np.float32)

        tr_new[f"{col}_prod_zscore"] = ((train_df[col] - tr_mean) / tr_std).astype(np.float32)
        te_new[f"{col}_prod_zscore"] = ((test_df[col] - te_mean) / te_std).astype(np.float32)

    train_out = pd.concat([train_df, pd.DataFrame(tr_new, index=train_df.index)], axis=1)
    test_out = pd.concat([test_df, pd.DataFrame(te_new, index=test_df.index)], axis=1)
    return train_out, test_out


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
    assert full_feats["app_id"].is_unique, "КРИТИЧНО: обнаружены дубликаты app_id между партициями!"

    for c in full_feats.select_dtypes(include=["float64"]).columns:
        full_feats[c] = full_feats[c].astype(np.float32)

    return full_feats


def main():
    train_dir = RAW_DIR / "alfabattle2_train_transactions_contest"
    test_dir = RAW_DIR / "alfabattle2_test_transactions_contest"

    logger.info("=== ЭТАП 1: Расширенные табличные фичи v3 (Train) ===")
    train_feats = process_all_partitions(train_dir)
    train_target = pd.read_csv(RAW_DIR / "alfabattle2_train_target.csv")
    train_data = train_target.merge(train_feats, on="app_id", how="left")

    logger.info("=== ЭТАП 2: Расширенные табличные фичи v3 (Test) ===")
    test_feats = process_all_partitions(test_dir)
    test_target = pd.read_csv(RAW_DIR / "alfabattle2_test_target_contest.csv")
    test_data = test_target.merge(test_feats, on="app_id", how="left")

    logger.info("=== ЭТАП 3: Групповые признаки по кредитному продукту ===")
    train_data, test_data = add_product_group_features(train_data, test_data)

    logger.info("=== ЭТАП 4: TF-IDF + SVD по всей истории ===")
    seq_cols = ["mcc", "operation_type", "mcc_category"]
    train_seqs = collect_sequences(train_dir, seq_cols, tail_n=None)
    test_seqs = collect_sequences(test_dir, seq_cols, tail_n=None)

    tfidf_configs_full = [
        ("mcc", 24, (1, 2)),
        ("operation_type", 8, (1, 2)),
        ("mcc_category", 8, (1, 2)),
    ]
    train_data, test_data = add_tfidf_svd(train_data, test_data, train_seqs, test_seqs, tfidf_configs_full)

    logger.info("=== ЭТАП 5: Свежий TF-IDF + SVD по последним 30 транзакциям ===")
    train_seqs_recent = collect_sequences(train_dir, ["mcc", "operation_type"], tail_n=30)
    test_seqs_recent = collect_sequences(test_dir, ["mcc", "operation_type"], tail_n=30)

    tfidf_configs_recent = [
        ("mcc", 12, (1, 2)),
        ("operation_type", 6, (1, 2)),
    ]
    train_data, test_data = add_tfidf_svd(
        train_data, test_data, train_seqs_recent, test_seqs_recent, tfidf_configs_recent, prefix="recent30"
    )

    train_data.to_parquet(PROCESSED_DIR / "train_v3.parquet", index=False)
    test_data.to_parquet(PROCESSED_DIR / "test_v3.parquet", index=False)
    logger.info(f"Готово! Train v3: {train_data.shape}, Test v3: {test_data.shape}")


if __name__ == "__main__":
    main()
