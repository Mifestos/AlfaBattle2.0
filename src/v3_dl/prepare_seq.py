import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw" / "archive"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 100

CAT_COLS = (
    "mcc",
    "mcc_category",
    "operation_type",
    "operation_type_group",
    "operation_kind",
    "card_type",
    "payment_system",
    "currency",
    "ecommerce_flag",
    "income_flag",
    "day_of_week",
    "hour",
)

NUM_COLS = ("amnt", "hour_diff", "days_before")


def extract_sequences(
    transactions_dir: Path,
    app_id_to_idx: dict,
    cat_seqs: np.ndarray,
    num_seqs: np.ndarray,
    seq_lens: np.ndarray,
    cat_maps: dict | None = None,
    is_train: bool = True,
) -> dict:
    files = sorted(transactions_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Parquet-файлы не найдены в {transactions_dir}")

    if is_train:
        cat_maps = {c: {} for c in CAT_COLS}

    read_cols = ["app_id", "transaction_number", *CAT_COLS, *NUM_COLS]

    for file in files:
        df = pd.read_parquet(file, columns=read_cols)
        df = df.sort_values(["app_id", "transaction_number"], kind="mergesort").reset_index(drop=True)

        for col in CAT_COLS:
            vals = df[col].fillna(-999).astype(np.int32).to_numpy()
            col_map = cat_maps[col]
            if is_train:
                for u in np.unique(vals):
                    u_int = int(u)
                    if u_int not in col_map:
                        col_map[u_int] = len(col_map) + 2
            df[col] = np.fromiter((col_map.get(int(v), 1) for v in vals), dtype=np.int16, count=len(vals))

        amnt = df["amnt"].fillna(0.0).to_numpy(dtype=np.float32)
        hd = np.log1p(np.clip(df["hour_diff"].fillna(0.0).to_numpy(dtype=np.float32), 0.0, None)) / 10.0
        db = df["days_before"].fillna(0.0).to_numpy(dtype=np.float32) / 365.0
        num_matrix = np.column_stack((amnt, hd, db)).astype(np.float16)
        cat_matrix = df[list(CAT_COLS)].to_numpy(dtype=np.int16)

        app_ids = df["app_id"].to_numpy()
        change_idx = np.flatnonzero(app_ids[1:] != app_ids[:-1]) + 1
        starts = np.concatenate((np.zeros(1, dtype=np.int64), change_idx))
        ends = np.concatenate((change_idx, np.full(1, len(app_ids), dtype=np.int64)))

        for s, e in zip(starts, ends):
            row_idx = app_id_to_idx.get(int(app_ids[s]))
            if row_idx is None:
                continue

            length = min(int(e - s), MAX_LEN)
            cat_seqs[row_idx, MAX_LEN - length :] = cat_matrix[e - length : e]
            num_seqs[row_idx, MAX_LEN - length :] = num_matrix[e - length : e]
            seq_lens[row_idx] = length

    return cat_maps


def main():
    logger.info(f"Параметры: max_len={MAX_LEN}, cat_cols={len(CAT_COLS)}, num_cols={len(NUM_COLS)}")

    train_target = pd.read_csv(RAW_DIR / "alfabattle2_train_target.csv")
    test_target = pd.read_csv(RAW_DIR / "alfabattle2_test_target_contest.csv")

    n_train, n_test = len(train_target), len(test_target)
    n_cats, n_nums = len(CAT_COLS), len(NUM_COLS)

    train_cat = np.zeros((n_train, MAX_LEN, n_cats), dtype=np.int16)
    train_num = np.zeros((n_train, MAX_LEN, n_nums), dtype=np.float16)
    train_len = np.zeros(n_train, dtype=np.int16)

    test_cat = np.zeros((n_test, MAX_LEN, n_cats), dtype=np.int16)
    test_num = np.zeros((n_test, MAX_LEN, n_nums), dtype=np.float16)
    test_len = np.zeros(n_test, dtype=np.int16)

    train_map = {int(aid): idx for idx, aid in enumerate(train_target["app_id"])}
    test_map = {int(aid): idx for idx, aid in enumerate(test_target["app_id"])}

    logger.info("Обработка последовательностей Train...")
    cat_maps = extract_sequences(
        RAW_DIR / "alfabattle2_train_transactions_contest",
        train_map, train_cat, train_num, train_len, is_train=True,
    )

    logger.info("Обработка последовательностей Test...")
    extract_sequences(
        RAW_DIR / "alfabattle2_test_transactions_contest",
        test_map, test_cat, test_num, test_len, cat_maps=cat_maps, is_train=False,
    )

    prod_vals = sorted(train_target["product"].fillna(-1).astype(int).unique())
    prod_map = {int(v): i + 1 for i, v in enumerate(prod_vals)}
    train_prod = train_target["product"].fillna(-1).astype(int).map(prod_map).fillna(0).to_numpy(dtype=np.int16)
    test_prod = test_target["product"].fillna(-1).astype(int).map(prod_map).fillna(0).to_numpy(dtype=np.int16)

    vocab_sizes = np.array([len(cat_maps[c]) + 2 for c in CAT_COLS], dtype=np.int32)
    out_path = PROCESSED_DIR / "seq_v3.npz"

    np.savez(
        out_path,
        train_cat=train_cat,
        train_num=train_num,
        train_len=train_len,
        train_prod=train_prod,
        y=train_target["flag"].astype(np.float32).to_numpy(),
        test_cat=test_cat,
        test_num=test_num,
        test_len=test_len,
        test_prod=test_prod,
        vocab_sizes=vocab_sizes,
        prod_vocab_size=len(prod_map) + 1,
    )
    logger.info(f"Сохранено: {out_path} (train={n_train}, test={n_test})")


if __name__ == "__main__":
    main()
