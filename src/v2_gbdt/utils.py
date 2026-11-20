import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PREDS_DIR = ROOT_DIR / "data" / "predictions"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

PREDS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

VERSION = "v2"
RANDOM_STATE = 42
N_FOLDS = 5


def load_data_and_splits(for_catboost: bool = False):
    """Загружает данные, готовит типы колонок и возвращает фиксированные индексы сплита 80/20 и 5 фолдов."""
    train_df = pd.read_parquet(PROCESSED_DIR / f"train_{VERSION}.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / f"test_{VERSION}.parquet")

    assert train_df["app_id"].is_unique, "В train обнаружены дубликаты app_id!"

    ignore_cols = {"app_id", "flag"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    X = train_df[feature_cols].copy()
    y = train_df["flag"].astype(int).values
    X_test = test_df[feature_cols].copy()

    cat_cols = [c for c in ["product"] if c in feature_cols]

    for col in cat_cols:
        if for_catboost:
            X[col] = X[col].fillna("MISSING").astype(str)
            X_test[col] = X_test[col].fillna("MISSING").astype(str)
        else:
            X[col] = X[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    indices = np.arange(len(train_df))
    train_idx, holdout_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(train_idx, y[train_idx]))

    return X, y, X_test, test_df["app_id"], train_idx, holdout_idx, folds, cat_cols


def evaluate_and_save(
    model_name: str,
    y: np.ndarray,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    oof_train: np.ndarray,
    holdout_pred: np.ndarray,
    test_pred: np.ndarray,
    test_app_ids: pd.Series,
):
    """Считает метрики, сохраняет .npz для ансамбля (v4) и .csv для сабмита."""
    oof_auc = roc_auc_score(y[train_idx], oof_train)
    holdout_auc = roc_auc_score(y[holdout_idx], holdout_pred)

    logger.info("=" * 55)
    logger.info(
        f"[{VERSION.upper()} | {model_name.upper()}] "
        f"CV OOF AUC: {oof_auc:.5f} | Holdout (20%) AUC: {holdout_auc:.5f}"
    )
    logger.info("=" * 55)


    np.savez(
        PREDS_DIR / f"{VERSION}_{model_name}.npz",
        oof_train=oof_train,
        holdout_pred=holdout_pred,
        test_pred=test_pred,
        train_idx=train_idx,
        holdout_idx=holdout_idx,
    )

    sub = pd.DataFrame({"app_id": test_app_ids, "flag": test_pred})
    assert not sub["flag"].isna().any(), f"Обнаружены NaN в предсказаниях {model_name}!"
    sub_path = SUBMISSIONS_DIR / f"submission_{VERSION}_{model_name}.csv"
    sub.to_csv(sub_path, index=False)
    logger.info(f"Сабмит сохранён: {sub_path}")
