import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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
CONFIGS_DIR = ROOT_DIR / "configs"

PREDS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

VERSION = "v3"
RANDOM_STATE = 42
N_FOLDS = 5


def apply_tuned_params(model_name: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Подгружает найденные Optuna параметры из configs/gbdt_params.json поверх базовых."""
    cfg_path = CONFIGS_DIR / "gbdt_params.json"
    params = base_params.copy()

    if not cfg_path.exists():
        return params

    with open(cfg_path, "r", encoding="utf-8") as f:
        try:
            registry = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Не удалось прочитать {cfg_path}, используются базовые параметры.")
            return params

    if model_name in registry:
        entry = registry[model_name]
        params.update(entry.get("fixed_params", {}))
        params.update(entry.get("tuned_params", {}))
        logger.info(
            f"[{model_name.upper()}] Загружен конфиг от {entry.get('updated_at')} "
            f"(Tune AUC: {entry.get('best_tune_auc')}): {entry.get('tuned_params')}"
        )

    return params


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

    cat_cols = [c for c in ("product",) if c in feature_cols]

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
