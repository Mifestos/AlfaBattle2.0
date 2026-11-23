import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PREDS_DIR = ROOT_DIR / "data" / "predictions"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "lgbm": "v3_lgbm.npz",
    "xgb": "v3_xgb.npz",
    "cat": "v3_cat.npz",
    "gru": "v3_gru.npz",
    "hybrid": "v3_hybrid.npz",
}


def to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Перевод вероятностей [0, 1] в пространство логарифма шансов (логитов)."""
    return logit(np.clip(probs, eps, 1.0 - eps))


def fast_auc(y_true: np.ndarray, y_score: np.ndarray, n_pos: int, n_neg: int) -> float:
    """Быстрый расчёт ROC-AUC через статистику Манна — Уитни."""
    ranks = rankdata(y_score, method="average")
    return float((np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def optimize_logit_weights(oof_logits: np.ndarray, y_true: np.ndarray, seed: int = 42) -> np.ndarray:
    """Подбор оптимальных неотрицательных весов в пространстве логитов строго по CV OOF."""
    _, n_models = oof_logits.shape
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    best_w = np.ones(n_models, dtype=np.float64) / n_models
    best_score = fast_auc(y_true, oof_logits @ best_w, n_pos, n_neg)

    rng = np.random.default_rng(seed)
    candidates = np.vstack((best_w, rng.dirichlet(np.ones(n_models), size=300)))
    for w in candidates:
        score = fast_auc(y_true, oof_logits @ w, n_pos, n_neg)
        if score > best_score:
            best_score = score
            best_w = w

    def objective(raw_w: np.ndarray) -> float:
        w = np.clip(raw_w, 0.0, None)
        s = np.sum(w)
        if s == 0:
            return 0.0
        w /= s
        return -fast_auc(y_true, oof_logits @ w, n_pos, n_neg)

    res = minimize(objective, best_w, method="Nelder-Mead", options={"maxiter": 150})
    w_opt = np.clip(res.x, 0.0, None)
    return w_opt / np.sum(w_opt)


def main():
    train_df = pd.read_parquet(PROCESSED_DIR / "train_v2.parquet", columns=["app_id", "flag"])
    test_df = pd.read_parquet(PROCESSED_DIR / "test_v2.parquet", columns=["app_id"])
    y = train_df["flag"].astype(int).values

    oof_dict, holdout_dict, test_dict = {}, {}, {}
    train_idx, holdout_idx = None, None

    logger.info("=== 1. Загрузка и проверка моделей первого уровня ===")
    for name, fname in MODELS.items():
        path = PREDS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл предсказаний: {path}")

        data = np.load(path)
        if train_idx is None:
            train_idx = data["train_idx"]
            holdout_idx = data["holdout_idx"]
        else:
            assert np.array_equal(train_idx, data["train_idx"]), f"Несовпадение train_idx в {fname}"
            assert np.array_equal(holdout_idx, data["holdout_idx"]), f"Несовпадение holdout_idx в {fname}"

        oof_dict[name] = data["oof_train"]
        holdout_dict[name] = data["holdout_pred"]
        test_dict[name] = data["test_pred"]

        cv_auc = roc_auc_score(y[train_idx], oof_dict[name])
        ho_auc = roc_auc_score(y[holdout_idx], holdout_dict[name])
        logger.info(f"  {name.upper():<6} | CV OOF AUC: {cv_auc:.5f} | Holdout AUC: {ho_auc:.5f}")

    model_names = list(oof_dict.keys())
    y_tr, y_ho = y[train_idx], y[holdout_idx]

    corr_df = pd.DataFrame(oof_dict).corr()
    logger.info("\n=== 2. Корреляция предсказаний (OOF) ===\n" + corr_df.round(4).to_string())

    oof_logits = to_logits(np.column_stack([oof_dict[m] for m in model_names]))
    ho_logits = to_logits(np.column_stack([holdout_dict[m] for m in model_names]))
    te_logits = to_logits(np.column_stack([test_dict[m] for m in model_names]))

    logger.info("\n=== 3. Оптимизация Weighted Logit Ensemble ===")
    weights = optimize_logit_weights(oof_logits, y_tr)

    oof_ens = expit(oof_logits @ weights)
    ho_ens = expit(ho_logits @ weights)
    te_ens = expit(te_logits @ weights)

    cv_ens_auc = roc_auc_score(y_tr, oof_ens)
    ho_ens_auc = roc_auc_score(y_ho, ho_ens)
    best_single_ho = max(roc_auc_score(y_ho, holdout_dict[m]) for m in model_names)

    weights_str = ", ".join(f"{m.upper()}={w:.4f}" for m, w in zip(model_names, weights))
    logger.info(f"  Оптимальные веса: {weights_str}")
    logger.info("=" * 65)
    logger.info(f"  WEIGHTED LOGIT ENSEMBLE | CV OOF AUC: {cv_ens_auc:.5f} | Holdout AUC: {ho_ens_auc:.5f}")
    logger.info(f"  Прирост над лучшей одиночной моделью (Holdout): +{ho_ens_auc - best_single_ho:.5f}")
    logger.info("=" * 65)

    out_path = SUBMISSIONS_DIR / "submission_v4_ensemble.csv"
    pd.DataFrame({"app_id": test_df["app_id"], "flag": te_ens}).to_csv(out_path, index=False)
    logger.info(f"Финальный сабмит сохранён: {out_path}")


if __name__ == "__main__":
    main()
