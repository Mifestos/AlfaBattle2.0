from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from utils import PREDS_DIR, PROCESSED_DIR, SUBMISSIONS_DIR, VERSION, logger


def main():
    train_df = pd.read_parquet(PROCESSED_DIR / f"train_{VERSION}.parquet", columns=["app_id", "flag"])
    test_df = pd.read_parquet(PROCESSED_DIR / f"test_{VERSION}.parquet", columns=["app_id"])
    y = train_df["flag"].astype(int).values

    models = ["lgbm", "xgb", "cat"]
    oof_list, holdout_list, test_list = [], [], []
    train_idx, holdout_idx = None, None

    for m in models:
        data = np.load(PREDS_DIR / f"{VERSION}_{m}.npz")
        oof_list.append(data["oof_train"])
        holdout_list.append(data["holdout_pred"])
        test_list.append(data["test_pred"])
        train_idx = data["train_idx"]
        holdout_idx = data["holdout_idx"]

        logger.info(
            f"{m.upper():<6} | CV OOF: {roc_auc_score(y[train_idx], data['oof_train']):.5f} "
            f"| Holdout: {roc_auc_score(y[holdout_idx], data['holdout_pred']):.5f}"
        )

    oof_matrix = np.column_stack(oof_list)
    holdout_matrix = np.column_stack(holdout_list)
    test_matrix = np.column_stack(test_list)

    eq_weights = np.ones(len(models)) / len(models)
    logger.info("-" * 55)
    logger.info(
        f"MEAN BLEND | CV OOF: {roc_auc_score(y[train_idx], oof_matrix @ eq_weights):.5f} "
        f"| Holdout: {roc_auc_score(y[holdout_idx], holdout_matrix @ eq_weights):.5f}"
    )

    oof_ranks = np.column_stack([rankdata(col) / len(col) for col in oof_list])
    holdout_ranks = np.column_stack([rankdata(col) / len(col) for col in holdout_list])
    test_ranks = np.column_stack([rankdata(col) / len(col) for col in test_list])

    logger.info(
        f"RANK BLEND | CV OOF: {roc_auc_score(y[train_idx], oof_ranks @ eq_weights):.5f} "
        f"| Holdout: {roc_auc_score(y[holdout_idx], holdout_ranks @ eq_weights):.5f}"
    )

    def loss_func(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        pred = oof_matrix @ weights
        return -roc_auc_score(y[train_idx], pred)

    res = minimize(loss_func, eq_weights, method="Nelder-Mead")
    best_w = np.clip(res.x, 0, 1)
    best_w /= np.sum(best_w)

    logger.info("-" * 55)
    logger.info(f"Оптимальные веса (LGBM, XGB, CAT): {np.round(best_w, 4)}")
    logger.info(
        f"WEIGHTED   | CV OOF: {roc_auc_score(y[train_idx], oof_matrix @ best_w):.5f} "
        f"| Holdout: {roc_auc_score(y[holdout_idx], holdout_matrix @ best_w):.5f}"
    )

    sub = pd.DataFrame({"app_id": test_df["app_id"], "flag": test_matrix @ best_w})
    sub.to_csv(SUBMISSIONS_DIR / f"submission_{VERSION}_gbdt_blend.csv", index=False)
    logger.info(f"Сабмит блендинга сохранён: submissions/submission_{VERSION}_gbdt_blend.csv")


if __name__ == "__main__":
    main()
