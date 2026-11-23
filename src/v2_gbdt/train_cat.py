import numpy as np
from catboost import CatBoostClassifier
from utils import (
    RANDOM_STATE,
    N_FOLDS,
    logger,
    apply_tuned_params,
    load_data_and_splits,
    evaluate_and_save,
)


def main():
    X, y, X_test, test_ids, train_idx, holdout_idx, folds, cat_cols = load_data_and_splits(for_catboost=True)

    params = {
        "iterations": 6000,
        "learning_rate": 0.04,
        "depth": 7,
        "l2_leaf_reg": 10.0,
        "eval_metric": "AUC",
        "cat_features": cat_cols,
        "task_type": "GPU",
        "gpu_ram_part": 0.85,
        "verbose": 0,
    }
    params = apply_tuned_params("cat", params)

    oof_train = np.zeros(len(train_idx))
    holdout_pred = np.zeros(len(holdout_idx))
    test_pred = np.zeros(len(X_test))

    logger.info("Запуск обучения CatBoost...")

    for fold, (tr_sub, va_sub) in enumerate(folds, 1):
        tr_idx = train_idx[tr_sub]
        va_idx = train_idx[va_sub]

        model = CatBoostClassifier(**params, random_seed=RANDOM_STATE + fold)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=(X.iloc[va_idx], y[va_idx]),
            early_stopping_rounds=300,
            verbose=False,
        )

        oof_train[va_sub] = model.predict_proba(X.iloc[va_idx])[:, 1]
        holdout_pred += model.predict_proba(X.iloc[holdout_idx])[:, 1] / N_FOLDS
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        logger.info(f"CAT  | Fold {fold}/{N_FOLDS} | Best iter: {model.get_best_iteration()}")

    evaluate_and_save("cat", y, train_idx, holdout_idx, oof_train, holdout_pred, test_pred, test_ids)


if __name__ == "__main__":
    main()
