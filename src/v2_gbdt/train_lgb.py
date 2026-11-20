import numpy as np
import lightgbm as lgb
from utils import (
    RANDOM_STATE,
    N_FOLDS,
    logger,
    load_data_and_splits,
    evaluate_and_save,
)


def main():
    X, y, X_test, test_ids, train_idx, holdout_idx, folds, _ = load_data_and_splits(for_catboost=False)

    params = {
        "n_estimators": 5000,
        "learning_rate": 0.02,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 150,
        "reg_alpha": 5.0,
        "reg_lambda": 10.0,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "subsample_freq": 1,
        "n_jobs": -1,
        "verbose": -1,
    }


    oof_train = np.zeros(len(train_idx))
    holdout_pred = np.zeros(len(holdout_idx))
    test_pred = np.zeros(len(X_test))

    logger.info("Запуск обучения LightGBM...")

    for fold, (tr_sub, va_sub) in enumerate(folds, 1):
        tr_idx = train_idx[tr_sub]
        va_idx = train_idx[va_sub]

        model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE + fold)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_X=X.iloc[va_idx], eval_y=y[va_idx],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        oof_train[va_sub] = model.predict_proba(X.iloc[va_idx])[:, 1]
        holdout_pred += model.predict_proba(X.iloc[holdout_idx])[:, 1] / N_FOLDS
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        logger.info(f"LGBM | Fold {fold}/{N_FOLDS} | Best iter: {model.best_iteration_}")

    evaluate_and_save("lgbm", y, train_idx, holdout_idx, oof_train, holdout_pred, test_pred, test_ids)


if __name__ == "__main__":
    main()
