import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[2]

from src.v2_gbdt.utils import RANDOM_STATE, load_data_and_splits, logger


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIG_PATH = ROOT_DIR / "configs" / "gbdt_params.json"
STORAGE_PATH = ROOT_DIR / "src" / "tuning" / "optuna_studies.db"
STORAGE_URL = f"sqlite:///{STORAGE_PATH.as_posix()}"


def save_best_config(
    model_name: str,
    study: optuna.Study,
    fixed_params: Dict[str, Any],
    final_lr: float,
    final_n_estimators: int,
) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry: Dict[str, Any] = {}

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                registry = json.load(f)
            except json.JSONDecodeError:
                registry = {}

    tuned_params = dict(study.best_params)

    if model_name in ("xgb", "lgbm"):
        tuned_params["learning_rate"] = final_lr
        tuned_params["n_estimators"] = final_n_estimators
    elif model_name == "cat":
        tuned_params["learning_rate"] = final_lr
        tuned_params["iterations"] = final_n_estimators

    registry[model_name] = {
        "best_tune_auc": round(float(study.best_value), 6),
        "best_trial": int(study.best_trial.number),
        "best_iteration_mean": int(study.best_trial.user_attrs.get("mean_best_iter", final_n_estimators)),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tuned_params": tuned_params,
        "fixed_params": fixed_params,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4, ensure_ascii=False)

    logger.info(f"[{model_name.upper()}] Конфигурация сохранена -> {CONFIG_PATH}")


def make_trial_callback(model_tag: str, n_folds: int) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"{model_tag:<4} | Trial {trial.number:03d} | PRUNED")
            return
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        duration = trial.user_attrs.get("duration_sec", 0.0)
        mean_iter = trial.user_attrs.get("mean_best_iter", -1)
        is_best = " *NEW BEST*" if study.best_trial.number == trial.number else ""
        logger.info(
            f"{model_tag:<4} | Trial {trial.number:03d} | "
            f"AUC ({n_folds}f): {trial.value:.5f} | "
            f"Iter: {mean_iter:4d} | Time: {duration:5.1f}s{is_best}"
        )

    return _callback


def tune_xgb(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    n_tune_folds: int,
) -> None:
    logger.info(f"=== Запуск Optuna: XGBoost (trials={n_trials}, folds={n_tune_folds}) ===")

    fixed_params = {
        "max_bin": 256,
        "enable_categorical": True,
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
    }

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params = {
            **fixed_params,
            "n_estimators": 4500,
            "learning_rate": 0.03,
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 20, 120),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.30, 0.75),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.50, 1.00),
            "subsample": trial.suggest_float("subsample", 0.70, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 35.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 60.0, log=True),
            "max_cat_threshold": trial.suggest_categorical("max_cat_threshold", [16, 32, 64]),
            "early_stopping_rounds": 150,
        }

        fold_scores: List[float] = []
        best_iters: List[int] = []

        for fold_idx, (tr_sub, va_sub) in enumerate(folds[:n_tune_folds], 1):
            tr_i, va_i = train_idx[tr_sub], train_idx[va_sub]
            callbacks = [XGBoostPruningCallback(trial, "validation_0-auc")] if fold_idx == 1 else []

            model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE + fold_idx, callbacks=callbacks)
            model.fit(
                X.iloc[tr_i],
                y[tr_i],
                eval_set=[(X.iloc[va_i], y[va_i])],
                verbose=False,
            )

            preds = model.predict_proba(X.iloc[va_i])[:, 1]
            fold_scores.append(roc_auc_score(y[va_i], preds))
            best_iters.append(int(model.best_iteration))

        trial.set_user_attr("mean_best_iter", int(np.mean(best_iters)))
        trial.set_user_attr("duration_sec", round(time.perf_counter() - t0, 2))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200)

    study = optuna.create_study(
        study_name="alfabattle_xgb_v2",
        storage=STORAGE_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[make_trial_callback("XGB", n_tune_folds)], gc_after_trial=True)

    logger.info(f"XGB Best AUC: {study.best_value:.5f} | Params: {study.best_params}")
    save_best_config("xgb", study, fixed_params, final_lr=0.015, final_n_estimators=7000)


def tune_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    n_tune_folds: int,
) -> None:
    logger.info(f"=== Запуск Optuna: LightGBM (trials={n_trials}, folds={n_tune_folds}) ===")

    fixed_params = {
        "subsample_freq": 1,
        "metric": "auc",
        "n_jobs": -1,
        "verbose": -1,
    }

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params = {
            **fixed_params,
            "n_estimators": 4500,
            "learning_rate": 0.03,
            "num_leaves": trial.suggest_int("num_leaves", 15, 45),
            "max_depth": trial.suggest_int("max_depth", 4, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 150, 600),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.30, 0.75),
            "subsample": trial.suggest_float("subsample", 0.70, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 35.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 60.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [255, 511]),
        }

        fold_scores: List[float] = []
        best_iters: List[int] = []

        for fold_idx, (tr_sub, va_sub) in enumerate(folds[:n_tune_folds], 1):
            tr_i, va_i = train_idx[tr_sub], train_idx[va_sub]
            callbacks = [lgb.early_stopping(150, verbose=False)]
            if fold_idx == 1:
                callbacks.append(LightGBMPruningCallback(trial, "auc"))

            model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE + fold_idx)
            model.fit(
                X.iloc[tr_i],
                y[tr_i],
                eval_set=[(X.iloc[va_i], y[va_i])],
                eval_metric="auc",
                callbacks=callbacks,
            )

            preds = model.predict_proba(X.iloc[va_i])[:, 1]
            fold_scores.append(roc_auc_score(y[va_i], preds))
            best_iters.append(int(model.best_iteration_))

        trial.set_user_attr("mean_best_iter", int(np.mean(best_iters)))
        trial.set_user_attr("duration_sec", round(time.perf_counter() - t0, 2))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200)

    study = optuna.create_study(
        study_name="alfabattle_lgbm_v2",
        storage=STORAGE_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[make_trial_callback("LGBM", n_tune_folds)], gc_after_trial=True)

    logger.info(f"LGBM Best AUC: {study.best_value:.5f} | Params: {study.best_params}")
    save_best_config("lgbm", study, fixed_params, final_lr=0.015, final_n_estimators=7000)


def tune_cat(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cat_cols: List[str],
    n_trials: int,
    n_tune_folds: int,
) -> None:
    logger.info(f"=== Запуск Optuna: CatBoost (trials={n_trials}, folds={n_tune_folds}) ===")

    fixed_params = {
        "bootstrap_type": "Bernoulli",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "gpu_ram_part": 0.85,
        "verbose": 0,
    }

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params = {
            **fixed_params,
            "iterations": 5000,
            "learning_rate": 0.045,
            "depth": trial.suggest_int("depth", 4, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 45.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 0.95),
            "border_count": trial.suggest_categorical("border_count", [128, 254]),
            "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 3),
            "cat_features": cat_cols,
        }

        fold_scores: List[float] = []
        best_iters: List[int] = []

        for fold_idx, (tr_sub, va_sub) in enumerate(folds[:n_tune_folds], 1):
            tr_i, va_i = train_idx[tr_sub], train_idx[va_sub]
            model = CatBoostClassifier(**params, random_seed=RANDOM_STATE + fold_idx)
            model.fit(
                X.iloc[tr_i],
                y[tr_i],
                eval_set=(X.iloc[va_i], y[va_i]),
                early_stopping_rounds=200,
                verbose=False,
            )

            preds = model.predict_proba(X.iloc[va_i])[:, 1]
            fold_scores.append(roc_auc_score(y[va_i], preds))
            best_iters.append(int(model.get_best_iteration()))

        trial.set_user_attr("mean_best_iter", int(np.mean(best_iters)))
        trial.set_user_attr("duration_sec", round(time.perf_counter() - t0, 2))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)

    study = optuna.create_study(
        study_name="alfabattle_cat_v2",
        storage=STORAGE_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[make_trial_callback("CAT", n_tune_folds)], gc_after_trial=True)

    logger.info(f"CAT Best AUC: {study.best_value:.5f} | Params: {study.best_params}")
    save_best_config("cat", study, fixed_params, final_lr=0.02, final_n_estimators=8500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna GBDT Hyperparameter Tuning Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["xgb", "lgbm", "cat", "all"],
        help="Модель для подбора параметров",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Количество новых триалов Optuna для каждой модели",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Количество фолдов для оценки одного триала",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    if args.model in ("xgb", "lgbm", "all"):
        X, y, _, _, train_idx, _, folds, _ = load_data_and_splits(for_catboost=False)
        if args.model in ("xgb", "all"):
            tune_xgb(X, y, train_idx, folds, n_trials=args.trials, n_tune_folds=args.folds)
        if args.model in ("lgbm", "all"):
            tune_lgbm(X, y, train_idx, folds, n_trials=args.trials, n_tune_folds=args.folds)
        del X, y
        gc.collect()

    if args.model in ("cat", "all"):
        X_cat, y_cat, _, _, train_idx, _, folds, cat_cols = load_data_and_splits(for_catboost=True)
        tune_cat(X_cat, y_cat, train_idx, folds, cat_cols, n_trials=args.trials, n_tune_folds=args.folds)
        del X_cat, y_cat
        gc.collect()

    logger.info(f"Весь пайплайн тюнинга завершён за {(time.perf_counter() - t_start) / 60:.1f} мин.")


if __name__ == "__main__":
    main()
