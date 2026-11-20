import logging
from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw" / "archive"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

VERSION = "v1"
RANDOM_STATE = 42


def main():
    train_target = pd.read_csv(RAW_DIR / "alfabattle2_train_target.csv")
    test_target = pd.read_csv(RAW_DIR / "alfabattle2_test_target_contest.csv")

    X = train_target[["product"]]
    y = train_target["flag"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    dc = DummyClassifier(strategy="prior")
    dc.fit(X_train, y_train)

    val_preds = dc.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    logger.info(f"[{VERSION}] Baseline Holdout ROC-AUC: {auc:.5f}")

    test_preds = dc.predict_proba(test_target[["product"]])[:, 1]

    submission = pd.DataFrame({
        "app_id": test_target["app_id"],
        "flag": test_preds
    })
    assert len(submission) == len(test_target), "Несоответствие длины сабмита!"
    assert not submission["flag"].isna().any(), "Обнаружены NaN в предсказаниях!"

    out_path = SUBMISSIONS_DIR / f"submission_{VERSION}.csv"
    submission.to_csv(out_path, index=False)
    logger.info(f"Сабмит сохранён в {out_path}")


if __name__ == "__main__":
    main()
