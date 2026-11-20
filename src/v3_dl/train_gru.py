import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw" / "archive"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PREDS_DIR = ROOT_DIR / "data" / "predictions"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

PREDS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

CFG = {
    "seed": 42,
    "n_folds": 5,
    "batch_size": 512,
    "epochs": 6,
    "lr": 1.5e-3,
    "weight_decay": 1e-4,
    "hidden_size": 128,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FastBatchLoader:
    """Векторизованный генератор батчей напрямую из RAM без накладных расходов Windows multiprocessing."""

    def __init__(self, cat_seqs, num_seqs, seq_lens, prods, y=None, batch_size=512, shuffle=False, drop_last=False):
        self.cat_seqs = cat_seqs
        self.num_seqs = num_seqs
        self.seq_lens = seq_lens
        self.prods = prods
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(prods)
        self.use_pin = DEVICE.type == "cuda"

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def _to_pinned(self, arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        t = torch.as_tensor(arr, dtype=dtype)
        return t.pin_memory() if self.use_pin else t

    def __iter__(self):
        indices = np.random.permutation(self.n_samples) if self.shuffle else np.arange(self.n_samples)
        limit = (self.n_samples // self.batch_size) * self.batch_size if self.drop_last else self.n_samples

        for start in range(0, limit, self.batch_size):
            idx = indices[start : start + self.batch_size]
            cat_b = self._to_pinned(self.cat_seqs[idx], torch.long)
            num_b = self._to_pinned(self.num_seqs[idx], torch.float32)
            len_b = self._to_pinned(self.seq_lens[idx], torch.long)
            prod_b = self._to_pinned(self.prods[idx], torch.long)

            if self.y is not None:
                y_b = self._to_pinned(self.y[idx], torch.float32)
                yield cat_b, num_b, len_b, prod_b, y_b
            else:
                yield cat_b, num_b, len_b, prod_b


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)


class AlfaBiGRUNet(nn.Module):
    def __init__(self, vocab_sizes: np.ndarray, num_features: int, prod_vocab_size: int, hidden_size: int = 128):
        super().__init__()
        self.embeddings = nn.ModuleList()
        emb_total_dim = 0

        for vs in vocab_sizes:
            emb_dim = min(32, max(4, (int(vs) + 1) // 3))
            self.embeddings.append(nn.Embedding(int(vs), emb_dim, padding_idx=0))
            emb_total_dim += emb_dim

        self.emb_drop = nn.Dropout(0.15)
        self.gru = nn.GRU(
            input_size=emb_total_dim + num_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.attn_pool = AttentionPooling(hidden_size * 2)
        self.prod_emb = nn.Embedding(int(prod_vocab_size), 8, padding_idx=0)

        head_in = hidden_size * 2 * 3 + 8
        self.head = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Dropout(0.3),
            nn.Linear(head_in, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, cat_x, num_x, lengths, prod_x):
        mask = cat_x[:, :, 0] != 0
        embs = [emb(cat_x[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x = self.emb_drop(torch.cat([*embs, num_x], dim=-1))

        out, _ = self.gru(x)
        attn_feat = self.attn_pool(out, mask)
        last_feat = out[:, -1, :]

        out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e4)
        max_feat = torch.max(out_masked, dim=1).values
        max_feat = torch.where(torch.isfinite(max_feat), max_feat, torch.zeros_like(max_feat))

        combined = torch.cat((attn_feat, last_feat, max_feat, self.prod_emb(prod_x)), dim=-1)
        return self.head(combined).squeeze(-1)


@torch.no_grad()
def predict(model: nn.Module, loader: FastBatchLoader) -> np.ndarray:
    model.eval()
    preds = []
    use_amp = DEVICE.type == "cuda"
    for batch in loader:
        cat_x, num_x, lengths, prod_x = [b.to(DEVICE, non_blocking=True) for b in batch[:4]]
        with torch.amp.autocast("cuda", enabled=use_amp):
            probs = torch.sigmoid(model(cat_x, num_x, lengths, prod_x))
        preds.append(probs.cpu().numpy())
    return np.concatenate(preds)


def main():
    seed_everything(CFG["seed"])
    logger.info(f"Устройство: {DEVICE} | Параметры: {CFG}")

    data = np.load(PROCESSED_DIR / "seq_v3.npz")
    train_cat, train_num, train_len, train_prod, y = (
        data["train_cat"], data["train_num"], data["train_len"], data["train_prod"], data["y"]
    )
    test_cat, test_num, test_len, test_prod = (
        data["test_cat"], data["test_num"], data["test_len"], data["test_prod"]
    )
    vocab_sizes = data["vocab_sizes"]
    prod_vocab_size = int(data["prod_vocab_size"])

    indices = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(indices, test_size=0.2, random_state=CFG["seed"], stratify=y)
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    folds = list(skf.split(train_idx, y[train_idx]))

    holdout_loader = FastBatchLoader(
        train_cat[holdout_idx], train_num[holdout_idx], train_len[holdout_idx], train_prod[holdout_idx],
        batch_size=CFG["batch_size"] * 2, shuffle=False,
    )
    test_loader = FastBatchLoader(
        test_cat, test_num, test_len, test_prod,
        batch_size=CFG["batch_size"] * 2, shuffle=False,
    )

    oof_train = np.zeros(len(train_idx), dtype=np.float32)
    holdout_pred = np.zeros(len(holdout_idx), dtype=np.float32)
    test_pred = np.zeros(len(test_prod), dtype=np.float32)

    criterion = nn.BCEWithLogitsLoss()
    use_amp = DEVICE.type == "cuda"

    for fold, (tr_sub, va_sub) in enumerate(folds, 1):
        seed_everything(CFG["seed"] + fold)
        tr_idx, va_idx = train_idx[tr_sub], train_idx[va_sub]

        tr_loader = FastBatchLoader(
            train_cat[tr_idx], train_num[tr_idx], train_len[tr_idx], train_prod[tr_idx], y[tr_idx],
            batch_size=CFG["batch_size"], shuffle=True, drop_last=True,
        )
        va_loader = FastBatchLoader(
            train_cat[va_idx], train_num[va_idx], train_len[va_idx], train_prod[va_idx],
            batch_size=CFG["batch_size"] * 2, shuffle=False,
        )

        model = AlfaBiGRUNet(vocab_sizes, train_num.shape[-1], prod_vocab_size, CFG["hidden_size"]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CFG["lr"], epochs=CFG["epochs"], steps_per_epoch=len(tr_loader), pct_start=0.1
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_auc = 0.0
        best_va_preds = None
        weights_path = PREDS_DIR / f"gru_fold_{fold}.pt"

        for epoch in range(1, CFG["epochs"] + 1):
            model.train()
            total_loss = 0.0

            for cat_x, num_x, lengths, prod_x, target in tr_loader:
                cat_x = cat_x.to(DEVICE, non_blocking=True)
                num_x = num_x.to(DEVICE, non_blocking=True)
                lengths = lengths.to(DEVICE, non_blocking=True)
                prod_x = prod_x.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(model(cat_x, num_x, lengths, prod_x), target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()

            va_preds = predict(model, va_loader)
            va_auc = roc_auc_score(y[va_idx], va_preds)
            logger.info(
                f"Фолд {fold}/{CFG['n_folds']} | Эпоха {epoch}/{CFG['epochs']} | "
                f"Loss: {total_loss / len(tr_loader):.4f} | Val AUC: {va_auc:.5f}"
            )

            if va_auc > best_auc:
                best_auc = va_auc
                best_va_preds = va_preds
                torch.save(model.state_dict(), weights_path)

        oof_train[va_sub] = best_va_preds
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        holdout_pred += predict(model, holdout_loader) / CFG["n_folds"]
        test_pred += predict(model, test_loader) / CFG["n_folds"]

        oof_auc = roc_auc_score(y[train_idx], oof_train)
    holdout_auc = roc_auc_score(y[holdout_idx], holdout_pred)
    logger.info(f"[V3 | BiGRU] CV OOF AUC: {oof_auc:.5f} | Holdout AUC: {holdout_auc:.5f}")

    np.savez(
        PREDS_DIR / "v3_gru.npz",
        oof_train=oof_train,
        holdout_pred=holdout_pred,
        test_pred=test_pred,
        train_idx=train_idx,
        holdout_idx=holdout_idx,
    )

    test_target = pd.read_csv(RAW_DIR / "alfabattle2_test_target_contest.csv")
    pd.DataFrame({"app_id": test_target["app_id"], "flag": test_pred}).to_csv(
        SUBMISSIONS_DIR / "submission_v3_gru.csv", index=False
    )


if __name__ == "__main__":
    main()

