import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
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
    "batch_size": 1024,
    "epochs": 8,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "cnn_channels": 96,
    "tab_hidden": 256,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HybridBatchLoader:
    """Векторизованный загрузчик батчей (последовательность + табличные фичи) напрямую из RAM."""

    def __init__(self, cat_seqs, num_seqs, tab_feats, prods, y=None, batch_size=1024, shuffle=False, drop_last=False):
        self.cat_seqs = cat_seqs
        self.num_seqs = num_seqs
        self.tab_feats = tab_feats
        self.prods = prods
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(prods)

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.random.permutation(self.n_samples) if self.shuffle else np.arange(self.n_samples)
        limit = (self.n_samples // self.batch_size) * self.batch_size if self.drop_last else self.n_samples

        for start in range(0, limit, self.batch_size):
            idx = indices[start : start + self.batch_size]
            cat_b = torch.as_tensor(self.cat_seqs[idx], dtype=torch.long)
            num_b = torch.as_tensor(self.num_seqs[idx], dtype=torch.float32)
            tab_b = torch.as_tensor(self.tab_feats[idx], dtype=torch.float32)
            prod_b = torch.as_tensor(self.prods[idx], dtype=torch.long)

            if self.y is not None:
                y_b = torch.as_tensor(self.y[idx], dtype=torch.float32)
                yield cat_b, num_b, tab_b, prod_b, y_b
            else:
                yield cat_b, num_b, tab_b, prod_b


class MultiScaleCNNBlock(nn.Module):
    """Параллельные 1D-свёртки с окнами 3, 5 и 7 для поиска локальных паттернов покупок."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.norm = nn.BatchNorm1d(out_channels * 3)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = torch.cat((self.conv3(x), self.conv5(x), self.conv7(x)), dim=1)
        return self.drop(self.act(self.norm(h)))


class HybridCNNTabularNet(nn.Module):
    def __init__(
        self,
        vocab_sizes: np.ndarray,
        num_seq_features: int,
        num_tab_features: int,
        prod_vocab_size: int,
        cnn_channels: int = 96,
        tab_hidden: int = 256,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList()
        emb_total_dim = 0
        for vs in vocab_sizes:
            emb_dim = min(32, max(4, (int(vs) + 1) // 3))
            self.embeddings.append(nn.Embedding(int(vs), emb_dim, padding_idx=0))
            emb_total_dim += emb_dim

        self.emb_drop = nn.Dropout(0.15)
        seq_in_dim = emb_total_dim + num_seq_features

        self.cnn1 = MultiScaleCNNBlock(seq_in_dim, cnn_channels)
        self.cnn2 = MultiScaleCNNBlock(cnn_channels * 3, cnn_channels)

        cnn_out_dim = cnn_channels * 3
        self.attn = nn.Sequential(
            nn.Linear(cnn_out_dim, cnn_out_dim // 2),
            nn.Tanh(),
            nn.Linear(cnn_out_dim // 2, 1),
        )

        self.tab_net = nn.Sequential(
            nn.Linear(num_tab_features, tab_hidden),
            nn.BatchNorm1d(tab_hidden),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(tab_hidden, tab_hidden // 2),
            nn.BatchNorm1d(tab_hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        self.prod_emb = nn.Embedding(int(prod_vocab_size), 8, padding_idx=0)

        head_in = cnn_out_dim * 3 + (tab_hidden // 2) + 8
        self.head = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Dropout(0.3),
            nn.Linear(head_in, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, cat_x, num_x, tab_x, prod_x):
        mask = cat_x[:, :, 0] != 0  # [B, T]
        embs = [emb(cat_x[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x = self.emb_drop(torch.cat([*embs, num_x], dim=-1))  # [B, T, C]

        x_t = x.transpose(1, 2)
        h = self.cnn1(x_t)
        h = self.cnn2(h)
        out = h.transpose(1, 2)  

        scores = self.attn(out).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        attn_feat = torch.sum(out * weights, dim=1)

        out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e4)
        max_feat = torch.max(out_masked, dim=1).values
        max_feat = torch.where(torch.isfinite(max_feat), max_feat, torch.zeros_like(max_feat))

        last_feat = out[:, -1, :]

        tab_feat = self.tab_net(tab_x)
        p_feat = self.prod_emb(prod_x)

        combined = torch.cat((attn_feat, max_feat, last_feat, tab_feat, p_feat), dim=-1)
        return self.head(combined).squeeze(-1)


@torch.no_grad()
def predict(model: nn.Module, loader: HybridBatchLoader) -> np.ndarray:
    model.eval()
    preds = []
    use_amp = DEVICE.type == "cuda"
    for batch in loader:
        cat_x, num_x, tab_x, prod_x = [b.to(DEVICE, non_blocking=True) for b in batch[:4]]
        with torch.amp.autocast("cuda", enabled=use_amp):
            probs = torch.sigmoid(model(cat_x, num_x, tab_x, prod_x))
        preds.append(probs.cpu().numpy())
    return np.concatenate(preds)


def main():
    seed_everything(CFG["seed"])
    logger.info(f"Устройство: {DEVICE} | Параметры Hybrid 1D-CNN: {CFG}")

    seq_data = np.load(PROCESSED_DIR / "seq_v3.npz")
    train_cat, train_num, train_prod, y = (
        seq_data["train_cat"], seq_data["train_num"], seq_data["train_prod"], seq_data["y"]
    )
    test_cat, test_num, test_prod = (
        seq_data["test_cat"], seq_data["test_num"], seq_data["test_prod"]
    )
    vocab_sizes = seq_data["vocab_sizes"]
    prod_vocab_size = int(seq_data["prod_vocab_size"])

    train_df = pd.read_parquet(PROCESSED_DIR / "train_v3.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test_v3.parquet")
    ignore_cols = {"app_id", "flag", "product"}
    tab_cols = [c for c in train_df.columns if c not in ignore_cols]

    X_tab_raw = train_df[tab_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    X_test_tab_raw = test_df[tab_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    indices = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(indices, test_size=0.2, random_state=CFG["seed"], stratify=y)
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    folds = list(skf.split(train_idx, y[train_idx]))

    oof_train = np.zeros(len(train_idx), dtype=np.float32)
    holdout_pred = np.zeros(len(holdout_idx), dtype=np.float32)
    test_pred = np.zeros(len(test_prod), dtype=np.float32)

    criterion = nn.BCEWithLogitsLoss()
    use_amp = DEVICE.type == "cuda"

    for fold, (tr_sub, va_sub) in enumerate(folds, 1):
        seed_everything(CFG["seed"] + fold)
        tr_idx, va_idx = train_idx[tr_sub], train_idx[va_sub]

        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=CFG["seed"])
        X_tr_tab = qt.fit_transform(X_tab_raw[tr_idx]).astype(np.float32)
        X_va_tab = qt.transform(X_tab_raw[va_idx]).astype(np.float32)
        X_ho_tab = qt.transform(X_tab_raw[holdout_idx]).astype(np.float32)
        X_te_tab = qt.transform(X_test_tab_raw).astype(np.float32)

        tr_loader = HybridBatchLoader(
            train_cat[tr_idx], train_num[tr_idx], X_tr_tab, train_prod[tr_idx], y[tr_idx],
            batch_size=CFG["batch_size"], shuffle=True, drop_last=True,
        )
        va_loader = HybridBatchLoader(
            train_cat[va_idx], train_num[va_idx], X_va_tab, train_prod[va_idx],
            batch_size=CFG["batch_size"] * 2, shuffle=False,
        )
        ho_loader = HybridBatchLoader(
            train_cat[holdout_idx], train_num[holdout_idx], X_ho_tab, train_prod[holdout_idx],
            batch_size=CFG["batch_size"] * 2, shuffle=False,
        )
        te_loader = HybridBatchLoader(
            test_cat, test_num, X_te_tab, test_prod,
            batch_size=CFG["batch_size"] * 2, shuffle=False,
        )

        model = HybridCNNTabularNet(
            vocab_sizes, train_num.shape[-1], X_tr_tab.shape[-1], prod_vocab_size,
            CFG["cnn_channels"], CFG["tab_hidden"],
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CFG["lr"], epochs=CFG["epochs"], steps_per_epoch=len(tr_loader), pct_start=0.1
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_auc = 0.0
        best_va_preds = None
        weights_path = PREDS_DIR / f"hybrid_cnn_fold_{fold}.pt"

        for epoch in range(1, CFG["epochs"] + 1):
            model.train()
            total_loss = 0.0

            for cat_x, num_x, tab_x, prod_x, target in tr_loader:
                cat_x = cat_x.to(DEVICE, non_blocking=True)
                num_x = num_x.to(DEVICE, non_blocking=True)
                tab_x = tab_x.to(DEVICE, non_blocking=True)
                prod_x = prod_x.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(model(cat_x, num_x, tab_x, prod_x), target)

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
        holdout_pred += predict(model, ho_loader) / CFG["n_folds"]
        test_pred += predict(model, te_loader) / CFG["n_folds"]

    oof_auc = roc_auc_score(y[train_idx], oof_train)
    holdout_auc = roc_auc_score(y[holdout_idx], holdout_pred)
    logger.info(f"[V3 | Hybrid 1D-CNN] CV OOF AUC: {oof_auc:.5f} | Holdout AUC: {holdout_auc:.5f}")

    np.savez(
        PREDS_DIR / "v3_hybrid.npz",
        oof_train=oof_train,
        holdout_pred=holdout_pred,
        test_pred=test_pred,
        train_idx=train_idx,
        holdout_idx=holdout_idx,
    )

    test_target = pd.read_csv(RAW_DIR / "alfabattle2_test_target_contest.csv")
    pd.DataFrame({"app_id": test_target["app_id"], "flag": test_pred}).to_csv(
        SUBMISSIONS_DIR / "submission_v3_hybrid.csv", index=False
    )


if __name__ == "__main__":
    main()
