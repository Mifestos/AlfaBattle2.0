"""
Оптимизированный dataset с поддержкой категориальных эмбеддингов и маски.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    def __init__(self, trx_files: list[Path], target_df: pd.DataFrame, tabular_df: pd.DataFrame, max_len: int = 30) -> None:
        self.max_len = max_len
        self.app_ids = target_df["app_id"].to_numpy()
        if "flag" in target_df.columns:
            self.labels = target_df["flag"].to_numpy(dtype=np.float32)
        else:
            self.labels = np.zeros(len(target_df), dtype=np.float32)
        self.app_id_to_idx = {app_id: i for i, app_id in enumerate(self.app_ids)}
        
        self.tabular_features = (
            tabular_df.set_index("app_id").reindex(self.app_ids).fillna(0)
            .replace([np.inf, -np.inf], 0).to_numpy(dtype=np.float32)
        )
        
        
        self.sequences = np.zeros((len(self.app_ids), max_len, 8), dtype=np.float32)
        
        cols = ["app_id", "amnt", "hour", "day_of_week", "days_before", "ecommerce_flag", "mcc_category"]
        for path in trx_files:
            trx = pd.read_parquet(path, columns=cols)
            trx = trx.sort_values(["app_id", "days_before"], ascending=[True, False])
            
            for app_id, group in trx.groupby("app_id", sort=False):
                if app_id not in self.app_id_to_idx: continue
                idx = self.app_id_to_idx[app_id]
                seq = self._extract_sequence_features(group)
                
                if len(seq) > max_len:
                    self.sequences[idx] = seq[-max_len:]
                else:
                    self.sequences[idx, -len(seq):, :] = seq
                    self.sequences[idx, :-len(seq), 6] = 1.0 

    @staticmethod
    def _extract_sequence_features(group: pd.DataFrame) -> np.ndarray:
        amnt = group["amnt"].to_numpy(dtype=np.float32)
        amnt = np.clip(np.nan_to_num(amnt, nan=0.0), -0.99, 1e6)
        
        hour = np.clip(group["hour"].to_numpy(dtype=np.float32), 0, 23) / 23.0
        dow = np.clip(group["day_of_week"].to_numpy(dtype=np.float32), 0, 6)
        days_before = np.clip(group["days_before"].to_numpy(dtype=np.float32), 0, 365) / 365.0
        ecom = group["ecommerce_flag"].fillna(0).to_numpy(dtype=np.float32)
        mcc = np.clip(group["mcc_category"].fillna(0).to_numpy(dtype=np.float32), 0, 99)
        
        return np.stack([
            np.log1p(amnt), hour, mcc, days_before, ecom, dow, 
            np.zeros_like(amnt), amnt
        ], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.app_ids)

    def __getitem__(self, idx: int):
        total_amnt = np.log1p(np.sum(self.sequences[idx, :, 7]))
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.from_numpy(self.tabular_features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(total_amnt, dtype=torch.float32),
            self.app_ids[idx],
        )