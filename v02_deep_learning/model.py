"""
Мультимодальный трансформер с эмбеддингами, CLS-токеном и Multi-Task обучением.
"""
import torch
import torch.nn as nn

class SequenceTransformer(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 4, num_layers: int = 1, dropout: float = 0.1, max_len: int = 30):
        super().__init__()

        self.mcc_emb = nn.Embedding(100, d_model // 4, padding_idx=0)
        self.dow_emb = nn.Embedding(7, d_model // 4, padding_idx=0)
        self.num_proj = nn.Linear(4, d_model // 2)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = nn.Parameter(torch.randn(1, max_len + 1, d_model))
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_feats = x[:, :, [0, 1, 3, 4]] 
        mcc_idx = x[:, :, 2].long()
        dow_idx = x[:, :, 5].long()
        mask = x[:, :, 6].bool() 
        
       
        emb = torch.cat([self.num_proj(num_feats), self.mcc_emb(mcc_idx), self.dow_emb(dow_idx)], dim=-1)
        
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        emb = torch.cat([cls_tokens, emb], dim=1) + self.pos_enc
        
        cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        
        out = self.encoder(emb, src_key_padding_mask=full_mask)
        return out

class FTTransformer(nn.Module):
    def __init__(self, num_features: int, d_model: int = 32, nhead: int = 4, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = self.dropout(torch.cat([cls, x], dim=1))
        return self.encoder(x)[:, 0]

class MultimodalTransformer(nn.Module):
    def __init__(self, tab_input_dim: int = 36, d_model: int = 32, nhead: int = 4, num_layers: int = 1, dropout: float = 0.15):
        super().__init__()
        self.seq_encoder = SequenceTransformer(d_model, nhead, num_layers, dropout)
        self.tab_encoder = FTTransformer(tab_input_dim, d_model, nhead, num_layers, dropout)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        self.head_main = nn.Sequential(
            nn.Linear(d_model * 2, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.head_aux = nn.Sequential(
            nn.Linear(d_model * 2, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, seq_x: torch.Tensor, tab_x: torch.Tensor):
        seq_out = self.seq_encoder(seq_x)
        tab_out = self.tab_encoder(tab_x)
        
        attn_out, _ = self.cross_attn(
            tab_out.unsqueeze(1), seq_out, seq_out
        )
        attn_out = self.norm(attn_out.squeeze(1) + tab_out)
        
        combined = torch.cat([attn_out, seq_out[:, 0, :]], dim=1)
        
        return self.head_main(combined), self.head_aux(combined)