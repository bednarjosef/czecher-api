import time, torch, torch.nn as nn
from typing import Optional, Union

from czecher_tokenizer.bpe_tokenizer import GPTTokenizer


class CzecherTransformer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embedding_dim: int = 256, max_tokens: int = 128, d_model: int = 256, nhead: int = 4, num_layers: int = 5, dim_ff: int = 512, dropout: float = 0.1, max_len=None):  # max_tokens = 512
        super().__init__()
        self.pad_id = pad_id
        if max_len:
            max_tokens = max_len
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.pos = nn.Embedding(max_tokens, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self._config = dict(
            vocab_size=vocab_size,
            pad_id=pad_id,
            embedding_dim=embedding_dim,
            max_tokens=max_tokens,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        # self.to("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        x = self.embed(input_ids) + self.pos(pos_ids)
        key_pad = input_ids.eq(self.pad_id)

        h = self.encoder(x, src_key_padding_mask=key_pad)
        h = self.final_ln(h)
        logits = self.head(h).squeeze(-1)
        return logits

    def _checkpoint_payload(self, optimizer=None, global_steps=0, extra: dict | None = None):
        return {
            "format": "CommaModel.v1",
            "config": self._config,
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "global_steps": global_steps,
            "extra": extra or {},
        }

    @classmethod
    def load(cls, path: str = 'model.pt', map_location: Optional[str | torch.device] = None):
        ckpt = torch.load(path, map_location=map_location)
        config = ckpt["config"]
        model = cls(**config)
        state = ckpt["state_dict"]  # ignore optimizer/extra
        model.load_state_dict(state, strict=True)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        del ckpt, state
        import gc; gc.collect()
        print(f'Model loaded from {path}.')
        return model

    @torch.no_grad()
    def predict_logits(self, input_ids, device=None):
        if device is None:
            device = next(self.parameters()).device
        x = torch.as_tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = self(x)[0]
        mask = (x[0] != self.pad_id)
        return logits, mask

    @torch.no_grad()
    def predict_probs(self, input_ids, device=None):
        logits, mask = self.predict_logits(input_ids, device)
        probs = torch.sigmoid(logits)
        return probs.tolist(), mask

    @torch.no_grad()
    def punctuate(self, text: str, tokenizer: GPTTokenizer, threshold: Union[float, str] = 0.5, device='cuda'):
        ts = time.time()
        """Return a corrected sentence."""
        if isinstance(threshold, str):
            if threshold.lower() != 'best':
                raise ValueError(f"Unsupported threshold string: {threshold}")
            if not hasattr(self, "get_best_eval"):
                raise AttributeError("get_best_eval is required when threshold='best'.")
            best_eval = self.get_best_eval()
            if isinstance(best_eval, dict):
                if "eval/best_threshold" not in best_eval:
                    raise KeyError("get_best_eval must provide 'eval/best_threshold'.")
                best_eval = best_eval["eval/best_threshold"]
            threshold = float(best_eval)
        token_ids = tokenizer.tokenize(text, max_tokens=128)
        probs, _mask = self.predict_probs(token_ids, device)

        pad_stop = len(token_ids)
        if self.pad_id in token_ids:
            pad_stop = token_ids.index(self.pad_id)

        punctuated = ''
        for idx in range(pad_stop):
            punctuated += tokenizer.detokenize([token_ids[idx]])
            if probs[idx] >= threshold:
                punctuated += ','
        return round(time.time()-ts, 3), punctuated
    
