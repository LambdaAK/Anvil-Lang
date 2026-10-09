"""The same model in PyTorch, to check the Anvil clone against and to time against.

BertModel as Hugging Face's transformers library writes it (embeddings, then 12 layers of
self-attention and MLP with post-layer-norm residuals), loading the same model.safetensors. It needs
only PyTorch and NumPy: the safetensors file is read directly.
"""
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

D, HEADS, FF, LAYERS, VOCAB, EPS = 384, 12, 1536, 12, 30522, 1e-12
DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}


def read_safetensors(path: str) -> dict:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        out[name] = np.frombuffer(data[a:b], dtype=DTYPES[info["dtype"]]).reshape(info["shape"]).copy()
    return out


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.query, self.key, self.value, self.out = (nn.Linear(D, D) for _ in range(4))
        self.norm1 = nn.LayerNorm(D, eps=EPS)
        self.up, self.down = nn.Linear(D, FF), nn.Linear(FF, D)
        self.norm2 = nn.LayerNorm(D, eps=EPS)

    def forward(self, x, mask, drop):
        n, t, _ = x.shape
        split = lambda y: y.view(n, t, HEADS, D // HEADS).transpose(1, 2)        # [n, heads, t, dh]
        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(D // HEADS) + mask
        p = F.dropout(scores.softmax(-1), drop, self.training)
        mixed = (p @ v).transpose(1, 2).reshape(n, t, D)
        x = self.norm1(x + F.dropout(self.out(mixed), drop, self.training))
        return self.norm2(x + F.dropout(self.down(F.gelu(self.up(x))), drop, self.training))


class Bert(nn.Module):
    def __init__(self, weights: dict, maxlen: int = 64):
        super().__init__()
        self.words = nn.Embedding(VOCAB, D)
        self.places = nn.Embedding(maxlen, D)
        self.segment = nn.Parameter(torch.zeros(D))
        self.norm = nn.LayerNorm(D, eps=EPS)
        self.layers = nn.ModuleList(Layer() for _ in range(LAYERS))
        w = {k: torch.from_numpy(v) for k, v in weights.items()}
        with torch.no_grad():
            self.words.weight.copy_(w["embeddings.word_embeddings.weight"])
            self.places.weight.copy_(w["embeddings.position_embeddings.weight"][:maxlen])
            self.segment.copy_(w["embeddings.token_type_embeddings.weight"][0])
            self.norm.weight.copy_(w["embeddings.LayerNorm.weight"])
            self.norm.bias.copy_(w["embeddings.LayerNorm.bias"])
            for k, layer in enumerate(self.layers):
                at = f"encoder.layer.{k}."
                pairs = [(layer.query, "attention.self.query"), (layer.key, "attention.self.key"),
                         (layer.value, "attention.self.value"), (layer.out, "attention.output.dense"),
                         (layer.norm1, "attention.output.LayerNorm"), (layer.up, "intermediate.dense"),
                         (layer.down, "output.dense"), (layer.norm2, "output.LayerNorm")]
                for module, name in pairs:
                    module.weight.copy_(w[at + name + ".weight"])
                    module.bias.copy_(w[at + name + ".bias"])

    def forward(self, ids, drop=0.0):
        n, t = ids.shape
        mask = ((ids == 0).float() * torch.finfo(torch.float32).min)[:, None, None, :]   # as transformers does
        x = self.words(ids) + self.places.weight[:t] + self.segment
        x = F.dropout(self.norm(x), drop, self.training)
        for layer in self.layers:
            x = layer(x, mask, drop)
        return x


class Sentiment(nn.Module):
    def __init__(self, weights: dict):
        super().__init__()
        self.bert = Bert(weights)
        self.head = nn.Linear(D, 2)
        nn.init.normal_(self.head.weight, 0, 0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids, drop=0.0):
        return self.head(F.dropout(self.bert(ids, drop)[:, 0], drop, self.training))
