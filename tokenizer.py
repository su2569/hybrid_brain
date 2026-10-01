"""字符级分词器。加载 61050 字符的预置词表。"""
import json
from typing import List


class CharTokenizer:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, itos: List[str]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def from_vocab_json(cls, path: str, vocab_limit: int = None):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        itos = list(data["itos"])
        if vocab_limit is not None:
            itos = itos[:vocab_limit]
        return cls(itos)

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str, add_bos=False, add_eos=False,
               max_len=None) -> List[int]:
        ids = [self.stoi.get(ch, self.UNK) for ch in text]
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            if i in (self.PAD, self.BOS, self.EOS):
                continue
            if i == self.UNK:
                out.append("□")
            elif 0 <= i < len(self.itos):
                out.append(self.itos[i])
        return "".join(out)

    def unk_rate(self, text: str) -> float:
        if not text:
            return 0.0
        n_unk = sum(1 for ch in text if ch not in self.stoi)
        return n_unk / len(text)