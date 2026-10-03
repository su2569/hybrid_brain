"""Qwen BPE tokenizer 适配器。接口对齐 CharTokenizer。"""
from transformers import AutoTokenizer


class QwenTokenizer:
    def __init__(self, path="/mnt/workspace/models/Qwen2.5-0.5B"):
        self.tok = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.PAD = self.tok.pad_token_id
        self.UNK = self.tok.unk_token_id
        self.BOS = self.tok.bos_token_id
        self.EOS = self.tok.eos_token_id

    def __len__(self):
        return len(self.tok)

    def encode(self, text, add_bos=False, add_eos=False, max_len=None):
        # Qwen BPE 不加特殊 token 时和普通分词一样
        ids = self.tok.encode(text, add_special_tokens=False)
        if add_bos and self.BOS is not None:
            ids = [self.BOS] + ids
        if add_eos and self.EOS is not None:
            ids = ids + [self.EOS]
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)
