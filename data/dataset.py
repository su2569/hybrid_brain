"""数据加载：JSON → JSONL 自动缓存。

首次加载 JSON 时转成 JSONL 存到 data/translated/，
之后直接读 JSONL（快 5-10 倍，内存友好）。
"""
import gzip
import json
import os
from typing import List, Iterator, Optional

import torch
from torch.utils.data import Dataset

from tokenizer import CharTokenizer


# ============================================================
# 基础 IO
# ============================================================
CACHE_DIR = "data/translated"


def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _first_char(path: str) -> Optional[str]:
    with _open(path) as f:
        for _ in range(64):
            c = f.read(1)
            if not c:
                return None
            if not c.isspace():
                return c
    return None


def _stream_json_array(path: str, chunk_size: int = 4 << 20) -> Iterator:
    """流式解析 JSON 数组。pos 指针，避免 O(n²)。"""
    decoder = json.JSONDecoder()
    with _open(path) as f:
        while True:
            c = f.read(1)
            if not c:
                return
            if c == "[":
                break

        buf = f.read(chunk_size)
        pos = 0
        n = 0

        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1

            if pos >= len(buf):
                more = f.read(chunk_size)
                if not more:
                    return
                buf = more
                pos = 0
                continue

            if buf[pos] == "]":
                return

            try:
                obj, end = decoder.raw_decode(buf, pos)
                pos = end
                n += 1
                if n % 50000 == 0:
                    print(f"    [stream] 已解析 {n:,} 条...")
                yield obj
            except json.JSONDecodeError:
                more = f.read(chunk_size)
                if not more:
                    return
                buf = buf[pos:] + more
                pos = 0


# ============================================================
# 对象 → 文本
# ============================================================
def _dialog_to_text(dialog: List[str], assistant_name: str,
                    max_turns: int = 10) -> Optional[str]:
    if not isinstance(dialog, list):
        return None
    cleaned = []
    for u in dialog:
        if not isinstance(u, str):
            continue
        u = u.replace(" ", "").strip()
        if 2 <= len(u) <= 100:
            cleaned.append(u)
    if len(cleaned) < 2:
        return None
    cleaned = cleaned[:max_turns]
    lines = []
    for i, u in enumerate(cleaned):
        role = "用户" if i % 2 == 0 else assistant_name
        lines.append(f"{role}：{u}")
    return "\n".join(lines)


def _sample_to_text(obj: dict, assistant_name: str) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    t = obj.get("type", "text")
    if t == "qa":
        inp = (obj.get("input") or "").strip()
        out = (obj.get("output") or "").strip()
        if inp and out:
            return f"用户：{inp}\n{assistant_name}：{out}"
        return None
    if t == "text":
        return (obj.get("text") or "").strip() or None
    if t == "dialogue":
        msgs = obj.get("messages", [])
        lines = []
        for m in msgs:
            role = m.get("role", "?")
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"{role}：{content}")
        return "\n".join(lines) or None
    for k in ("text", "content"):
        if obj.get(k):
            return str(obj[k]).strip()
    return None


def _obj_to_text(obj, assistant_name: str) -> Optional[str]:
    if isinstance(obj, list):
        return _dialog_to_text(obj, assistant_name)
    if isinstance(obj, dict):
        if "dialog" in obj and isinstance(obj["dialog"], list):
            return _dialog_to_text(obj["dialog"], assistant_name)
        return _sample_to_text(obj, assistant_name)
    if isinstance(obj, str):
        return obj.strip() or None
    return None


# ============================================================
# 缓存逻辑
# ============================================================
def _cache_path(src_path: str, assistant_name: str) -> str:
    """LCCC/raw/LCCC-base_train.json → data/translated/LCCC-base_train.昔涟.jsonl"""
    name = os.path.basename(src_path)
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".json"):
        name = name[:-5]
    return os.path.join(CACHE_DIR, f"{name}.{assistant_name}.jsonl")


def _needs_rebuild(src: str, cache: str) -> bool:
    if not os.path.exists(cache):
        return True
    # 源文件更新了 → 重建
    return os.path.getmtime(src) > os.path.getmtime(cache)


def _build_cache(src_path: str, cache_path: str, assistant_name: str) -> int:
    """流式转 JSON → JSONL。返回条数。"""
    first = _first_char(src_path)
    n = 0
    tmp = cache_path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as out:
        if first == "[":
            # LCCC 原生数组
            for obj in _stream_json_array(src_path):
                txt = _obj_to_text(obj, assistant_name)
                if txt:
                    out.write(json.dumps({"text": txt}, ensure_ascii=False))
                    out.write("\n")
                    n += 1
                    if n % 50000 == 0:
                        print(f"    [cache] {n:,} 条...")

        elif first == "{":
            # 先试整体 JSON
            data = None
            try:
                with _open(src_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, MemoryError, UnicodeDecodeError):
                data = None

            if data is not None:
                items = []
                if isinstance(data, dict) and "samples" in data:
                    items = data["samples"]
                elif isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = [data]

                for obj in items:
                    txt = _obj_to_text(obj, assistant_name)
                    if txt:
                        out.write(json.dumps({"text": txt}, ensure_ascii=False))
                        out.write("\n")
                        n += 1
            else:
                # JSONL 回退
                with _open(src_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or not line.startswith("{"):
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        txt = _obj_to_text(obj, assistant_name)
                        if txt:
                            out.write(json.dumps({"text": txt}, ensure_ascii=False))
                            out.write("\n")
                            n += 1
        else:
            # 纯文本
            with _open(src_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(json.dumps({"text": line}, ensure_ascii=False))
                        out.write("\n")
                        n += 1

    os.replace(tmp, cache_path)
    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"    [cache] {cache_path} ({n:,} 条, {size_mb:.1f} MB)")
    return n


def ensure_cache(src_path: str, assistant_name: str = "昔涟") -> str:
    """确保 JSONL 缓存存在。返回缓存路径。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = _cache_path(src_path, assistant_name)

    if _needs_rebuild(src_path, cache):
        print(f"[cache] 构建 {os.path.basename(src_path)} → "
              f"{os.path.basename(cache)}")
        _build_cache(src_path, cache, assistant_name)
    else:
        print(f"[cache] 命中 {cache}")

    return cache


# ============================================================
# 加载器
# ============================================================
def _load_jsonl(path: str, max_samples: Optional[int] = None) -> List[str]:
    """逐行读 JSONL，返回 text 列表。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and len(out) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("text")
                if txt:
                    out.append(txt)
            except json.JSONDecodeError:
                continue
    return out


def load_corpus(path: str, max_samples: Optional[int] = None,
                assistant_name: str = "昔涟") -> List[str]:
    """统一入口。

    - .json / .json.gz   → 走缓存（首次转换，之后直读）
    - .jsonl / .jsonl.gz → 直接读
    - .txt / .txt.gz     → 直接读
    """
    if not os.path.exists(path):
        print(f"[warn] 文件不存在: {path}")
        return []

    lower = path.lower()

    if lower.endswith(".jsonl") or lower.endswith(".jsonl.gz"):
        return _load_jsonl(path, max_samples)

    if lower.endswith(".txt") or lower.endswith(".txt.gz"):
        out = []
        with _open(path) as f:
            for line in f:
                if max_samples and len(out) >= max_samples:
                    break
                line = line.strip()
                if line:
                    out.append(line)
        return out

    # .json → 走缓存
    cache = ensure_cache(path, assistant_name)
    return _load_jsonl(cache, max_samples)


# ============================================================
# Dataset
# ============================================================
class TextDataset(Dataset):
    """滑窗切分为 LM 样本。pad=0，label=-100。"""

    def __init__(self, texts: List[str], tok: CharTokenizer, seq_len: int):
        self.tok = tok
        self.seq_len = seq_len
        self.examples = []
        for t in texts:
            ids = tok.encode(t, add_bos=True, add_eos=True)
            for i in range(0, max(len(ids) - 1, 1), seq_len):
                chunk = ids[i:i + seq_len + 1]
                if len(chunk) >= 2:
                    self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x, y = chunk[:-1], chunk[1:]
        pad_n = self.seq_len - len(x)
        x = x + [self.tok.PAD] * pad_n
        y = y + [-100] * pad_n
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long))