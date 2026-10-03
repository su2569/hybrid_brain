"""加载公共数据集（data3-8）。"""
import csv
import json
import os


def load_cnewsum(path="/mnt/data3/train.json", max_n=100_000):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                if 20 <= len(text) <= 2000:
                    out.append(text)
            except json.JSONDecodeError:
                continue
    return out


def load_couplet(path="/mnt/data4/train.csv", max_n=100_000):
    """中文对联。CSV 单列：每行是 '上联 下联'"""
    out = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            if line == "text1":
                continue
            if 5 <= len(line) <= 200:
                out.append(line)
    return out


def load_poems(path="/mnt/data5/en_poetry_train.json", max_n=5000):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for obj in data[:max_n]:
        instruction = (obj.get("instruction") or "").strip()
        output = (obj.get("output") or "").strip()
        if instruction and output:
            out.append(f"{instruction}\n{output}")
    return out


def load_dureader(path="/mnt/data6/train.csv", max_n=100_000):
    """中文问答。CSV 带 BOM。"""
    out = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return out
        header = [h.strip().lstrip("\ufeff") for h in header]
        try:
            i1 = header.index("text1")
            i2 = header.index("text2")
        except ValueError:
            return out

        for i, row in enumerate(reader):
            if i >= max_n:
                break
            if len(row) <= max(i1, i2):
                continue
            t1 = (row[i1] or "").strip()
            t2 = (row[i2] or "").strip()
            for part in t1.split("[SEP]"):
                part = part.strip()
                if 10 <= len(part) <= 1000:
                    out.append(part)
            if 10 <= len(t2) <= 1000:
                out.append(t2)
    return out


def load_medical(path="/mnt/data7/chinese_medical_train_sampled.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for obj in data:
        instruction = (obj.get("instruction") or "").strip()
        output = (obj.get("output") or "").strip()
        if instruction and output:
            out.append(f"用户：{instruction}\n助手：{output}")
    return out


def load_ner(path="/mnt/data8/train.csv", max_n=50_000):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            chars = []
            for p in parts:
                if p == "O" or (len(p) >= 2 and p[1] == "-"):
                    break
                chars.append(p)
            if len(chars) >= 10:
                out.append("".join(chars))
    return out


def load_all_public(verbose=True):
    loaders = [
        ("CNewSum", load_cnewsum),
        # ("couplet", load_couplet),  # 暂时去掉
        # ("poems(en)", load_poems),
        # ("DuReader", load_dureader),
        # ("medical", load_medical),
        # ("NER", load_ner),
    ]
    all_texts = []
    for name, fn in loaders:
        try:
            texts = fn()
            all_texts.extend(texts)
            if verbose:
                print(f"  [augment] {name}: {len(texts):,} 条")
        except Exception as e:
            if verbose:
                print(f"  [augment] {name}: 跳过 ({type(e).__name__}: {e})")
    return all_texts
