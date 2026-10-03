"""扫描语料，把 UNK 字符补进词表。

用项目自带的 load_corpus，兼容所有格式。
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_corpus


VOCAB = 'data/all_chars_vocab.json'
LCCC = 'data/lccc_50k.json'
CYRENE = 'data/cyrene_clean.json'
MIN_FREQ = 3


def scan_unk(vocab_path, texts):
    """扫描所有文本，统计 UNK 字符"""
    with open(vocab_path) as f:
        existing = set(json.load(f)['itos'])

    unk = Counter()
    total = 0
    for txt in texts:
        for ch in txt:
            total += 1
            if ch not in existing:
                unk[ch] += 1
    return unk, total


def add_chars(vocab_path, chars):
    with open(vocab_path) as f:
        data = json.load(f)
    itos = data['itos']
    existing = set(itos)
    added = []
    for ch in chars:
        if ch not in existing:
            itos.append(ch)
            added.append(ch)
    data['itos'] = itos
    data['n_chars'] = len(itos)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return len(itos), added


if __name__ == '__main__':
    print(f'[1/4] 加载 LCCC: {LCCC}')
    lccc = load_corpus(LCCC, max_samples=20000, assistant_name='昔涟')
    print(f'  {len(lccc):,} 条')

    print(f'[2/4] 加载昔涟: {CYRENE}')
    cyrene = load_corpus(CYRENE, assistant_name='昔涟')
    print(f'  {len(cyrene):,} 条')

    texts = lccc + cyrene
    print(f'  合计 {len(texts):,} 条')

    print(f'[3/4] 扫描 UNK 字符...')
    unk, total = scan_unk(VOCAB, texts)
    print(f'  总字符: {total:,}')
    print(f'  UNK 字符种类: {len(unk)}')
    print(f'  UNK 字符总数: {sum(unk.values()):,}')
    print(f'  UNK 率: {sum(unk.values())/total:.4f}')
    print('  Top 30:')
    for ch, cnt in unk.most_common(30):
        print(f'    {repr(ch)} U+{ord(ch):04X} × {cnt}')

    to_add = [ch for ch, cnt in unk.items() if cnt >= MIN_FREQ]
    print(f'\n[4/4] 写入词表（频次 >= {MIN_FREQ}）')
    print(f'  准备加入 {len(to_add)} 个字符')

    new_size, added = add_chars(VOCAB, to_add)
    print(f'  旧大小: 61050')
    print(f'  新大小: {new_size}')
    print(f'  前 50: {"".join(repr(c) for c in added[:50])}')
