"""生成 astrbot / codex / dsh 工具的扩充数据。"""
import json, random, os
from collections import Counter

random.seed(42)

# ============ 新工具定义 ============
NEW_TOOLS = {
    "astrbot_search": {
        "params": ["query"],
        "desc": "用 AstrBot 插件搜索信息",
        "queries": [
            ("直接", ["用 astrbot 搜一下{q}", "astrbot 搜索{q}", "帮我用 astrbot 查{q}"]),
            ("祈使", ["astrbot 帮我搜{q}", "麻烦用 astrbot 搜索{q}", "用 astrbot 查一下{q}"]),
            ("疑问", ["能用 astrbot 搜到{q}吗", "astrbot 能搜{q}吗"]),
        ],
    },
    "codex_exec": {
        "params": ["cmd", "cwd"],
        "desc": "在 Codex 环境执行 shell 命令",
        "queries": [
            ("直接", ["codex 执行{cmd}", "在{cwd}执行{cmd}", "codex 在{cwd}跑{cmd}"]),
            ("祈使", ["帮我用 codex 执行{cmd}", "在{cwd}下用 codex 跑{cmd}"]),
            ("疑问", ["能用 codex 在{cwd}跑{cmd}吗", "codex 在{cwd}能执行{cmd}吗"]),
        ],
    },
    "dsh_read": {
        "params": ["path"],
        "desc": "读取 DeepSeekHarness 工作区文件",
        "queries": [
            ("直接", ["dsh 读一下{path}", "用 dsh 读{path}", "dsh 看看{path}"]),
            ("祈使", ["麻烦用 dsh 读{path}", "帮我用 dsh 打开{path}"]),
            ("疑问", ["能用 dsh 读{path}吗", "dsh 能读{path}吗"]),
        ],
    },
}

# 参数值池
VALS = {
    "query": ["量子计算", "最新 AI 新闻", "唐诗三百首", "红烧肉做法", "Python 教程",
              "股票行情", "天气预告", "股票代码", "开源项目", "论文 2024",
              "上海地铁", "星座运势", "历史事件", "科学新闻"],
    "cmd":   ["ls -la", "git status", "cat README.md", "python main.py",
              "pip install -r requirements.txt", "pytest -v", "make build"],
    "cwd":   ["/home/user", "/workspace", "/tmp", "/var/log",
              "/mnt/data", "~/project"],
    "path":  ["README.md", "/etc/hosts", "config.yaml", "data.json",
              "main.py", "requirements.txt", "logs/app.log"],
}

def fmt_call(name, params):
    p_str = json.dumps(params, ensure_ascii=False, separators=(',', ':'))
    return f'<call type="action" name="{name}" params=\'{p_str}\'>调用{name}</call>'

def fmt_query(tpl, vals):
    """把 {q}/{cmd}/{cwd}/{path} 替换成实际值"""
    keys = list(vals.keys())
    # 每个模板的占位符不同（q/cmd/cwd/path）
    out = tpl
    for k, v in vals.items():
        out = out.replace(f"{{{k}}}", v)
    return out

generated = []
for tool, meta in NEW_TOOLS.items():
    params = meta["params"]
    # 生成 70 条/工具
    n_per_tool = 70
    for _ in range(n_per_tool):
        # 每个参数随机取值
        vals = {}
        for p in params:
            if p == "query": vals["q"] = random.choice(VALS["query"])
            elif p == "cmd": vals["cmd"] = random.choice(VALS["cmd"])
            elif p == "cwd": vals["cwd"] = random.choice(VALS["cwd"])
            elif p == "path": vals["path"] = random.choice(VALS["path"])
        
        # 随机选一种措辞
        variants = meta["queries"]
        _, tpl_list = random.choice(variants)
        tpl = random.choice(tpl_list)
        q = fmt_query(tpl, vals)
        
        # 构造 call params
        call_params = {}
        if "query" in params: call_params["query"] = vals["q"]
        if "cmd" in params: call_params["cmd"] = vals["cmd"]
        if "cwd" in params: call_params["cwd"] = vals["cwd"]
        if "path" in params: call_params["path"] = vals["path"]
        
        a = fmt_call(tool, call_params)
        
        generated.append({
            "user": q,
            "assistant": a,
            "tool": tool,
        })

# 打乱
random.shuffle(generated)

# 拆 train / val（90/10）
n_val = max(20, len(generated) // 10)
val = generated[:n_val]
train = generated[n_val:]

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"  {path}: {len(items)}")

write_jsonl("data/tool_lora_v2/new_train.jsonl", train)
write_jsonl("data/tool_lora_v2/new_val.jsonl", val)

# 统计
print("\n工具分布:")
for k, v in Counter(x["tool"] for x in generated).most_common():
    print(f"  {k}: {v}")

print(f"\n[ok] 生成 {len(generated)} 条（train={len(train)} val={len(val)}）")
