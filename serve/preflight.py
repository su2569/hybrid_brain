"""服务启动前自检：检查所有路径和依赖。

用法：
    python3 serve/preflight.py
或作为模块调用：
    from serve.preflight import check_all
    check_all()  # 失败时 raise
"""
import os
from serve import logger
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 配置（与 router.py 保持一致）
# ============================================================
PATHS = {
    "HEADS_PATH": "/mnt/workspace/checkpoints/heads_qwen.pt",
    "INTENT_BGE_JOBLIB": "/mnt/workspace/checkpoints/intent_bge.joblib",
    "BGE_PATH": "/mnt/workspace/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    "RERANKER_PATH": "/mnt/workspace/models/models/BAAI--bge-reranker-base/snapshots/master",
    "GENERATOR_PATH": "/mnt/workspace/models/qwen3_cyrene_merged",
    "CHAT_GENERATOR_PATH": "/mnt/workspace/models/qwen3_cyrene_merged",
    "DUREDER_PATH": "/mnt/workspace/data6/dureader_robust-data/train.json",
}

OPTIONAL = {"RERANKER_PATH", "DUREDER_PATH"}  # 缺失不阻塞启动


# ============================================================
# 检查函数
# ============================================================
def _check_dir(path, must_files=("config.json",)):
    """目录存在 + 关键文件存在。"""
    if not os.path.isdir(path):
        return False, f"目录不存在: {path}"
    for f in must_files:
        if not os.path.exists(os.path.join(path, f)):
            return False, f"缺文件: {path}/{f}"
    return True, "ok"


def _check_file(path):
    if not os.path.isfile(path):
        return False, f"文件不存在: {path}"
    size_mb = os.path.getsize(path) / 1e6
    if size_mb < 0.001:
        return False, f"文件为空: {path}"
    return True, f"{size_mb:.1f} MB"


def _check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA 不可用"
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return True, f"{name} ({mem:.0f} GB)"
    except Exception as e:
        return False, str(e)[:100]


def _find_bge_small():
    """自动搜 BGE-small。"""
    for pat in [
        "/mnt/workspace/models/**/bge-small-zh-v1.5/**/config.json",
        "/mnt/workspace/models/**/bge-small-zh-v1.5/config.json",
    ]:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0].rsplit("/", 1)[0]
    return None


def _find_qwen3():
    """自动搜 Qwen3。"""
    for pat in [
        "/mnt/workspace/models/**/Qwen--Qwen3-1.7B/**/config.json",
        "/mnt/workspace/models/**/Qwen3-1.7B/**/config.json",
    ]:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0].rsplit("/", 1)[0]
    return None


# ============================================================
# 主检查
# ============================================================
def check_all(strict=False, verbose=True):
    """跑所有检查。返回 (ok, issues)。"""
    issues = []
    warnings = []
    
    def _p(status, name, detail):
        if not verbose:
            return
        if status == "ok":
            logger.ok("CHECK", f"{name:25s} {detail}")
        elif status == "warn":
            logger.warn("CHECK", f"{name:25s} {detail}")
        elif status == "fail":
            logger.error("CHECK", f"{name:25s} {detail}")
        else:
            logger.info("CHECK", f"{name:25s} {detail}")

    if verbose:
        logger.load("═" * 55)
        logger.load("  HybridBrain 启动前自检")
        logger.load("═" * 55)

    # 1. GPU
    if verbose:
        logger.load("[1/7] GPU")
    ok, detail = _check_gpu()
    _p("ok" if ok else "fail", "CUDA", detail)
    if not ok:
        issues.append(("CUDA", detail))

    # 2. 三头
    if verbose:
        logger.load("[2/7] Checkpoints")
    for name in ["HEADS_PATH", "INTENT_BGE_JOBLIB"]:
        path = PATHS[name]
        ok, detail = _check_file(path)
        _p("ok" if ok else "fail", name, detail)
        if not ok:
            issues.append((name, detail))

    # 3. BGE-small
    if verbose:
        logger.load("[3/7] BGE 检索器")
    bge_path = PATHS["BGE_PATH"]
    if not os.path.exists(bge_path):
        # 自动搜
        found = _find_bge_small()
        if found:
            bge_path = found
            _p("warn", "BGE_PATH", f"原路径不存在，自动找到: {found}")
        else:
            _p("fail", "BGE_PATH", f"路径不存在: {bge_path}")
            issues.append(("BGE_PATH", bge_path))
    else:
        ok, detail = _check_dir(bge_path)
        _p("ok" if ok else "fail", "BGE_PATH", detail)
        if not ok:
            issues.append(("BGE_PATH", detail))

    # 4. Reranker（可选）
    if verbose:
        logger.load("[4/7] BGE-reranker（可选）")
    rr_path = PATHS["RERANKER_PATH"]
    if not os.path.exists(rr_path):
        _p("warn", "RERANKER_PATH", f"不存在（可选，跳过 rerank）")
        warnings.append(("RERANKER_PATH", rr_path))
    else:
        ok, detail = _check_dir(rr_path)
        _p("ok" if ok else "warn", "RERANKER_PATH", detail)
        if not ok:
            warnings.append(("RERANKER_PATH", detail))

    # 5. Generator
    if verbose:
        logger.load("[5/7] RAG / Chat 生成器")
    gen_path = PATHS["GENERATOR_PATH"]
    if not os.path.exists(gen_path):
        found = _find_qwen3()
        if found:
            _p("warn", "GENERATOR_PATH", f"原路径不存在，自动找到: {found}")
        else:
            _p("fail", "GENERATOR_PATH", f"路径不存在: {gen_path}")
            issues.append(("GENERATOR_PATH", gen_path))
    else:
        ok, detail = _check_dir(gen_path,
                                must_files=("config.json",))
        _p("ok" if ok else "fail", "GENERATOR_PATH", detail)
        if not ok:
            issues.append(("GENERATOR_PATH", detail))
        # 检查 safetensors 大小
        st_files = glob.glob(os.path.join(gen_path, "*.safetensors"))
        if st_files:
            total = sum(os.path.getsize(f) for f in st_files) / 1e9
            _p("ok" if total > 1 else "warn", "  safetensors",
               f"{len(st_files)} 个, {total:.1f} GB")
            if total < 1:
                warnings.append(("safetensors", f"只有 {total:.1f} GB，可能不完整"))
    
    chat_path = PATHS["CHAT_GENERATOR_PATH"]
    if chat_path != gen_path:
        ok, detail = _check_dir(chat_path)
        _p("ok" if ok else "warn", "CHAT_GENERATOR_PATH", detail)

    # 6. DuReader 数据
    if verbose:
        logger.load("[6/7] DuReader 数据")
    dr_path = PATHS["DUREDER_PATH"]
    if not os.path.exists(dr_path):
        _p("warn", "DUREDER_PATH", f"不存在（KB 会为空或从缓存加载）")
        warnings.append(("DUREDER_PATH", dr_path))
    else:
        size_mb = os.path.getsize(dr_path) / 1e6
        _p("ok" if size_mb > 1 else "warn", "DUREDER_PATH", f"{size_mb:.1f} MB")

    # 7. 输出
    if verbose:
        logger.load("═" * 55)
        if issues:
            logger.error("CHECK", f"发现 {len(issues)} 个致命问题:")
            for name, detail in issues:
                logger.debug("CHECK", f"  - {name}: {detail}")
        if warnings:
            logger.warn("CHECK", f"发现 {len(warnings)} 个警告（不阻塞）:")
            for name, detail in warnings:
                logger.debug("CHECK", f"  - {name}: {detail}")
        if not issues and not warnings:
            logger.ok("CHECK", "所有检查通过")
        elif not issues:
            logger.ok("CHECK", "无致命问题，可以启动")
        logger.load("═" * 55)

    if strict and issues:
        raise RuntimeError(f"启动前自检失败，{len(issues)} 个问题")
    return len(issues) == 0, issues


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="有致命问题则退出码非 0")
    args = ap.parse_args()
    ok, issues = check_all(strict=args.strict)
    sys.exit(0 if ok else 1)
