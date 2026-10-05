"""从对话里自动检测反馈信号。"""
import re

# 纠正模式（放宽：出现在句子任意位置）
CORRECT_PAT = re.compile(
    r"(?:不对|错了|不是的|不对吧|不是这样|应该是|应该说|"
    r"我的意思是|说错|搞错|其实是|并非|而是)"
    r"[，,：:。\s]*(.+)", re.DOTALL)

# 追问（弱负信号）
FOLLOWUP_PAT = re.compile(
    r"(?:再说一遍|换个说法|听不懂|什么意思|没听懂|没看懂|"
    r"重新说|讲清楚点)")

# 明确正反馈
POSITIVE_PAT = re.compile(
    r"^(?:谢谢|感谢|太棒了|厉害|完美|正解|就是这|对的|"
    r"没错|棒|牛|给力|爱了)")


def detect_feedback(query: str) -> dict:
    """返回 {'type': ..., 'correction': ..., 'confidence': ...} 或 None。

    在下一轮对话里检测用户对**上一轮回答**的反馈。
    """
    if not query:
        return None
    q = query.strip()

    # 短句"不对"单独出现
    if q in ("不对", "错了", "不对吧", "不是", "不是这样"):
        return {"type": "followup", "confidence": 0.4}

    # 纠正（最高优先级）
    m = CORRECT_PAT.search(q)
    if m:
        correction = m.group(1).strip()
        if len(correction) >= 2:
            return {
                "type": "correct",
                "correction": correction[:200],
                "confidence": 0.85,
            }

    # 追问
    if FOLLOWUP_PAT.search(q):
        return {"type": "followup", "confidence": 0.3}

    # 正反馈（只在短句里）
    if len(q) <= 20 and POSITIVE_PAT.match(q):
        return {"type": "up", "confidence": 0.6}

    return None


if __name__ == "__main__":
    tests = [
        "不对，景元是仙舟罗浮的将军",
        "你搞错了，正确答案是 A",
        "其实是 3.14",
        "错了错了",
        "换个说法",
        "谢谢",
        "太棒了！",
        "今天天气不错",
    ]
    for t in tests:
        print(f"  {t:35s} → {detect_feedback(t)}")
