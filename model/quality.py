"""输出质量检测：严重异常直接丢弃，轻微异常进人工审核。"""
from collections import Counter


REPEAT_REQUEST_KEYWORDS = (
    "再说一遍", "重复", "再来一遍", "说几遍", "念一遍",
    "重复到", "说100遍", "说一百遍", "一直说", "一直重复",
)


class QualityChecker:
    MIN_TOKENS = 5
    MAX_REPEAT_NGRAM = 2

    def check(self, output_text, input_text="", consensus=None):
        severe = self._severe(output_text, input_text)
        if severe:
            return "severe", severe
        mild = self._mild(output_text, input_text, consensus)
        if mild:
            return "mild", mild
        return "ok", ""

    def _severe(self, text, input_text):
        if not text:
            return "empty"
        if len(text) < self.MIN_TOKENS:
            return f"too_short({len(text)})"
        if self._is_garbled(text):
            return "garbled"
        if self._is_excessive_repeat(text, input_text):
            return "excessive_repeat"
        return None

    def _is_garbled(self, text):
        has_content = any(
            c.isalnum() or "\u4e00" <= c <= "\u9fff" for c in text)
        if not has_content:
            return True
        if len(text) > 10:
            cnt = Counter(text)
            rare = sum(v for k, v in cnt.items()
                       if not (k.isalnum() or "\u4e00" <= k <= "\u9fff"))
            if rare / len(text) > 0.5:
                return True
        run, max_run = 0, 0
        for c in text:
            if c.isalnum() or "\u4e00" <= c <= "\u9fff":
                run = 0
            else:
                run += 1
                max_run = max(max_run, run)
        return max_run > 10

    def _is_excessive_repeat(self, text, input_text):
        if self._user_requests_repeat(input_text):
            return False
        if len(text) < 6:
            return False
        ngrams = [text[i:i + 3] for i in range(len(text) - 2)]
        cnt = Counter(ngrams)
        if not cnt:
            return False
        return cnt.most_common(1)[0][1] > self.MAX_REPEAT_NGRAM

    def _user_requests_repeat(self, input_text):
        if not input_text:
            return False
        return any(k in input_text for k in REPEAT_REQUEST_KEYWORDS)

    def _mild(self, text, input_text, consensus):
        if consensus is not None and consensus < 0.5:
            return f"low_consensus({consensus:.2f})"
        if input_text and not self._is_relevant(text, input_text):
            return "irrelevant"
        return None

    def _is_relevant(self, text, input_text):
        a = set(input_text)
        b = set(text)
        if not a:
            return True
        return len(a & b) / len(a) > 0.1
