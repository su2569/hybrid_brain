"""Adapter 共享工具。"""
from typing import Union


def parse_oai_content(content: Union[str, list, None]) -> dict:
    """把 OpenAI 的 content 归一为内部 {text, attachments}。

    支持：
      - str（旧客户端）
      - list[dict]（多模态）
      - None
    """
    if content is None:
        return {"text": "", "attachments": []}

    # 情况 1：纯字符串
    if isinstance(content, str):
        return {"text": content, "attachments": []}

    # 情况 2：数组
    if isinstance(content, list):
        texts = []
        attachments = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")

            if t == "text":
                texts.append(item.get("text", ""))
            elif t == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    attachments.append({
                        "type": "image",
                        "mime": "image/jpeg",
                        "source": {"kind": "url", "url": url},
                    })
            elif t == "input_audio":
                data = (item.get("input_audio") or {}).get("data")
                fmt = (item.get("input_audio") or {}).get("format", "wav")
                if data:
                    attachments.append({
                        "type": "audio",
                        "mime": f"audio/{fmt}",
                        "source": {"kind": "base64", "data": data},
                    })
            elif t == "video_url":
                url = (item.get("video_url") or {}).get("url")
                if url:
                    attachments.append({
                        "type": "video",
                        "mime": "video/mp4",
                        "source": {"kind": "url", "url": url},
                    })
            elif t == "audio_url":
                url = (item.get("audio_url") or {}).get("url")
                if url:
                    attachments.append({
                        "type": "audio",
                        "mime": "audio/wav",
                        "source": {"kind": "url", "url": url},
                    })

        return {"text": "\n".join(texts), "attachments": attachments}

    return {"text": "", "attachments": []}


def now_iso():
    """当前 ISO 8601 时间（CST）。"""
    from datetime import datetime
    from serve.time_utils import CST
    return datetime.now(CST).isoformat()
