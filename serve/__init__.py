"""HybridBrain serve 包初始化。

在重型库导入前屏蔽可选依赖，省 import 时间。
注意：PIL / torchvision 是 AutoTokenizer 必需的，不能屏蔽。
"""
import sys

# 只屏蔽确定无害的（timm 图像、wandb 实验跟踪）
_BLOCKED = ("timm", "wandb")
for mod in _BLOCKED:
    if mod not in sys.modules:
        sys.modules[mod] = None
