"""连通性自检：用 openai SDK 调一次 Agnes 2.5 Flash，打印回答与 token 用量。密钥只显示前 6 位。"""

import os            # 读环境变量
import time          # 只用来算耗时
from pathlib import Path

from openai import OpenAI   # 官方 SDK；Agnes 走 OpenAI 兼容协议，所以能直接复用

ROOT = Path(__file__).resolve().parent   # 本文件所在目录，用它在旁边找 .env


def load_env(path=ROOT / ".env"):
    """把 .env 里的 KEY=VALUE 读进环境变量。手写十行，省一个 python-dotenv 依赖。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # 空行和 # 注释跳过，其余必须含等号才处理
        if not line or line.startswith("#") or "=" not in line:
            continue
        # 只按第一个等号切，值里允许再出现等号
        k, v = line.split("=", 1)
        # setdefault：系统里已设过就尊重系统的，方便临时从命令行覆盖
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def mask(key):
    """密钥只露前 6 位，else 分支防住空 key 时切片报错。"""
    return (key[:6] + "*" * 8) if key else "(empty)"


def main():
    load_env()
    # 三项配置：.env 里没写就用默认值兜底
    key = os.getenv("AGNES_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    base = os.getenv("AGNES_BASE_URL", "https://api.agnes-ai.cn/v1")
    model = os.getenv("AGNES_MODEL", "agnes-2.5-flash")

    # 建客户端：base_url 换成 Agnes 域名，90 秒超时，最多重试 2 次
    client = OpenAI(api_key=key, base_url=base, timeout=90, max_retries=2)

    t0 = time.time()
    # 一次普通对话调用，底层就是 POST {base}/chat/completions
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "你知道什么是Agent吗？用一句话描述！"}],
        temperature=0.3,   # 越低越稳定；自检要的是可复现，不是有创意
    )
    ms = (time.time() - t0) * 1000     # 毫秒
    u = resp.usage                     # SDK 已解析好的 token 统计

    print("base  :", base)
    print("model :", resp.model)       # 用服务端回的模型名，确认实际落到哪个模型
    print("key   :", mask(key))
    print("answer:", resp.choices[0].message.content.strip())   # 正文在第一个 choice 里
    print("token : prompt=%s completion=%s total=%s  latency=%.0fms" % (
        u.prompt_tokens, u.completion_tokens, u.total_tokens, ms))


if __name__ == "__main__":
    main()
