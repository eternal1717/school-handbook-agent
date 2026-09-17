"""前端截图工具：用 CDP 驱动无头 Edge/Chrome，既能截静态页，也能模拟点击后再截。

## 为什么需要它

改前端最怕的是「接口全对、静态检查全过，但界面是坏的」。有两类问题**只有肉眼看才知道**：

1. CSS 写错导致尺寸失效。实测踩过两个：
   - `<svg>` 不写 width/height，浏览器按 300×150 的替换尺寸渲染，把按钮撑成大白十字；
   - `<span class="fill">` 是 inline 元素，`width`/`height` 直接被忽略，
     进度条宽度永远为 0——所有条看起来一样长，其实根本没渲染填充。
2. 布局错位、颜色变量没生效、文字被父容器裁掉。

这两类问题任何静态检查都发现不了，必须截图看。

## 为什么不直接用 `--screenshot`

Edge 的 `--screenshot` 参数只能截「页面刚加载完」的样子，
没法点按钮、没法展开折叠面板，也就看不到交互后的状态。
这里走 CDP（Chrome DevTools Protocol），可以 `Runtime.evaluate` 注入 JS，
于是「点开一条会话再截图」这种需求就能满足了。

## 依赖

只用标准库 + `websockets`（FastAPI/uvicorn 生态里本来就带，不需要额外安装）。
浏览器用系统自带的 Edge，不下载 Chromium。

## 用法

    # 1. 先起浏览器（headless，带调试端口）
    python tests/ui_shot.py --serve

    # 2. 再截图（另开一个终端）
    python tests/ui_shot.py --url "http://127.0.0.1:8010/#trace" --out shot.png
    python tests/ui_shot.py --url "http://127.0.0.1:8010/" --out chat.png \
        --js "document.querySelector('.conv-item').click()" --wait 8

窗口宽度默认 1440。窄屏（<1080）会触发响应式，链路徽标和小屏侧栏布局会变，
想看完整版式就用默认值。
"""
import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEBUG_PORT = 9222
CDP_HOST = f"http://127.0.0.1:{DEBUG_PORT}"

# 系统自带浏览器的常见位置。刻意不下载 Chromium：
# 一个 500MB 的浏览器只为截图不值当，而且用户对磁盘占用很敏感。
BROWSER_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser() -> str:
    for path in BROWSER_CANDIDATES:
        if os.path.exists(path):
            return path
    raise SystemExit("找不到系统浏览器（Edge / Chrome），请手动指定路径")


def serve(width: int, height: int) -> None:
    """启动一个带远程调试端口的无头浏览器，等它把端口监听起来。"""
    browser = find_browser()
    profile = Path(os.environ.get("TEMP", "/tmp")) / "zhijin-ui-shot-profile"
    args = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--remote-debugging-port={DEBUG_PORT}",
        f"--user-data-dir={profile}",
        f"--window-size={width},{height}",
        "about:blank",
    ]
    print("启动：", " ".join(args))
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 等调试端口就绪。最多等 20 秒——冷启动加上杀毒软件扫描，慢的时候要几秒。
    for _ in range(40):
        try:
            urllib.request.urlopen(f"{CDP_HOST}/json/version", timeout=1).read()
            print(f"CDP 就绪（pid={proc.pid}）。现在可以另开终端跑截图命令了。")
            return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    proc.terminate()
    raise SystemExit("CDP 端口未就绪，浏览器可能启动失败")


class CDP:
    """极简 CDP 客户端。只需要发命令、等对应 id 的回复，不必处理事件流。"""

    def __init__(self, ws):
        self.ws = ws
        self._id = 0

    async def send(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        current = self._id
        await self.ws.send(json.dumps({"id": current, "method": method, "params": params or {}}))
        while True:
            message = json.loads(await self.ws.recv())
            if message.get("id") == current:
                if "error" in message:
                    raise RuntimeError(f"{method} 失败: {message['error']}")
                return message.get("result") or {}


async def capture(url: str, out: Path, js: str, wait: float, width: int, height: int) -> None:
    import websockets

    targets = json.loads(urllib.request.urlopen(f"{CDP_HOST}/json", timeout=5).read())
    pages = [t for t in targets if t.get("type") == "page"]
    if not pages:
        raise SystemExit("没有可用的 page target")
    ws_url = pages[0]["webSocketDebuggerUrl"]

    # 截图体积可能几 MB，base64 之后更大，默认 1MB 上限会直接断开
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        cdp = CDP(ws)
        await cdp.send("Page.enable")
        await cdp.send("Runtime.enable")
        # 显式设视口，否则窗口尺寸和实际渲染尺寸可能对不上
        await cdp.send(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
        )
        await cdp.send("Page.navigate", {"url": url})
        await asyncio.sleep(wait)

        if js:
            result = await cdp.send(
                "Runtime.evaluate",
                {"expression": js, "awaitPromise": True, "returnByValue": True},
            )
            # JS 报错必须说一声。不检查的话，注入的点击脚本因为选择器没匹配到而
            # 抛 TypeError，工具却照常截图、照常打印「已保存」——
            # 于是截到的是一张什么都没发生的页面，还以为是页面本身坏了。
            if result.get("exceptionDetails"):
                detail = result["exceptionDetails"]
                text = (detail.get("exception") or {}).get("description") or detail.get("text")
                print("JS 执行出错：", (text or "").split("\n")[0])
            value = (result.get("result") or {}).get("value")
            if value is not None:
                print("JS 返回：", value)
            # 交互之后要留出接口请求 + 动画的时间，不等的话截到的是半截状态
            await asyncio.sleep(wait)

        shot = await cdp.send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(shot["data"]))
    print(f"已保存 {out}（{out.stat().st_size} bytes）")


def main() -> None:
    parser = argparse.ArgumentParser(description="无头浏览器截图工具")
    parser.add_argument("--serve", action="store_true", help="启动带调试端口的无头浏览器")
    parser.add_argument("--url", help="要截图的地址")
    parser.add_argument("--out", default="shot.png", help="输出文件路径")
    parser.add_argument("--js", default="", help="截图前要在页面里执行的 JS，可用来点击、展开")
    parser.add_argument("--wait", type=float, default=6.0, help="导航后等待秒数（SPA 要留足渲染时间）")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()

    if args.serve:
        serve(args.width, args.height)
        return
    if not args.url:
        parser.error("要么给 --serve，要么给 --url")

    try:
        asyncio.run(capture(args.url, Path(args.out), args.js, args.wait, args.width, args.height))
    except (ConnectionRefusedError, OSError) as error:
        print(f"连不上 CDP（{error}）。先跑一次 `python tests/ui_shot.py --serve`。", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
