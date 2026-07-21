#!/usr/bin/env python3
"""生成自包含的机械图纸样例图片（纯标准库，无 PIL / 无联网）。

真实 CAD 图纸由 runtime/MechVQA/benchmark_data 提供（需联网下载）。本脚本只为
batch_test.py 提供离线、可重放的管线校验用例：用 zlib + struct 写出 8 位灰度
PNG，画一些线框/圆弧，保证 vLLM 服务能正常 ingest 图片字节，便于在没有真实数据
时跑通一键批量测试与 mock 模式。生成的图片不是真实图纸，模型输出仅供管线校验。
"""

from __future__ import annotations

import math
import struct
import sys
import zlib
from pathlib import Path

WIDTH = 320
HEIGHT = 240
FIXTURES: list[dict] = [
    {"name": "sample_shaft.png", "kind": "shaft"},
    {"name": "sample_flange.png", "kind": "flange"},
    {"name": "sample_plate.png", "kind": "plate"},
]


def make_canvas() -> list[list[int]]:
    return [[255] * WIDTH for _ in range(HEIGHT)]


def draw_h_line(canvas: list[list[int]], x1: int, x2: int, y: int, value: int = 0) -> None:
    if x2 < x1:
        x1, x2 = x2, x1
    for x in range(max(0, x1), min(WIDTH, x2 + 1)):
        canvas[y][x] = value


def draw_v_line(canvas: list[list[int]], x: int, y1: int, y2: int, value: int = 0) -> None:
    if y2 < y1:
        y1, y2 = y2, y1
    for y in range(max(0, y1), min(HEIGHT, y2 + 1)):
        canvas[y][x] = value


def draw_rect(canvas: list[list[int]], x1: int, x2: int, y1: int, y2: int, value: int = 0) -> None:
    draw_h_line(canvas, x1, x2, y1, value)
    draw_h_line(canvas, x1, x2, y2, value)
    draw_v_line(canvas, x1, y1, y2, value)
    draw_v_line(canvas, x2, y1, y2, value)


def draw_circle(canvas: list[list[int]], cx: int, cy: int, r: int, value: int = 0) -> None:
    steps = max(8, int(2 * math.pi * r))
    for i in range(steps):
        angle = 2 * math.pi * i / steps
        x = round(cx + r * math.cos(angle))
        y = round(cy + r * math.sin(angle))
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            canvas[y][x] = value


def draw_shaft(canvas: list[list[int]]) -> None:
    # 阶梯轴侧视：两个矩形拼接 + 中心线
    draw_rect(canvas, 40, 120, 90, 150)
    draw_rect(canvas, 120, 250, 100, 140)
    draw_h_line(canvas, 30, 260, 120, value=128)
    draw_circle(canvas, 60, 120, 8)
    draw_circle(canvas, 60, 120, 4)


def draw_flange(canvas: list[list[int]]) -> None:
    # 法兰盘：外圆 + 内孔 + 螺孔圆周
    draw_circle(canvas, 160, 120, 90)
    draw_circle(canvas, 160, 120, 40)
    for i in range(8):
        angle = 2 * math.pi * i / 8
        x = round(160 + 68 * math.cos(angle))
        y = round(120 + 68 * math.sin(angle))
        draw_circle(canvas, x, y, 6)
    draw_h_line(canvas, 50, 270, 120, value=128)


def draw_plate(canvas: list[list[int]]) -> None:
    # 板件：矩形 + 4 个角孔 + 尺寸线
    draw_rect(canvas, 50, 270, 50, 190)
    for cx, cy in [(70, 70), (250, 70), (70, 170), (250, 170)]:
        draw_circle(canvas, cx, cy, 8)
    draw_h_line(canvas, 50, 270, 30, value=128)
    draw_v_line(canvas, 30, 50, 190, value=128)


def write_png(path: Path, canvas: list[list[int]]) -> None:
    raw = bytearray()
    for row in canvas:
        raw.append(0)  # filter type 0
        raw.extend(row)
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 0, 0, 0, 0)  # 8-bit grayscale
    png = signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "fixtures" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fixture in FIXTURES:
        canvas = make_canvas()
        if fixture["kind"] == "shaft":
            draw_shaft(canvas)
        elif fixture["kind"] == "flange":
            draw_flange(canvas)
        elif fixture["kind"] == "plate":
            draw_plate(canvas)
        target = out_dir / fixture["name"]
        write_png(target, canvas)
        print(f"wrote {target} ({target.stat().st_size} bytes)")
    print(f"fixtures dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
