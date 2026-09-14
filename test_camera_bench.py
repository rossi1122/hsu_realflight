#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_camera_bench.py — 找出取像鏈的瓶頸 (vision_dive)
═══════════════════════════════════════════════════════════════════
背景 (2026/09/11):
  v4l2-ctl 直接量測: 擷取卡能給 60.96 fps @ 1920x1080 YUYV, USB 3.0 5Gbps.
  但 run_vision_flight 實際只拿到 15 fps —— 差 4 倍, 瓶頸在軟體.

  兩個可疑點:
    1. 程式要求 MJPG, 但這張卡只有 YUYV → 要求被無聲忽略, 走未壓縮,
       OpenCV 要在 CPU 上做 YUYV→BGR (1920x1080 每幀 4.15MB 進 6MB 出).
    2. 拿到的 15 剛好是要求值 30 的一半, 而卡片根本沒有 15fps 模式.
       這是 BUFFERSIZE=1 + 消費端稍慢時的典型「鎖半速」徵狀.

本腳本把幾種取像方式各跑一段時間, 直接比 fps, 讓你挑最快的那個.

用法:
  python3 test_camera_bench.py                 # 全部跑一遍, 每種 8 秒
  python3 test_camera_bench.py --secs 15
  python3 test_camera_bench.py --only gst_nv   # 只跑某一種

跑之前建議先:
  sudo nvpmodel -q && sudo nvpmodel -m 0 && sudo jetson_clocks
"""

import argparse
import sys
import time

try:
    import cv2
except ImportError:
    print("需要 opencv-python")
    sys.exit(1)

W, H = 1920, 1080


def _drain(cap, secs, label):
    """跑 secs 秒, 回傳 (fps, 取得幀數, 平均每幀耗時 ms)."""
    if cap is None or not cap.isOpened():
        print(f"  {label:22s} → 開不起來")
        return None
    # 暖機
    for _ in range(10):
        cap.read()
    n = 0
    worst = 0.0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs:
        t1 = time.perf_counter()
        ok, f = cap.read()
        dt = time.perf_counter() - t1
        if ok and f is not None:
            n += 1
            worst = max(worst, dt)
    span = time.perf_counter() - t0
    cap.release()
    fps = n / span
    print(f"  {label:22s} → {fps:6.2f} fps   ({n} 幀 / {span:.1f}s, "
          f"最慢一幀 {worst * 1000:.0f} ms)")
    return fps


def m_v4l2(fourcc, req_fps, bufsize, label, secs):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"  {label:22s} → VideoCapture 開不起來")
        return None
    if bufsize is not None:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, bufsize)
        except Exception:
            pass
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(W))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(H))
    cap.set(cv2.CAP_PROP_FPS, float(req_fps))
    return _drain(cap, secs, label)


def m_gst(pipeline, label, secs):
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return _drain(cap, secs, label)


GST_PLAIN = (
    f"v4l2src device=/dev/video0 io-mode=2 ! "
    f"video/x-raw,format=YUY2,width={W},height={H},framerate=60/1 ! "
    f"videoconvert ! video/x-raw,format=BGR ! "
    f"appsink drop=true max-buffers=1 sync=false"
)

# Jetson 硬體轉換: YUY2 → NVMM NV12 → BGRx, 最後一步 BGRx→BGR 才回 CPU
GST_NVVIDCONV = (
    f"v4l2src device=/dev/video0 io-mode=2 ! "
    f"video/x-raw,format=YUY2,width={W},height={H},framerate=60/1 ! "
    f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
    f"nvvidconv ! video/x-raw,format=BGRx ! "
    f"videoconvert ! video/x-raw,format=BGR ! "
    f"appsink drop=true max-buffers=1 sync=false"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secs', type=float, default=8.0, help='每種方法跑幾秒')
    ap.add_argument('--only', help='只跑指定的一種 (見下方 key)')
    args = ap.parse_args()

    print(__doc__)
    print(f"OpenCV {cv2.__version__}")
    gst_ok = 'GStreamer' in cv2.getBuildInformation() and \
             'GStreamer:                   YES' in cv2.getBuildInformation()
    print(f"OpenCV GStreamer 支援: {'YES' if gst_ok else '不確定 (下面會知道)'}\n")

    tests = {
        'now': lambda: m_v4l2('MJPG', 30, 1, '現況 (MJPG req, 30fps)',
                              args.secs),
        'yuyv30': lambda: m_v4l2('YUYV', 30, 1, 'YUYV 明示, 30fps',
                                 args.secs),
        'yuyv60': lambda: m_v4l2('YUYV', 60, 1, 'YUYV 明示, 60fps',
                                 args.secs),
        'yuyv60_buf3': lambda: m_v4l2('YUYV', 60, 3, 'YUYV 60fps, buffer=3',
                                      args.secs),
        'gst': lambda: m_gst(GST_PLAIN, 'GStreamer videoconvert', args.secs),
        'gst_nv': lambda: m_gst(GST_NVVIDCONV, 'GStreamer nvvidconv (HW)',
                                args.secs),
    }

    if args.only:
        if args.only not in tests:
            print(f"沒有這個測試: {args.only}. 可用: {', '.join(tests)}")
            sys.exit(1)
        tests = {args.only: tests[args.only]}

    print("每種方法之間會等 1 秒讓裝置釋放.\n")
    res = {}
    for k, fn in tests.items():
        try:
            res[k] = fn()
        except Exception as e:
            print(f"  {k:22s} → 例外: {e}")
            res[k] = None
        time.sleep(1.0)

    print("\n" + "=" * 58)
    good = {k: v for k, v in res.items() if v}
    if not good:
        print("全部失敗 — 檢查 /dev/video0 有沒有被別的程式佔用")
        return
    best = max(good, key=good.get)
    base = res.get('now')
    print(f"最快: {best}  ({good[best]:.1f} fps)")
    if base:
        print(f"現況: now   ({base:.1f} fps)"
              f"   → 提升 {good[best] / base:.2f}x")
    print()
    print("判讀:")
    print("  · yuyv60 明顯 > yuyv30  → 之前是被『鎖半速』, 改要求 60 即可")
    print("  · gst/gst_nv 明顯勝出   → 瓶頸是 OpenCV 的色彩轉換,")
    print("                            FrameGrabber 應改用 GStreamer pipeline")
    print("  · 全部都差不多 ~15      → CPU 真的跑不動, 考慮降解析度或換卡")
    print("=" * 58)
    print("\n※ 視覺迴圈 (YOLO) 上限約 24 fps, 所以相機拿到 30 就夠了,")
    print("  不需要追 60.")


if __name__ == '__main__':
    main()
