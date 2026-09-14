#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_gimbal.py — 實飛前雲台動作測試 (vision_dive)
═══════════════════════════════════════════════════════════════════
只需要雲台接上 (Jetson 插電 + 網路通到 SIYI), 不用飛機 / ROS / GPS / 相機.
把實飛會用到的雲台動作一項一項跑, 讓你上機前先確認行為對不對.

跑得起來的前提: config.SIYI_IP / SIYI_PORT 指對雲台.

用法:
  python3 test_gimbal.py            # 互動選單
  python3 test_gimbal.py --all      # 依序自動跑一遍 (每步間停頓)

★ 能測 (離線):
    1. 連線 + 開機 2x zoom
    2. STANDBY: FOLLOW + 看前方 (setAngles 0, STANDBY_PITCH)
    3. TRACK:  餵假 bbox, 看雲台朝該方向轉 (分區 P)
    4. FPV 切換
    5. 讀 getAttitude
★ 測不到 (一定要上機):
    - FOLLOW 下 yaw 真跟機頭 (要飛機真的轉)
    - FPV 貼機身在空中俯衝的實際表現
  這兩項腳本會明講, 別誤以為測過了.
"""

import argparse
import sys
import time

from config import Config
from gimbal import GimbalManager


def _pause(msg="按 Enter 繼續 (Ctrl-C 中止) ..."):
    try:
        input(msg)
    except (EOFError, KeyboardInterrupt):
        print()
        raise


def test_connect(gm):
    print("\n[測試 1] 連線 + 開機 2x zoom")
    print("  預期: 雲台連上, zoom 到 2x, 回到 STANDBY 指向 (0, %.0f°)"
          % gm.cfg.GIMBAL_STANDBY_PITCH_DEG)
    ok = gm.connect()
    if not ok:
        print("  ✗ 連線失敗 — 檢查 config.SIYI_IP/PORT 與網路")
        return False
    print("  ✓ 連線成功")
    y, p, r = gm.get_attitude()
    print(f"  目前雲台角度: yaw={y:.1f} pitch={p:.1f} roll={r:.1f}")
    return True


def test_standby(gm, hold_s=5.0):
    print(f"\n[測試 2] STANDBY: FOLLOW + 看前方, 持續 {hold_s:.0f}s")
    print(f"  預期: 雲台 pitch 壓到 {gm.cfg.GIMBAL_STANDBY_PITCH_DEG:.0f}° "
          f"(看前下方), yaw 回中")
    print("  ★ 注意: 「yaw 跟機頭」要飛機真的轉才看得出, 桌上看不到 — 只能確認 pitch 到位")
    t0 = time.time()
    while time.time() - t0 < hold_s:
        gm.standby_tick()
        time.sleep(0.2)
    y, p, r = gm.get_attitude()
    print(f"  結果: yaw={y:.1f} pitch={p:.1f} "
          f"(pitch 應接近 {gm.cfg.GIMBAL_STANDBY_PITCH_DEG:.0f})")


def test_track(gm):
    print("\n[測試 3] TRACK: 餵假 bbox, 看雲台朝該方向轉")
    fx, fy, cx, cy = _intr(gm.cfg)
    w, h = gm.cfg.IMG_W, gm.cfg.IMG_H
    cases = [
        ("目標在畫面右邊 (應往右轉, yaw 減)", cx + 400, cy),
        ("目標在畫面左邊 (應往左轉, yaw 增)", cx - 400, cy),
        ("目標在畫面上方 (應抬頭, pitch 增)", cx, cy - 300),
        ("目標在畫面下方 (應低頭, pitch 減)", cx, cy + 300),
        ("目標回中心 (應停住)", cx, cy),
    ]
    for desc, u, v in cases:
        print(f"\n  → {desc}  bbox=({u:.0f},{v:.0f})")
        y0, p0, _ = gm.get_attitude()
        # 送幾拍讓雲台動 (track_tick 內部有頻率節流)
        t0 = time.time()
        while time.time() - t0 < 2.5:
            gm.track_tick(True, u, v, w, h, fx, fy)
            time.sleep(0.1)
        y1, p1, _ = gm.get_attitude()
        print(f"    yaw {y0:.1f}→{y1:.1f} (Δ{y1-y0:+.1f})  "
              f"pitch {p0:.1f}→{p1:.1f} (Δ{p1-p0:+.1f})")
        _pause()


def test_lost(gm):
    print("\n[測試 4] TRACK 丟失: has_target=False, 應保持/回指向")
    fx, fy, cx, cy = _intr(gm.cfg)
    w, h = gm.cfg.IMG_W, gm.cfg.IMG_H
    print(f"  持續送「無目標」{gm.cfg.GIMBAL_LOST_TIMEOUT_S+1:.0f}s "
          f"(超過 LOST_TIMEOUT 應回 STANDBY 指向)")
    t0 = time.time()
    while time.time() - t0 < gm.cfg.GIMBAL_LOST_TIMEOUT_S + 1.0:
        gm.track_tick(False, None, None, w, h, fx, fy)
        time.sleep(0.2)
    y, p, _ = gm.get_attitude()
    print(f"  結果: yaw={y:.1f} pitch={p:.1f}")


def test_fpv(gm):
    print("\n[測試 5] FPV 切換")
    print("  預期: 雲台切 FPV 模式 (貼機身). 桌上雲台會停在當前指向不再世界穩定.")
    print("  ★ 注意: FPV 在空中俯衝的實際表現桌上測不到, 只能確認切換命令有送出")
    gm.enter_fpv()
    time.sleep(1.0)
    print(f"  雲台模式旗標: {gm.mode} (應為 fpv)")


def _intr(cfg):
    return cfg.effective_intrinsics()[0], cfg.effective_intrinsics()[1], \
        cfg.effective_intrinsics()[2], cfg.effective_intrinsics()[3]


def run_all(gm):
    if not test_connect(gm):
        return
    _pause()
    test_standby(gm)
    _pause()
    test_track(gm)
    test_lost(gm)
    _pause()
    test_fpv(gm)
    print("\n[完成] 離線能測的都跑完了.")
    print("上機前仍要實飛確認: (a) FOLLOW yaw 跟機頭  (b) FPV 俯衝表現")


def menu(gm):
    connected = False
    while True:
        print("\n" + "=" * 50)
        print("雲台測試選單 (實飛前):")
        print("  1) 連線 + 2x zoom      2) STANDBY 看前方")
        print("  3) TRACK 假 bbox        4) TRACK 丟失回指向")
        print("  5) FPV 切換             a) 全部依序")
        print("  q) 離開")
        try:
            c = input("選擇: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if c == 'q':
            break
        if c == '1':
            connected = test_connect(gm)
        elif c in ('2', '3', '4', '5', 'a') and not connected:
            print("  ★ 請先執行 1) 連線")
        elif c == '2':
            test_standby(gm)
        elif c == '3':
            test_track(gm)
        elif c == '4':
            test_lost(gm)
        elif c == '5':
            test_fpv(gm)
        elif c == 'a':
            test_standby(gm); test_track(gm); test_lost(gm); test_fpv(gm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='依序自動跑一遍')
    args = ap.parse_args()

    cfg = Config()
    print(f"[test_gimbal] 目標雲台 {cfg.SIYI_IP}:{cfg.SIYI_PORT}")
    gm = GimbalManager(cfg)
    try:
        if args.all:
            run_all(gm)
        else:
            menu(gm)
    finally:
        gm.close()
        print("[test_gimbal] 雲台連線已關閉")


if __name__ == '__main__':
    main()
