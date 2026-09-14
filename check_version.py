#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_version.py — 確認專案裡每個檔案都是最新版 (vision_dive)
═══════════════════════════════════════════════════════════════════
2026/09/11 掛機驗證後改了一整批檔案, 很容易漏掉一兩個.
漏掉的後果不是報錯, 是【靜默失效】, 例如:
  · run_vision_flight.py 舊 + vision.py 新 → raw 影片完全不會被錄到
  · attitude_bridge.py 舊 → DIVE 的 roll/pitch 被飛控忽略 (mask 7 問題)
  · postprocess_compare.py 舊 → 後處理對齊錯位

本腳本用「關鍵字指紋」檢查每個檔案有沒有包含該版本的特徵字串.

用法:
  cd ~/catkin_ws/src/hsu_realflight/vision_dive
  python3 check_version.py
"""

import os
import sys

# 檔案 → [(關鍵字, 最少出現次數, 這個改動在做什麼)]
CHECKS = {
    'attitude_bridge.py': [
        ('RUDDER_VERIFIED', 2, '方向舵安全閘 (掛機驗過才開)'),
        ('MASK_NONE', 2, 'SEARCH 記 mask=0 表示不送姿態'),
        ('MASK_RPY = 128', 1, '★ DIVE 用 mask 128 (mask 7 實測無效)'),
        ('_remux', 0, ''),                      # 佔位, 不檢查
        ('send_heading', 2, '43002 GUIDED_CHANGE_HEADING'),
        ('MASK_ROLL_PITCH', 0, ''),             # 佔位
        ('dive_hold_heading', 1, '版本 B 真的關掉方向舵'),
        ('_rudder_to_yaw_field', 2, 'yaw 欄位 = 方向舵映射'),
        ('lateral_fallback', 3, '43002 失敗退回 legacy bank'),
        # ── 2026/09/12 ──
        ('COMMAND_STALE_S', 2, '★ 主迴圈失聯保護 (停止重送舊命令)'),
        ('_last_submit_t', 3, '★ stale watchdog 的時戳'),
        ('stale_events', 3, 'stale 次數統計'),
        ('MASK_RPY_NOTHR', 2, '★ PULLOUT mask 192 (油門交 TECS)'),
        ('speed_mps', 3, 'send_tecs_targets 可指定速度 (SEARCH 15 / 拉平 18)'),
    ],
    'run_vision_flight.py': [
        ('no-annot', 2, '--no-annot 省一路 NVENC'),
        ('grabber.recorder', 1, '★ raw 錄影掛到取像執行緒 (沒有就錄不到)'),
        ('read_with_id', 1, '去重: 不重複處理同一張影像'),
        ('vision_hb', 10, '視覺執行緒心跳看門狗'),
        ('bench_dive', 1, '--bench-dive 獨立旗標'),
        ('fc_nav_pitch_deg', 3, '★ 記錄飛控真的收到的姿態目標'),
        ('miss_pred_m', 2, '★ 逐幀 miss distance'),
        ('raw_frame_idx', 2, '對齊 raw 影片用的幀號'),
        ('BENCH_HANDOFF_ANY_PITCH', 1, '掛機交棒 DIVE'),
        ('roll_deg=s[', 1, 'submit 傳 roll 給 roll protection'),
        # ── 2026/09/12 ──
        ("tag='/巡航'", 1, '★ 撥 GUIDED 送當下高度 (不是 70m)'),
        ("tag='/拉平'", 1, 'PULLOUT/PRE_DIVE 才送 70m'),
        ('_miss_warned', 3, '★ miss 計算失敗不再靜默'),
        ('cfg.MOVING_TARGET = moving', 1, '★ 命令列模式同步回 cfg'),
        ('brg_stale', 3, 'stale 狀態記進 CSV'),
        ('bench-pitch', 1, '--bench-pitch 旗標 (DIVE 基準姿態覆寫)'),
        ('RUDDER_LIMIT_DEG', 1, '★ 從 config.py 拉方向舵參數進 bcfg'),
        ('first_frame_id', 2, 'raw 幀號偏移'),
        ("out.get('pid_u_sp'", 1,
         '★ virt_sp_ref 記真正的 PID setpoint (不是 bbox 量測值)'),
    ],
    'dive_guidance.py': [
        # PITCH_CMD_MIN_DEG 的值由下面「跨檔一致性」檢查 (字串比對太脆弱)
        ('BENCH_HANDOFF_ANY_PITCH', 3, '掛機交棒 DIVE'),
        ('dive_hold_heading', 1, '★ 版本 B 的語意載體 (沒有 A/B 就沒差別)'),
        ('BENCH_GEOMETRY_AGL_M)', 3, '掛機推頭幾何'),
        # ── 2026/09/12 ──
        ('tecs_resend = True', 3, '★ TECS 70m 提前到 PRE_DIVE'),
        ('BENCH_DIVE_PITCH_BASE_DEG', 3, '掛機 DIVE 基準姿態覆寫'),
    ],
    'vision.py': [
        ('CAM_REQUEST_FPS', 1, '★ 相機 15→30fps'),
        ('read_with_id', 1, '回傳幀號給去重用'),
        ('self.recorder', 3, '★ raw 錄影掛在取像執行緒'),
        ('def stats', 1, '相機健康統計'),
        ('reopens', 3, '相機斷線自動重開'),
        # ── 2026/09/12 ──
        ('MOVING_TARGET', 1, "★★ TARGET_MOVING 打錯字 (09/08 視覺執行緒猝死主因)"),
        ('frame_id=self.frame_id', 1, 'raw 錄影帶幀號'),
    ],
    'video_writer.py': [
        ('_remux_gst', 2, '沒有 ffmpeg 時用 gst 修時基'),
        ('start_new_session', 2, '★ Ctrl+C 不再打死 gst (影片會壞)'),
        ('_write_fps_sidecar', 2, '.fps.txt 旁檔'),
        ('enqueued', 4, '入列/寫入/丟棄分開計數'),
        ('queue_len', 3, 'raw queue 放大到 120'),
        ('died_at_t', 3, '寫入停止時刻'),
        # ── 2026/09/12 ──
        ('first_frame_id', 4, '★ raw 幀號偏移 (後處理對齊用)'),
        ('stop_requested_backlog', 3, '★ release() 會把 queue 清完'),
    ],
    'config.py': [
        ('CAM_REQUEST_FPS', 1, '★ 相機要求 fps 與標稱 fps 拆開'),
        # ── 2026/09/12 ──
        ('SEARCH_SPEED_MPS', 1, '★ SEARCH 空速與 AUTO 一致'),
        ('PREDIVE_REACH_RADIUS_M = 15.0', 1, 'PRE_DIVE 觸發半徑 30→15'),
        ('RUDDER_LIMIT_DEG', 1, '★ 方向舵迴路參數移到 config.py'),
        ('RUDDER_VSCHED', 1, '空速排程'),
    ],
    'postprocess_compare.py': [
        ('raw_frame_idx', 2, '★ 用幀號對齊 (raw 已與 YOLO 解耦, 不再 1:1)'),
        # ── 2026/09/12 ──
        ('_raw_offset', 3, '★ 讀 .fps.txt 的幀號偏移'),
    ],
    'dkf_tracker_gimbal.py': [
        # ── 2026/09/12 ──
        ('% 360.0 - 180.0', 1,
         '★ ego yaw 差分 wrap (穿越正北會產生 13000px 假位移)'),
    ],
}

# 不該再出現的字串 (出現就是舊版)
FORBIDDEN = {
    'vision.py': [('self.cfg.TARGET_MOVING',
                   '★★ config 沒有這個屬性 → AttributeError. '
                   '2026/09/08 視覺執行緒 t=754.7s 猝死就是這行')],
}

# 必須存在的新檔案
NEW_FILES = {
    'check_bench_log.py': '飛完自動判讀每個模式',
    'make_annot_video.py': '離線重建標註影片',
    'test_rudder_bench.py': '掛機方向舵驗證',
    'test_camera_bench.py': '相機取像效能對照',
}

# 文件
DOCS = {
    'RUN_CASES.md': ('本次更新摘要', '4 個 case 執行小抄'),
}


def main():
    here = os.getcwd()
    print(f"檢查目錄: {here}\n")
    bad_files = []
    total_miss = 0

    for fn, checks in CHECKS.items():
        if not os.path.exists(fn):
            print(f"✗ {fn}  —— 檔案不存在!")
            bad_files.append(fn)
            continue
        try:
            src = open(fn, encoding='utf-8').read()
        except Exception as e:
            print(f"✗ {fn}  —— 讀不到: {e}")
            bad_files.append(fn)
            continue

        miss = []
        for kw, n, why in checks:
            if n <= 0:
                continue
            if src.count(kw) < n:
                miss.append((kw, why))
        if miss:
            bad_files.append(fn)
            total_miss += len(miss)
            print(f"✗ {fn}  ({len(src.splitlines())} 行) —— 舊版, "
                  f"缺 {len(miss)} 項:")
            for kw, why in miss:
                print(f"      · {kw:<26} {why}")
        else:
            print(f"✓ {fn}  ({len(src.splitlines())} 行)")

    # ── 禁用字串 ──
    for fn, items in FORBIDDEN.items():
        if not os.path.exists(fn):
            continue
        src = open(fn, encoding='utf-8').read()
        for kw, why in items:
            # 只看實際程式碼 (略過註解行, 我們在註解裡有提到這個名字)
            hit = [ln for ln in src.splitlines()
                   if kw in ln and not ln.strip().startswith('#')]
            if hit:
                print(f"✗ {fn}  —— 還有舊寫法 `{kw}`: {why}")
                bad_files.append(fn)

    print()
    for fn, why in NEW_FILES.items():
        if os.path.exists(fn):
            print(f"✓ {fn}  ({why})")
        else:
            print(f"✗ {fn}  —— 缺這個新檔案 ({why})")
            bad_files.append(fn)

    print()
    for fn, (kw, why) in DOCS.items():
        if not os.path.exists(fn):
            print(f"✗ {fn}  —— 缺文件 ({why})")
            bad_files.append(fn)
        elif kw not in open(fn, encoding='utf-8').read():
            print(f"✗ {fn}  —— 舊版 ({why})")
            bad_files.append(fn)
        else:
            print(f"✓ {fn}")

    # ── 跨檔一致性 ──
    print("\n" + "─" * 58)
    print("跨檔一致性")
    print("─" * 58)
    try:
        dg = open('dive_guidance.py', encoding='utf-8').read()
        ab = open('attitude_bridge.py', encoding='utf-8').read()
        import re
        m1 = re.search(r'PITCH_CMD_MIN_DEG\s*=\s*(-?[\d.]+)', dg)
        m2 = re.search(r'REQUIRED_PITCH_LIM_MIN_DEG\s*=\s*(-?[\d.]+)', ab)
        v1 = float(m1.group(1)) if m1 else None
        v2 = float(m2.group(1)) if m2 else None
        print(f"  dive_guidance.PITCH_CMD_MIN_DEG          = {v1}")
        print(f"  attitude_bridge.REQUIRED_PITCH_LIM_MIN_DEG = {v2}")
        if v1 is not None and v1 == v2:
            print(f"  ✓ 兩者一致. 飛控 PTCH_LIM_MIN_DEG 也要設成 {v1:.0f}")
        else:
            print("  ✗ 不一致! 軟體要得比飛控多會被靜默夾掉 → 積分飽和")
            bad_files.append('(pitch 限幅不一致)')
    except Exception as e:
        print(f"  (檢查失敗: {e})")

    # ── 語法 ──
    print("\n" + "─" * 58)
    print("語法檢查")
    print("─" * 58)
    import py_compile
    for fn in list(CHECKS) + list(NEW_FILES):
        if not os.path.exists(fn):
            continue
        try:
            py_compile.compile(fn, doraise=True, cfile='/tmp/_vc.pyc')
            print(f"  ✓ {fn}")
        except Exception as e:
            print(f"  ✗ {fn}: {e}")
            bad_files.append(fn)

    print("\n" + "=" * 58)
    if bad_files:
        print(f"結論: 有 {len(set(bad_files))} 個檔案需要更新")
        for f in sorted(set(bad_files)):
            print(f"   - {f}")
    else:
        print("結論: 全部都是最新版 ✓")
    print("=" * 58)
    return 1 if bad_files else 0


if __name__ == '__main__':
    sys.exit(main())
