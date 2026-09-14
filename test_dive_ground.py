#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_dive_ground.py — 地面 DIVE 舵面 + 雲台行為測試 (vision_dive)
═══════════════════════════════════════════════════════════════════
在地面強制進入 DIVE 並「鎖在 DIVE」(不觸發 PULLOUT), 你用鍵盤模擬
「目標在畫面左/右/上/下」, 觀察:
  - 飛控舵面反應 (elevator/aileron/rudder) — 需真連飛控 + GUIDED
  - 雲台反應 (FPV) — 需真連雲台

╔═══════════════════════════════════════════════════════════════╗
║ ⚠⚠ 安全鐵則: 一定要先【拆槳】(remove propellers)!            ║
║    DIVE 段油門排程可能上油門, 沒拆槳飛機會衝出去或傷人.        ║
╚═══════════════════════════════════════════════════════════════╝

需要的硬體 (缺的部分該項就跳過, 程式會提示):
  - 飛控通電 + MAVROS 連線 + 切 GUIDED  → 才看得到舵面
  - 雲台連線                            → 才看得到雲台
  (兩者可只接一個; 舵面測試務必拆槳)

用法:
  python3 test_dive_ground.py               # 舵面 + 雲台都測
  python3 test_dive_ground.py --gimbal-only # 只測雲台 (不連飛控/MAVROS)
  python3 test_dive_ground.py --no-gimbal   # 只測舵面

鍵盤 (進 DIVE 後):
  a/d = 目標左/右 (看 aileron/rudder 橫向修)
  w/s = 目標上/下 (看 elevator 縱向修)
  c   = 目標回中心
  空白= 目標消失一下 (看掉幀凍結)
  q   = 結束
"""

import argparse
import sys
import threading
import time

from config import Config
from dive_guidance import GuidanceConfig, MissionStateMachine


class Keyboard:
    """非阻塞鍵盤 (單鍵)."""
    def __init__(self):
        self.key = None
        self.run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self.run:
                self.key = sys.stdin.read(1)
                if self.key == 'q':
                    self.run = False
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def get(self):
        k = self.key
        self.key = None
        return k


def build_gcfg(cfg):
    fx, fy, cx, cy = cfg.effective_intrinsics()
    return GuidanceConfig(
        DIVE_ANGLE_DEG=cfg.DIVE_ANGLE_DEG,
        ALPHA_NOM_DEG=cfg.ALPHA_NOM_DEG, ALPHA_DIVE_DEG=cfg.ALPHA_DIVE_DEG,
        # ★ 鎖在 DIVE: PULLOUT 門檻設到不可能觸發
        PULLOUT_AGL_M=-1000.0, PULLOUT_VMAX_MPS=9999.0,
        ABORT_MIN_AGL_M=-1000.0, ABORT_TARGET_LOST_S=9999.0,
        FX=fx, FY=fy, CX=cx, CY=cy, IMG_W=cfg.IMG_W, IMG_H=cfg.IMG_H,
        PID_LAT_KP=cfg.PID_LAT_KP, PID_LAT_KI=cfg.PID_LAT_KI, PID_LAT_KD=cfg.PID_LAT_KD,
        PID_LON_KP=cfg.PID_LON_KP, PID_LON_KI=cfg.PID_LON_KI, PID_LON_KD=cfg.PID_LON_KD,
        DIVE_AIM_V_OFFSET_PX=cfg.dive_aim_v_offset_px,
        GIMBAL_DIVE_TILT_BODY_DEG=0.0, GIMBAL_FPV_SETTLE_S=0.0,
    )


def force_into_dive(sm, cfg):
    """餵假狀態把狀態機推進 DIVE (跳過真實幾何)."""
    fx, fy, cx, cy = cfg.effective_intrinsics()
    sm.arm()
    # 直接呼叫內部轉換 (繞過 SEARCH/PRE_DIVE 的幾何條件)
    sm.mode = sm.PRE_DIVE
    sm._predive_t0 = 0.0
    sm._predive_agl0 = cfg.CRUISE_AGL_M
    sm._pushover_started = True
    sm._pitch_cmd = -(cfg.DIVE_ANGLE_DEG) + cfg.ALPHA_DIVE_DEG
    sm._heading_cmd = 0.0
    # 進 DIVE
    sm._enter_dive({'pitch_deg': sm._pitch_cmd, 'ac_n': 0, 'ac_e': 0}, 0.0,
                   cfg.DIVE_ANGLE_DEG)
    print(f"[force] 已強制進入 DIVE, PULLOUT 已鎖 (不會觸發)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gimbal-only', action='store_true')
    ap.add_argument('--no-gimbal', action='store_true')
    args = ap.parse_args()

    cfg = Config()
    fx, fy, cx, cy = cfg.effective_intrinsics()
    gcfg = build_gcfg(cfg)

    print(__doc__)
    print("=" * 60)
    ans = input("★ 已確認【拆槳】了嗎? 輸入 yes 繼續: ").strip().lower()
    if ans != 'yes':
        print("未確認拆槳, 中止.")
        sys.exit(0)

    # ── 雲台 (可選) ──
    gimbal = None
    if not args.no_gimbal:
        try:
            from gimbal import GimbalManager
            gimbal = GimbalManager(cfg)
            if gimbal.connect():
                gimbal.enter_fpv()      # DIVE = FPV
                print("[雲台] 已連線並切 FPV")
            else:
                print("[雲台] 連線失敗, 跳過雲台")
                gimbal = None
        except Exception as e:
            print(f"[雲台] 不可用 ({e}), 跳過")
            gimbal = None

    # ── 飛控 bridge (可選; 只測雲台就不連) ──
    bridge = None
    io = None
    if not args.gimbal_only:
        try:
            import rospy
            from attitude_bridge import ArduPilotAttitudeBridge, AttitudeBridgeConfig
            rospy.init_node('test_dive_ground', anonymous=True)
            bcfg = AttitudeBridgeConfig()
            bcfg.THR_TRIM = cfg.THR_TRIM
            bcfg.THR_MIN = cfg.THR_MIN
            bridge = ArduPilotAttitudeBridge(dive_angle_deg=cfg.DIVE_ANGLE_DEG, cfg=bcfg)
            bridge.start()
            print("[飛控] bridge 已啟動 — 請確認飛控已切 GUIDED 才會動舵面")
            print("[飛控] ⚠ 再次確認拆槳!")
        except Exception as e:
            print(f"[飛控] MAVROS 不可用 ({e}), 只測雲台")
            bridge = None

    if gimbal is None and bridge is None:
        print("\n[中止] 雲台和飛控都沒連上, 沒東西可測.")
        sys.exit(1)

    # ── 強制進 DIVE ──
    sm = MissionStateMachine(gcfg, verbose=True)
    force_into_dive(sm, cfg)

    kb = Keyboard()
    print("\n" + "=" * 60)
    print("進入 DIVE 測試. 鍵盤: a/d=左/右 w/s=上/下 c=中心 空白=消失 q=結束")
    print("=" * 60 + "\n")

    # 假的飛機狀態 (固定, 只讓 DIVE 邏輯跑)
    agl = cfg.CRUISE_AGL_M
    pitch = -(cfg.DIVE_ANGLE_DEG)
    # 目標在畫面的位置 (你用鍵盤移動)
    off_u, off_v = 0.0, 0.0     # 相對中心的像素偏移
    detected = True
    t = 0.0
    dt = 0.05

    try:
        while kb.run:
            k = kb.get()
            if k == 'a': off_u -= 100
            elif k == 'd': off_u += 100
            elif k == 'w': off_v -= 100
            elif k == 's': off_v += 100
            elif k == 'c': off_u, off_v = 0.0, 0.0
            elif k == ' ': detected = not detected
            elif k == 'q': break

            u = cx + off_u
            v = cy + off_v
            inp = {
                't': t, 'guided': True, 'agl_m': agl, 'airspeed_mps': 20.0,
                'vn': 0.0, 've': 0.0, 'vd': 10.0,
                'yaw_true_deg': 0.0, 'pitch_deg': pitch, 'roll_deg': 0.0,
                'ac_n': 0.0, 'ac_e': 0.0,
                'tgt_n': 100.0, 'tgt_e': 0.0, 'tgt_valid': True,
                'tgt_vn': 0.0, 'tgt_ve': 0.0,
                'slant_m': 150.0, 'ground_m': 100.0,
                'bbox_detected': detected,
                'bbox_u': u if detected else None,
                'bbox_v': v if detected else None,
                'bbox_w': 40, 'bbox_h': 30,
                'bbox_center_offset_norm': (abs(off_u) + abs(off_v)) / 1000.0,
            }
            out = sm.update(inp)

            # 送舵面 (飛控)
            if bridge is not None:
                bridge.submit(out, 0.0, 0.0, 20.0, roll_deg=0.0)
                dbg = bridge.get_debug()
                brg_info = (f"舵面→ pitch={dbg.get('brg_pitch_sent','')} "
                            f"roll={dbg.get('brg_roll','')} "
                            f"yaw={dbg.get('brg_yaw_sent','')} "
                            f"thr={dbg.get('brg_thr','')}")
            else:
                brg_info = "(無飛控)"

            # 雲台 FPV 下不送角度 (body-fixed), 只讀回姿態
            if gimbal is not None:
                gy, gp, gr = gimbal.get_attitude()
                gim_info = f"雲台 yaw={gy:.1f} pitch={gp:.1f}"
            else:
                gim_info = "(無雲台)"

            tgt_desc = "消失" if not detected else \
                f"u{'+' if off_u>=0 else ''}{off_u:.0f} v{'+' if off_v>=0 else ''}{off_v:.0f}"
            print(f"\r目標[{tgt_desc:>12}] pitch_cmd={out['pitch_cmd_deg']:+6.1f} "
                  f"hdg_cmd={out['heading_cmd_deg']:5.1f} | {brg_info} | {gim_info}   ",
                  end='', flush=True)

            t += dt
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        kb.run = False
        print("\n\n[結束] 收尾中 ...")
        if bridge is not None:
            try: bridge.stop()
            except Exception: pass
        if gimbal is not None:
            gimbal.close()
        print("[結束] 完成. 記得檢查: 目標右→舵面/雲台往右修? 目標下→壓頭?")


if __name__ == '__main__':
    main()
