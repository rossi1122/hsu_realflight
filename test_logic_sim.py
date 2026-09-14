#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_logic_sim.py — 純邏輯模擬 (vision_dive)
═══════════════════════════════════════════════════════════════════
不用任何硬體 (無 ROS / 雲台 / 相機 / 飛機). 餵一條假的飛機軌跡, 把
【虛擬目標 + 狀態機 + 影像 PID】整條跑一遍, 讓你在家就驗證:

  - SEARCH → PRE_DIVE → DIVE → PULLOUT → DONE 轉換對不對
  - 進俯衝的距離門檻 (PREDIVE_DIST_M) 在對的地方觸發
  - 虛擬目標投影 / 影像誤差 / PID 命令合不合理
  - 換座標、調門檻、改 PID 增益後, 整個任務流程還對不對

★ 這是「運動學」模擬, 不是動力學: 飛機照命令「大致」飛 (簡化積分),
  目的是驗證邏輯流程與數值合理性, 不是模擬真實氣動響應.

用法:
  python3 test_logic_sim.py                    # 移動目標 (預設)
  python3 test_logic_sim.py --static           # 靜止目標
  python3 test_logic_sim.py --verbose          # 每拍都印 (預設只印關鍵)
  python3 test_logic_sim.py --predive-dist 250 # 臨時改門檻試
"""

import argparse
import math
import sys

from config import Config
from dive_guidance import GuidanceConfig, MissionStateMachine
from virtual_source import VirtualSource, StaticTarget, MovingTarget
from virtual_target_geom import gps_to_ned, extrapolate_impact, miss_distance


def build_gcfg(cfg, moving, predive_dist):
    fx, fy, cx, cy = cfg.effective_intrinsics()
    return GuidanceConfig(
        DIVE_ANGLE_DEG=cfg.DIVE_ANGLE_DEG,
        ALPHA_NOM_DEG=cfg.ALPHA_NOM_DEG, ALPHA_DIVE_DEG=cfg.ALPHA_DIVE_DEG,
        PREDIVE_TRIGGER_MODE=('target_dist' if moving else 'gps_point'),
        PREDIVE_REACH_RADIUS_M=cfg.PREDIVE_REACH_RADIUS_M,
        PREDIVE_DIST_M=predive_dist,
        PULLOUT_AGL_M=cfg.PULLOUT_AGL_M, PULLOUT_VMAX_MPS=cfg.PULLOUT_VMAX_MPS,
        ENTRY_MIN_AGL_M=60.0,
        FX=fx, FY=fy, CX=cx, CY=cy, IMG_W=cfg.IMG_W, IMG_H=cfg.IMG_H,
        PID_LAT_KP=cfg.PID_LAT_KP, PID_LAT_KI=cfg.PID_LAT_KI, PID_LAT_KD=cfg.PID_LAT_KD,
        PID_LON_KP=cfg.PID_LON_KP, PID_LON_KI=cfg.PID_LON_KI, PID_LON_KD=cfg.PID_LON_KD,
        DIVE_TARGET_LEAD=moving,
        DIVE_AIM_V_OFFSET_PX=cfg.dive_aim_v_offset_px,
        GIMBAL_DIVE_TILT_BODY_DEG=0.0, GIMBAL_FPV_SETTLE_S=0.3,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--static', action='store_true', help='靜止目標 (預設移動)')
    ap.add_argument('--verbose', action='store_true', help='每拍都印')
    ap.add_argument('--predive-dist', type=float, default=None,
                    help='臨時改進俯衝門檻 (m)')
    args = ap.parse_args()

    moving = not args.static
    cfg = Config()
    fx, fy, cx, cy = cfg.effective_intrinsics()
    predive_dist = args.predive_dist if args.predive_dist else cfg.PREDIVE_DIST_M
    gcfg = build_gcfg(cfg, moving, predive_dist)
    sm = MissionStateMachine(gcfg, verbose=True)

    # ── 假想場景 (home = 車起點, 好算) ──
    home_lat, home_lon = cfg.MOVING_START_GPS if moving else (23.0, 120.0)

    # 馬路方位 (移動) — 用 config 兩座標
    if moving:
        hn, he = gps_to_ned(cfg.MOVING_HEADING_GPS[0], cfg.MOVING_HEADING_GPS[1],
                            cfg.MOVING_START_GPS[0], cfg.MOVING_START_GPS[1])
        road_hdg = math.degrees(math.atan2(he, hn)) % 360.0
        sn, se = 0.0, 0.0     # 目標起點 = home
        target = MovingTarget(sn, se, road_hdg, cfg.TARGET_SPEED_MPS, t0=0.0)
        virt = VirtualSource(cfg, gcfg, home_lat, home_lon, target,
                             predive_point_ne=None)
        print(f"\n[模擬] 移動目標: 起點(0,0) 方位{road_hdg:.1f}° "
              f"{cfg.TARGET_SPEED_MPS}m/s 進俯衝門檻{predive_dist:.0f}m\n")
    else:
        road_hdg = 0.0        # 靜止: 目標在正北 400m, 沿此方向進場
        target = StaticTarget(400.0, 0.0)
        virt = VirtualSource(cfg, gcfg, home_lat, home_lon, target,
                             predive_point_ne=(300.0, 0.0))  # PRE_DIVE 點北300
        print(f"\n[模擬] 靜止目標: NED(400,0) PRE_DIVE點(300,0)\n")

    # ── 飛機初始: 在目標後方, 巡航高度, 對準馬路方向 ──
    # 追擊: 飛機沿 road_hdg 方向從後方 500m 追上來
    back = 500.0
    ac_n = -back * math.cos(math.radians(road_hdg))
    ac_e = -back * math.sin(math.radians(road_hdg))
    agl = cfg.CRUISE_AGL_M            # 120
    V = cfg.V_CRUISE_MPS             # 20
    chi = road_hdg                   # 對準馬路
    pitch = cfg.ALPHA_NOM_DEG        # 巡航攻角
    dt = 0.05                        # 20Hz 模擬

    sm.arm()
    t = 0.0
    mode_prev = sm.mode
    pullout_done = False
    max_t = 120.0

    print(f"{'t':>6} {'mode':>8} {'AGL':>6} {'V':>5} {'pd_dist':>7} "
          f"{'pitch_cmd':>9} {'hdg_cmd':>8} {'active':>6}")

    while t < max_t:
        # 虛擬目標欄位
        dive_mode = (sm.mode == sm.DIVE)
        vfields = virt.get_fields({
            'n': ac_n, 'e': ac_e, 'agl': agl,
            'theta_deg': pitch, 'chi_deg': chi,
            'gimbal_locked': dive_mode, 't': t})

        # 速度向量 (NED) — 沿 chi + 俯衝時的下沉
        gamma = math.radians(pitch)      # 近似: 航跡角≈pitch (簡化)
        vn = V * math.cos(gamma) * math.cos(math.radians(chi))
        ve = V * math.cos(gamma) * math.sin(math.radians(chi))
        vd = -V * math.sin(gamma)

        inp = {
            't': t, 'guided': True,
            'agl_m': agl, 'airspeed_mps': V,
            'vn': vn, 've': ve, 'vd': vd,
            'yaw_true_deg': chi, 'pitch_deg': pitch, 'roll_deg': 0.0,
            'ac_n': ac_n, 'ac_e': ac_e,
            **vfields,
        }
        out = sm.update(inp)

        # 模式轉換印出
        if out['mode'] != mode_prev:
            print(f"\n  >>> {mode_prev} → {out['mode']} @ t={t:.1f}s "
                  f"AGL={agl:.0f} V={V:.1f} pd_dist={vfields['predive_dist_m']:.0f}\n")
            mode_prev = out['mode']

        if args.verbose or out['mode'] in ('DIVE', 'PULLOUT'):
            print(f"{t:>6.1f} {out['mode']:>8} {agl:>6.1f} {V:>5.1f} "
                  f"{vfields['predive_dist_m']:>7.0f} {out['pitch_cmd_deg']:>9.1f} "
                  f"{out['heading_cmd_deg']:>8.1f} {int(out.get('cmd_active',0)):>6}")

        # ── 簡化運動學: 飛機朝 pitch_cmd / heading_cmd 靠攏, 再積分位置 ──
        pitch += (out['pitch_cmd_deg'] - pitch) * min(1.0, 5.0 * dt)   # 一階追命令
        chi += (((out['heading_cmd_deg'] - chi + 180) % 360 - 180)
                * min(1.0, 3.0 * dt))
        chi %= 360.0
        # 俯衝時加速, 平飛時保持
        if out['mode'] == 'DIVE':
            V += 0.5 * dt * abs(pitch) / 30.0 * 10      # 粗略加速
        V = max(cfg.V_STALL_MPS, min(35.0, V))
        ac_n += vn * dt
        ac_e += ve * dt
        agl += vd * dt * (-1)    # vd 下正 → agl 減; vd 為 -V sin(gamma), gamma>0 下沉
        agl = max(0.0, agl)

        # PULLOUT 成績
        if sm.pullout_snapshot is not None and not pullout_done:
            pullout_done = True
            sp = sm.pullout_snapshot
            i_n, i_e, tf = extrapolate_impact(sp['ac_n'], sp['ac_e'], sp['agl'],
                                              sp['vn'] or 0, sp['ve'] or 0,
                                              sp['vd'] or 0.1)
            if i_n is not None:
                md = miss_distance(i_n, i_e, tf, sp['tgt_n'], sp['tgt_e'],
                                   sp['tgt_vn'], sp['tgt_ve'],
                                   sp['vn'] or 0, sp['ve'] or 0)
                print(f"\n  ★★★ 模擬成績: miss = {md['miss_total']:.1f}m "
                      f"(沿航跡 {md['miss_lon']:+.1f}, 側向 {md['miss_lat']:+.1f})\n")

        if out['mode'] == 'DONE':
            print(f"\n[模擬] 任務完成 @ t={t:.1f}s")
            break
        if out.get('abort_reason'):
            print(f"\n[模擬] ABORT: {out['abort_reason']} @ t={t:.1f}s")
            break
        t += dt

    print("\n" + "=" * 60)
    print("模擬結束. 檢查上面的模式轉換順序與觸發時機是否合理:")
    print("  SEARCH → PRE_DIVE (到門檻) → DIVE (壓-30°) → PULLOUT (70m/超速) → DONE")
    print("★ 這是運動學模擬, 數值僅供邏輯驗證; 真實飛行看實飛 log.")


if __name__ == '__main__':
    main()
