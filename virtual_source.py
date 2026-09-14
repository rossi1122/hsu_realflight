# -*- coding: utf-8 -*-
"""
virtual_source.py — 虛擬目標源 (vision_dive)
═══════════════════════════════════════════════════════════════════
控制路徑的「影像誤差」來源: 不是真實 YOLO, 而是已知 GPS 的虛擬目標.
每拍用飛機真實狀態把目標 NED 投影成畫面 (u, v) → 餵給 PID.

結構 = 共同外殼 + 可換目標 (靜止/移動只差「目標當下在哪」):

  VirtualSource (外殼: 投影 / 進俯衝距離 / get_fields 輸出格式, 靜止移動共用)
    └─ target.now(t) → (n, e, vn, ve)   ← 唯一差異點
         ├─ StaticTarget : 固定 NED, 速度 0            (進俯衝用「到固定 PRE_DIVE 點」距離)
         └─ MovingTarget : 起點 + 航向×速度×(t−t0)     (進俯衝用「到目標當下位置」距離)

★ 投影用 dive_guidance.project_body_camera (與狀態機 lead_setpoint 同一條鏈),
  不另寫一份, 保證虛擬目標投影 = PID setpoint 投影 同鏈.
★ SEARCH/PRE_DIVE (雲台世界穩定, 理想指向) → 目標居中; DIVE (FPV) → body 投影.
"""

import math

from virtual_target_geom import gps_to_ned
from dive_guidance import project_body_camera, R_G2C, rot_y_np


# ══════════════════════════════════════════════════════════════════
#  目標: 靜止 / 移動 (只負責「當下在哪、速度多少」)
# ══════════════════════════════════════════════════════════════════
class StaticTarget:
    """靜止目標: 固定 NED, 速度 0."""
    moving = False

    def __init__(self, n, e):
        self.n0, self.e0 = n, e

    def now(self, t):
        return self.n0, self.e0, 0.0, 0.0


class MovingTarget:
    """
    等速直線目標: 起點 (n0,e0), 方位 heading_deg, 速度 speed (m/s).
    t0 = 任務起點時刻 (切 GUIDED); 之前停在起點.
    位置 = 起點 + 速度向量 × max(0, t − t0).
    """
    moving = True

    def __init__(self, n0, e0, heading_deg, speed_mps, t0):
        self.n0, self.e0 = n0, e0
        self.t0 = t0
        self.heading_deg = heading_deg
        self.speed = speed_mps
        self.vn = speed_mps * math.cos(math.radians(heading_deg))
        self.ve = speed_mps * math.sin(math.radians(heading_deg))

    def now(self, t):
        dt = max(0.0, t - self.t0)
        return (self.n0 + self.vn * dt, self.e0 + self.ve * dt,
                self.vn, self.ve)


# ══════════════════════════════════════════════════════════════════
#  外殼: 投影 + 進俯衝距離 + get_fields (靜止/移動共用)
# ══════════════════════════════════════════════════════════════════
class VirtualSource:
    def __init__(self, cfg, gcfg, home_lat, home_lon, target,
                 predive_point_ne=None):
        """
        target: StaticTarget 或 MovingTarget.
        predive_point_ne: (n,e) 固定 PRE_DIVE 點 (靜止版用);
                          None → 移動版, 進俯衝改用「到目標當下位置」水平距.
        """
        self.cfg = cfg
        self.gcfg = gcfg
        self.home_lat, self.home_lon = home_lat, home_lon
        self.target = target
        self.pd_point = predive_point_ne
        tilt = math.radians(getattr(gcfg, 'GIMBAL_DIVE_TILT_BODY_DEG', 0.0))
        self.R_bc = R_G2C @ rot_y_np(tilt)
        mode = 'MOVING' if target.moving else 'STATIC'
        if self.pd_point is not None:
            print(f"[VirtualSource] {mode} 目標起點 NED=({target.n0:.1f},{target.e0:.1f}) "
                  f"PRE_DIVE 點 NED=({self.pd_point[0]:.1f},{self.pd_point[1]:.1f})")
        else:
            print(f"[VirtualSource] {mode} 目標起點 NED=({target.n0:.1f},{target.e0:.1f}) "
                  f"進俯衝用「到目標水平距 <= {cfg.PREDIVE_DIST_M:.0f}m」")

    def target_now(self, t):
        """目標當下 NED + 速度 (n, e, vn, ve)."""
        return self.target.now(t)

    def get_fields(self, ac):
        """
        ac: dict — n, e, agl, theta_deg, chi_deg, gimbal_locked, t
        回傳餵給 MissionStateMachine.update 的視覺欄位 dict.
        """
        g = self.gcfg
        an, ae = ac['n'], ac['e']
        agl = ac['agl']
        t = ac['t']

        tn, te, tvn, tve = self.target_now(t)
        dxn, dye = tn - an, te - ae
        ground = math.hypot(dxn, dye)
        slant = math.hypot(ground, agl) if agl is not None else None

        # 進俯衝距離: 固定點 (靜止) 或 到目標當下位置 (移動)
        if self.pd_point is not None:
            predive_dist = math.hypot(self.pd_point[0] - an, self.pd_point[1] - ae)
        else:
            predive_dist = ground     # 到目標水平距

        half_diag = math.hypot(g.IMG_W / 2.0, g.IMG_H / 2.0)

        if ac.get('gimbal_locked', False) and agl is not None:
            # DIVE: body-fixed 真實投影
            u, v = project_body_camera(
                dxn, dye, agl, ac['theta_deg'], ac['chi_deg'], self.R_bc, g)
            if u is None:
                detected, u, v, off = False, None, None, 1.0
            else:
                detected = (0 <= u < g.IMG_W) and (0 <= v < g.IMG_H)
                off = math.hypot(u - g.CX, v - g.CY) / half_diag
        else:
            # SEARCH / PRE_DIVE: 世界穩定雲台理想指向 → 目標居中
            detected, u, v, off = True, g.CX, g.CY, 0.0

        # bbox 尺寸: 控制不用, 僅供 log/overlay, 粗略隨斜距縮放
        if slant is not None and slant > 3.0:
            bw = max(g.FX * 4.5 / slant, 4.0)
            bh = max(g.FY * 2.0 / slant, 3.0)
        else:
            bw = bh = None

        return {
            'tgt_n': tn, 'tgt_e': te, 'tgt_valid': True,
            'tgt_vn': tvn, 'tgt_ve': tve,          # 移動版有值 → miss 外插算車位移
            'slant_m': slant, 'ground_m': ground,
            'predive_dist_m': predive_dist,
            'bbox_detected': detected,
            'bbox_u': u, 'bbox_v': v,
            'bbox_w': bw if detected else None,
            'bbox_h': bh if detected else None,
            'bbox_center_offset_norm': off,
        }
