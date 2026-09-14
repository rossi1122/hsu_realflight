# -*- coding: utf-8 -*-
"""
dive_guidance.py — 終端俯衝導引律 (virtual_dive_pid 版: 影像 PID)
═══════════════════════════════════════════════════════════════════
來源: virtual_dive/dive_guidance.py (MPC 版), DIVE 段控制律改為影像 PID.
MPC (DiveMPC / CasADi / IPOPT) 已整個移除, 本資料夾不依賴 casadi.

狀態機 SEARCH → PRE_DIVE → DIVE → PULLOUT(交棒過渡) → DONE(TECS 平飛), +ABORT; 與 MPC 版
完全相同, 只有 DIVE 段的 `_update_dive()` 改呼叫 DivePID.

DIVE 段控制架構 (純影像迴路, 之後接 YOLO+DKF 控制器一行不動):
  ┌ bbox (u, v) ──► 量測低通 ──► e = (u_f, v_f) − (u*, v*)
  │                                   │   像素 → 角度 (atan(px/f))
  │   setpoint (u*, v*):              ▼
  │     靜止: 畫面中心           橫向 PID(e_u) → Δψ   heading_cmd = 當下航向 + Δψ
  │     動態: 中心 + 前置偏移    縱向 PID(e_v) → Δθ   pitch_cmd   = 當下 pitch + Δθ
  │     (lead_setpoint_px)       (兩軸都是增量式: 命令 = 量測 + 偏置, 對一階
  │                               內迴圈而言 Δ/τ 即角速率 → P 項自帶積分效果)
  │                                   ▼
  └──────────► 變化率限幅 + 絕對限幅 + (可選) 輸出低通 ──► bridge / plant

前置瞄準做在 setpoint 而不是控制器: 「讓目標本體穩定停在畫面中心偏
N 像素處」, PID 始終只壓 bbox, 實飛 YOLO 框的就是目標本體, 直接可用.

★ 座標約定 (全程 NED, 不用 MAVROS ENU):
    位置 n 北 / e 東 / agl 向上為正;  航向 0=北 90=東 (順時針);
    gamma 負 = 下降;  pitch 負 = 機頭向下;
    影像 u 右正 v 下正;  相機 body_zero: cam_x=body_y(右), cam_y=body_z(下),
    cam_z=body_x(前).
    ⇒ 目標在畫面右 (u>0) → 航向加大 (右轉); 目標在畫面下 (v>0) → pitch 減小.
"""

import math
import numpy as np


# ════════════════════════════════════════════════════════════════════
#  導引律參數 (GuidanceConfig)
# ════════════════════════════════════════════════════════════════════

class GuidanceConfig:
    """導引律所有可調參數."""

    # ── 任務幾何 ─────────────────────────────────────────────
    DIVE_ANGLE_DEG = 30.0
    ALPHA_NOM_DEG = 3.5
    ALPHA_DIVE_DEG = 1.3      # 俯衝攻角 (瞄準/前饋用; log 校)
    PREDIVE_SLANT_MARGIN = 1.30
    DIVE_ENTRY_DEPRESSION_TOL_DEG = 3.0

    # ── 相機內參 (像素 → 角度用; run_sim/run_flight 從 MissionConfig 覆寫) ──
    FX = 1066.7
    FY = 1059.5
    CX = 959.4932
    CY = 545.0292
    IMG_W = 1920
    IMG_H = 1080

    # ── SEARCH → PRE_DIVE 進入條件 ──────────────────────────
    # 觸發模式:
    #   'slant'      = 舊行為 (估距 + 居中 streak)
    #   'gps_point'  = 靜止版: 水平到達固定 PRE_DIVE GPS 點半徑內
    #   'target_dist'= 移動版: 飛機到目標當下位置水平距 ≤ PREDIVE_DIST_M
    #   後兩者都只比 predive_dist_m 跟門檻 (來源由 virtual_source 決定).
    PREDIVE_TRIGGER_MODE = 'slant'
    PREDIVE_REACH_RADIUS_M = 15.0   # gps_point 模式門檻 (由 config 覆寫)
    PREDIVE_DIST_M = 200.0          # target_dist 模式門檻 (30° 幾何: 120m 高→~208m)

    # ★ 地面測試用開關: SEARCH 是否「朝目標飛」還是「保持當下機頭」.
    #   False (預設, 實飛): heading_cmd = 指向目標方位角 (正常行為, 朝目標直飛).
    #   True  (地面測試):   heading_cmd = 當下機頭 → 切 GUIDED 不打 yaw, 純平飛.
    #   地面靜止時機頭亂指, 朝目標會 yaw 一大角度嚇人; 開此開關可乾淨測平飛/pitch/油門.
    #   ※ 實飛務必設回 False, 否則 SEARCH 不會對準目標.
    SEARCH_HOLD_HEADING = False
    ENTRY_CENTER_NORM_MAX = 0.30
    ENTRY_VALID_STREAK = 20
    ENTRY_MIN_AGL_M = 60.0
    BENCH_GEOMETRY_AGL_M = 0.0    # >0 (掛機測試): slant/PULLOUT 門檻幾何用此
                                  # AGL 下限, 讓地面上也能走完流程
    # ★ 2026/09/12 掛機專用: DIVE 縱向增量的【基準姿態】覆寫.
    #   DIVE 的 pitch 命令是 pitch_cmd = pitch_now + dtheta (相對實際姿態的
    #   增量). 桌上實際 pitch ≈ +1.2°, 所以一進 DIVE, pitch_cmd 就從 PRE_DIVE
    #   的 -28.7 掉回 +2 附近 —— 舵面回平, 看不到俯衝姿態.
    #   設成 -20 之後基準變成俯衝姿態, pitch_cmd ≈ -20 + dtheta, 升降舵會停在
    #   俯衝位置, 才看得出 DIVE 段的縱向行為.
    #   None = 用實際姿態 (實飛一律 None).
    BENCH_DIVE_PITCH_BASE_DEG = None
    BENCH_HANDOFF_ANY_PITCH = False  # True (掛機): PRE_DIVE→DIVE 不檢查實際
                                  # pitch 是否壓到位 (桌上飛機不會低頭)
    BENCH_FAKE_CLOSURE_MPS = 0.0  # >0 (--bench-full): PRE_DIVE 假裝以此速率
                                  # 逼近目標, 讓 PUSH 在桌上也會觸發

    # ── PRE_DIVE: 定高逼近 → 距錐界+margin 直接下 -30 姿態命令
    #   (壓多快由飛控/機體決定) → pitch 壓到容差內交棒 DIVE.
    #   可行前提: DIVE 是純追擊影像 PID, 不挑交棒幾何 (dep 24° 或 35° 都收斂).
    PREDIVE_PUSH_MARGIN_M = 20.0     # simple: 距俯衝錐邊界多少公尺前下壓頭命令
                                     #   (MC 掃描最佳 20; 45 起步太早 → 淺滑段長)
    PREDIVE_HANDOFF_PITCH_TOL_DEG = 6.0   # simple: pitch 距目標值此容差內交棒
    PREDIVE_TIMEOUT_S = 45.0

    # ── DIVE / 相機 (虛擬目標: body_zero 常數 R_bc, 無實體雲台) ──
    GIMBAL_DIVE_TILT_BODY_DEG = 0.0
    GIMBAL_FPV_SETTLE_S = 0.0
    DIVE_SETTLE_HOLD_PITCH = True
    RBC_ABORT_IF_TARGET_DRIFT = True
    RBC_DRIFT_NORM_MAX = 0.75
    RBC_DRIFT_FRAMES = 15

    # ── 動態目標前置瞄準 (setpoint 偏移) ──────────────────────
    DIVE_TARGET_LEAD = True

    # ★ DIVE 橫向 (yaw) 模式切換 — 兩個版本並存:
    #   False (預設, 版本 A): DIVE yaw 用影像 PID 修 (橫向追目標).
    #   True  (版本 B):       進 DIVE 那刻凍結航向, DIVE 期間不修 yaw,
    #                          純沿進入時的 heading 俯衝. 縱向 pitch 兩版都照修.
    #   共線追擊 (飛機已對準馬路) 時, 版本 B = 沿馬路直線俯衝不擺頭.
    DIVE_HOLD_HEADING = False

    # ★ 實飛 FPV 攻角補償: 雲台 tilt≈0, bbox 居中 = 機軸指目標, 但速度向量
    #   在其下 ALPHA_DIVE 度 → 系統性打短. 把縱向 setpoint 下移補回:
    #   v* += FY·tan(ALPHA_DIVE) ≈ +24px @1.3°. run_flight/主程式從此值覆寫,
    #   模擬 (R_bc 已含 tilt) 設 0 不重複補. 見 README「攻角補償」.
    DIVE_AIM_V_OFFSET_PX = 0.0

    # ── PULLOUT ──────────────────────────────────────────────
    PULLOUT_AGL_M = 70.0
    PULLOUT_LEAD_S = 0.50     # ★ 死區前置: 預測 lead 秒後 AGL ≤ 門檻就觸發
                              #   (log 校: 命令→響應死區 ~0.48s, 不補谷底貼 50m)
    PULLOUT_VMAX_MPS = 30.0          # ★ 超速觸發: 空速到此值即提前拉平 (安全優先;
                                     #   正常剖面峰值 ~36 → 幾乎每趟由此觸發)
    PULLOUT_HANDOFF_S = 0.5          # 交棒過渡: 強制 pitch=0 送此秒數再放手
    PULLOUT_HANDOFF_PITCH_DEG = 0.0  #   (讓飛控 hold 的「最後一筆」是安全值)
    TECS_HOLD_ALT_M = 70.0           # 交給 TECS 保持的高度 (切 GUIDED 即預送)
    TECS_SPEED_MPS = 18.0            # 交給 TECS 的目標空速

    # ══ ★ 影像 PID (DIVE 段) ═══════════════════════════════════
    # 誤差單位: 度 (像素偏移經 atan(px/f) 轉視角), 輸出單位: 度.
    # 橫向 (u → Δψ): heading_cmd = 當下航向 + Δψ, 對 plant 而言 Δψ/τ_χ 即
    #   轉彎率 → Kp 類似「每 1° 影像誤差命令多少度航向偏置」.
    PID_LAT_KP = 4.5
    PID_LAT_KI = 0.90
    PID_LAT_KD = 0.60
    PID_LAT_I_LIMIT_DEG = 8.0        # 積分項輸出上限 (抗飽和)
    PID_LAT_OUT_LIMIT_DEG = 30.0     # Δψ 上限 (= 原 CHI_CMD_RANGE 的一半, 保守)

    # 縱向 (v → Δθ): pitch_cmd = 當下 pitch + Δθ (同橫向, 增量式).
    PID_LON_KP = 1.3
    PID_LON_KI = 0.40
    PID_LON_KD = 0.30
    PID_LON_I_LIMIT_DEG = 8.0
    PID_LON_OUT_LIMIT_DEG = 15.0     # Δθ 上限

    PID_SP_FILT_TAU_S = 0.0   # setpoint 低通 (實驗: 加了單調變差, 預設關)
    # setpoint 預測前置: 補償「死區+迴路滯後」造成的移動 setpoint 追蹤延遲
    #   sp_used = sp + sp_rate · LEAD;  靜止目標 sp 不動, 自動無效
    PID_SP_LEAD_S = 0.0     # 實驗結論: 外推 setpoint 會用錯 t_go, 預設關

    # 濾波 (一階低通時間常數, 秒; 0 = 關)
    PID_MEAS_TAU_S = 0.10            # bbox 量測低通 (實飛 YOLO 抖動主要靠這層)
    PID_D_TAU_S = 0.20               # D 項微分濾波
    PID_OUT_TAU_S = 0.0              # 命令輸出低通 (預設關, 模擬抖再開)

    # 命令變化率上限 (°/s, 每拍按 dt 換算)
    PID_HEADING_RATE_MAX_DPS = 20.0  # 與 plant CHI_RATE_MAX 一致
    PID_PITCH_RATE_MAX_DPS = 25.0    # 與 plant GAMMA_RATE_MAX 一致

    # ── 安全 / ABORT ─────────────────────────────────────────
    ABORT_MIN_AGL_M = 55.0
    ABORT_TARGET_LOST_S = 1.5
    ABORT_PITCH_DEG = 12.0
    ABORT_HOLD_S = 2.5

    # ── 最終命令限幅 ─────────────────────────────────────────
    PITCH_CMD_MIN_DEG = -40.0        # ★ 必須與飛控 PTCH_LIM_MIN_DEG 一致!
                                     #   軟體要得比飛控多 → 命令被靜默夾掉,
                                     #   而抗積分飽和是拿這裡的值判斷的 →
                                     #   看不見的積分飽和 (2026/09 實飛曾發生
                                     #   PID 要 -42° 被飛控的 -35° 夾掉).
                                     #   (None → 自動算成 -(DIVE_ANGLE+20))
    PITCH_CMD_MAX_DEG = 15.0

    def __init__(self, **overrides):
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise KeyError(f"GuidanceConfig 沒有參數 {k}")
            setattr(self, k, v)
        if self.PITCH_CMD_MIN_DEG is None:
            self.PITCH_CMD_MIN_DEG = -(self.DIVE_ANGLE_DEG + 20.0)


# ════════════════════════════════════════════════════════════════════
#  角度 / 旋轉工具
# ════════════════════════════════════════════════════════════════════

def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def rot_y_np(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

def rot_z_np(ps):
    c, s = math.cos(ps), math.sin(ps)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

# Gimbal(body_zero) → Camera (CV: X右 Y下 Z前)
R_G2C = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)


def project_body_camera(dn, de, agl, theta_deg, chi_deg, R_bc, cfg):
    """NED 相對向量 (飛機→點, z 向下 = +agl) → body-fixed 相機像素 (u, v).
    與 virtual_target._project_body_camera 同一條旋轉鏈 (roll=0 近似).
    回傳 (u, v) 或 (None, None) 若點在相機後方."""
    v_w = np.array([dn, de, agl])
    R_wb = rot_y_np(math.radians(theta_deg)) @ rot_z_np(math.radians(chi_deg))
    v_cam = R_bc @ (R_wb @ v_w)
    if v_cam[2] <= 0.5:
        return None, None
    return (cfg.CX + cfg.FX * v_cam[0] / v_cam[2],
            cfg.CY + cfg.FY * v_cam[1] / v_cam[2])


# ════════════════════════════════════════════════════════════════════
#  濾波 / 單軸 PID
# ════════════════════════════════════════════════════════════════════

class Filter1st:
    """一階低通 y += (x − y)·dt/(τ+dt). τ ≤ 0 → 直通."""

    def __init__(self, tau_s):
        self.tau = float(tau_s)
        self.y = None

    def reset(self, y0=None):
        self.y = y0

    def step(self, x, dt):
        if self.tau <= 0.0 or self.y is None:
            self.y = x
            return x
        a = dt / (self.tau + dt)
        self.y += a * (x - self.y)
        return self.y


class PIDAxis:
    """單軸 PID: 濾波微分、積分限幅 + 飽和停積、輸出限幅.
    update(err, dt, freeze=False) → out;  .p/.i/.d 保留各項供記錄."""

    def __init__(self, kp, ki, kd, i_limit, out_limit, d_tau_s):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit, self.out_limit = i_limit, out_limit
        self._d_filt = Filter1st(d_tau_s)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self._prev_err = None
        self._d_filt.reset()
        self.p = self.i = self.d = self.out = 0.0

    def update(self, err, dt, freeze=False):
        self.p = self.kp * err

        if self._prev_err is None or dt <= 0.0:
            d_raw = 0.0
        else:
            d_raw = (err - self._prev_err) / dt
        self._prev_err = err
        self.d = self.kd * self._d_filt.step(d_raw, dt)

        # 抗飽和: 輸出已飽和且誤差同向時停積; 目標掉幀時凍結
        unsat = self.p + self.integral + self.d
        saturating = (abs(unsat) >= self.out_limit
                      and (unsat * err) > 0.0)
        if not freeze and not saturating:
            self.integral += self.ki * err * dt
            self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        self.i = self.integral

        self.out = max(-self.out_limit,
                       min(self.out_limit, self.p + self.i + self.d))
        return self.out


# ════════════════════════════════════════════════════════════════════
#  DivePID — DIVE 段影像 PID (取代 DiveMPC)
# ════════════════════════════════════════════════════════════════════

class DivePID:
    """
    輸入: bbox 像素 (u, v) + setpoint (u*, v*) + 當下 pitch / 航向 (NED, 度) + dt
    輸出: pitch_cmd_deg, heading_cmd_deg (絕對, NED)

    不需要飛機位置/速度/目標座標 — 只看畫面 + 姿態. 前置瞄準由 setpoint 帶入.
    增量式: 命令 = 當下量測 + PID 偏置 (Δθ, Δψ). θ_ff 只在目標掉幀無命令時退用.
    """

    def __init__(self, cfg: GuidanceConfig):
        self.cfg = cfg
        self.theta_ff = -cfg.DIVE_ANGLE_DEG + cfg.ALPHA_DIVE_DEG
        self.lat = PIDAxis(cfg.PID_LAT_KP, cfg.PID_LAT_KI, cfg.PID_LAT_KD,
                           cfg.PID_LAT_I_LIMIT_DEG, cfg.PID_LAT_OUT_LIMIT_DEG,
                           cfg.PID_D_TAU_S)
        self.lon = PIDAxis(cfg.PID_LON_KP, cfg.PID_LON_KI, cfg.PID_LON_KD,
                           cfg.PID_LON_I_LIMIT_DEG, cfg.PID_LON_OUT_LIMIT_DEG,
                           cfg.PID_D_TAU_S)
        self._u_filt = Filter1st(cfg.PID_MEAS_TAU_S)
        self._v_filt = Filter1st(cfg.PID_MEAS_TAU_S)
        self._pitch_out_filt = Filter1st(cfg.PID_OUT_TAU_S)
        self._hdg_out_filt = Filter1st(cfg.PID_OUT_TAU_S)
        self._spu_rate = Filter1st(0.15)   # setpoint 速率估計 (px/s, 輕濾波)
        self._spv_rate = Filter1st(0.15)
        self._sp_prev = None
        self.reset()

    def reset(self, pitch0=None, heading0=None):
        self.lat.reset()
        self.lon.reset()
        self._u_filt.reset()
        self._v_filt.reset()
        self._pitch_out_filt.reset()
        self._hdg_out_filt.reset()
        self._spu_rate.reset()
        self._spv_rate.reset()
        self._sp_prev = None
        self.pitch_cmd = pitch0 if pitch0 is not None else self.theta_ff
        self.heading_cmd = heading0 if heading0 is not None else 0.0
        self._have_cmd = (pitch0 is not None and heading0 is not None)
        self.last_compute_ms = 0.0
        self.info = {'u_err_deg': 0.0, 'v_err_deg': 0.0,
                     'dpsi': 0.0, 'dtheta': 0.0, 'detected': 0}

    # ------------------------------------------------------------------
    def _px_to_deg(self, dpx, f):
        return math.degrees(math.atan2(dpx, f))

    def update(self, u, v, u_sp, v_sp, detected, pitch_now_deg, heading_now_deg, dt):
        import time as _time
        t0 = _time.perf_counter()
        cfg = self.cfg

        if not detected or u is None or v is None:
            # 目標掉幀: 保持上一拍命令, 積分凍結 (ABORT 由狀態機計時)
            self.lat.update(0.0, dt, freeze=True)
            self.lon.update(0.0, dt, freeze=True)
            self.info.update({'detected': 0})
            self.last_compute_ms = (_time.perf_counter() - t0) * 1e3
            return self.pitch_cmd, self.heading_cmd

        # ── setpoint 預測前置 (動態目標: sp 隨前置幾何漂移, 補死區滯後) ──
        if self._sp_prev is not None and dt > 0:
            ru = self._spu_rate.step((u_sp - self._sp_prev[0]) / dt, dt)
            rv = self._spv_rate.step((v_sp - self._sp_prev[1]) / dt, dt)
        else:
            ru = self._spu_rate.step(0.0, dt)
            rv = self._spv_rate.step(0.0, dt)
        self._sp_prev = (u_sp, v_sp)
        u_sp_used = u_sp + ru * cfg.PID_SP_LEAD_S
        v_sp_used = v_sp + rv * cfg.PID_SP_LEAD_S

        u_f = self._u_filt.step(u, dt)
        v_f = self._v_filt.step(v, dt)
        e_u = self._px_to_deg(u_f - u_sp_used, cfg.FX)     # 右正
        e_v = self._px_to_deg(v_f - v_sp_used, cfg.FY)     # 下正

        dpsi = self.lat.update(e_u, dt)                # 目標在右 → 右轉
        dtheta = -self.lon.update(e_v, dt)             # 目標在下 → 壓頭

        pitch_raw = pitch_now_deg + dtheta
        hdg_raw = heading_now_deg + dpsi

        # 變化率限幅 (相對上一拍命令)
        if self._have_cmd:
            dp_max = cfg.PID_PITCH_RATE_MAX_DPS * dt
            dh_max = cfg.PID_HEADING_RATE_MAX_DPS * dt
            pitch_raw = self.pitch_cmd + max(-dp_max, min(dp_max,
                                             pitch_raw - self.pitch_cmd))
            dh = wrap_deg(hdg_raw - self.heading_cmd)
            hdg_raw = self.heading_cmd + max(-dh_max, min(dh_max, dh))
        self._have_cmd = True

        # 輸出低通 (預設關)
        pitch_raw = self._pitch_out_filt.step(pitch_raw, dt)
        if self._hdg_out_filt.tau > 0.0:
            # 航向過濾要用差值避免 ±180 跳變
            if self._hdg_out_filt.y is None:
                self._hdg_out_filt.y = hdg_raw
            self._hdg_out_filt.y += (dt / (self._hdg_out_filt.tau + dt)) * \
                wrap_deg(hdg_raw - self._hdg_out_filt.y)
            hdg_raw = self._hdg_out_filt.y

        self.pitch_cmd = max(cfg.PITCH_CMD_MIN_DEG,
                             min(cfg.PITCH_CMD_MAX_DEG, pitch_raw))
        self.heading_cmd = wrap_deg(hdg_raw)

        self.info.update({'u_err_deg': e_u, 'v_err_deg': e_v,
                          'dpsi': dpsi, 'dtheta': dtheta, 'detected': 1})
        self.last_compute_ms = (_time.perf_counter() - t0) * 1e3
        return self.pitch_cmd, self.heading_cmd


# ════════════════════════════════════════════════════════════════════
#  前置瞄準: 幾何 → 像素 setpoint
# ════════════════════════════════════════════════════════════════════

def lead_aim_point(inp, agl):
    """等速前置瞄準點 (NED): aim = tgt + v_tgt·t_go,
    t_go = slant / 相對速度沿視線接近率. 靜止目標退化為 tgt.
    實飛: tgt_vn/tgt_ve 由 DKF 提供."""
    tgt_n, tgt_e = inp['tgt_n'], inp['tgt_e']
    tvn = inp.get('tgt_vn') or 0.0
    tve = inp.get('tgt_ve') or 0.0
    if abs(tvn) < 0.1 and abs(tve) < 0.1:
        return tgt_n, tgt_e
    ac_n = inp.get('ac_n', 0.0) or 0.0
    ac_e = inp.get('ac_e', 0.0) or 0.0
    vn = inp.get('vn') or 0.0
    ve = inp.get('ve') or 0.0
    vd = inp.get('vd') or 0.0
    rn, re, rd = tgt_n - ac_n, tgt_e - ac_e, agl
    slant = math.sqrt(rn * rn + re * re + rd * rd)
    if slant < 5.0:
        return tgt_n, tgt_e
    # t_go = 到落地的時間. 俯衝中用 AGL/下降率 (與成績外插同一物理,
    # 超速提前拉平時尤其重要); 未俯衝 (vd 小) 退用斜距/接近率.
    if vd > 2.0:
        t_go = agl / vd
    else:
        closing = ((vn - tvn) * rn + (ve - tve) * re + vd * rd) / slant
        t_go = slant / max(closing, 3.0)
    t_go = min(t_go, 30.0)
    return tgt_n + tvn * t_go, tgt_e + tve * t_go


def lead_setpoint_px(inp, agl, R_bc, cfg):
    """
    把「前置點相對目標本體」的視角差投影成像素, 產生 PID setpoint (u*, v*):
        u* = CX + (u_tgt − u_aim),  v* = CY + (v_tgt − v_aim)
    意義: 前置點居中 ⟺ 目標本體停在 (u*, v*).
    用當下姿態 (pitch_deg, 航向) 與 NED 幾何投影; 靜止目標 → 畫面中心.
    投影失敗 (點在相機後方) → 退回畫面中心.
    """
    if not cfg.DIVE_TARGET_LEAD or inp.get('tgt_n') is None or agl is None:
        return cfg.CX, cfg.CY
    tvn = inp.get('tgt_vn') or 0.0
    tve = inp.get('tgt_ve') or 0.0
    if abs(tvn) < 0.1 and abs(tve) < 0.1:
        return cfg.CX, cfg.CY
    aim_n, aim_e = lead_aim_point(inp, agl)
    ac_n = inp.get('ac_n', 0.0) or 0.0
    ac_e = inp.get('ac_e', 0.0) or 0.0
    theta = inp.get('pitch_deg') or 0.0
    vn, ve = inp.get('vn'), inp.get('ve')
    if vn is not None and (vn * vn + ve * ve) > 4.0:
        chi = math.degrees(math.atan2(ve, vn))
    else:
        chi = inp.get('yaw_true_deg', 0.0) or 0.0
    u_t, v_t = project_body_camera(inp['tgt_n'] - ac_n, inp['tgt_e'] - ac_e,
                                   agl, theta, chi, R_bc, cfg)
    u_a, v_a = project_body_camera(aim_n - ac_n, aim_e - ac_e,
                                   agl, theta, chi, R_bc, cfg)
    if u_t is None or u_a is None:
        return cfg.CX, cfg.CY
    return cfg.CX + (u_t - u_a), cfg.CY + (v_t - v_a)


# ════════════════════════════════════════════════════════════════════
#  MissionStateMachine — SEARCH → PRE_DIVE → DIVE → PULLOUT → DONE
#  (+ ABORT 逃生門)
# ════════════════════════════════════════════════════════════════════

class MissionStateMachine:
    """
    每拍呼叫 update(inp dict) → 回傳 dict:
        mode, cmd_active, is_abort, is_pullout,
        pitch_cmd_deg, heading_cmd_deg, pid_*, ...
    PULLOUT 觸發瞬間會把成績快照放進 self.pullout_snapshot
    (呼叫端拿去算外插落點 / miss distance).

    inp 欄位 (缺的給 None):
        t, guided, agl_m, airspeed_mps, gs_mps, vn, ve, vd,
        yaw_true_deg, pitch_deg, roll_deg,
        tgt_n, tgt_e, tgt_valid, tgt_vn, tgt_ve, slant_m, ground_m,
        ac_n, ac_e,
        bbox_detected, bbox_u, bbox_v, bbox_w, bbox_h,
        bbox_center_offset_norm
    """

    SEARCH = 'SEARCH'
    PRE_DIVE = 'PRE_DIVE'
    DIVE = 'DIVE'
    PULLOUT = 'PULLOUT'
    ABORT = 'ABORT'
    DONE = 'DONE'

    def __init__(self, cfg: GuidanceConfig, verbose=True):
        self.cfg = cfg
        self.verbose = verbose
        self.mode = self.SEARCH

        self.pid = DivePID(cfg)
        self._pid_last_t = None
        self._sp_u, self._sp_v = cfg.CX, cfg.CY

        self.mission_armed = False
        self._valid_streak = 0
        self._search_agl_ref = None
        self._predive_t0 = None
        self._predive_agl0 = None
        self._pushover_started = False
        self._dive_t0 = None
        self._pullout_t0 = None
        self._pullout_pitch_start = None
        self._abort_t0 = None
        self._abort_reason = ''
        self._last_target_valid_t = None
        self._fpv_settled = False
        self._drift_frames = 0

        # DIVE 相機: body_zero → R_bc = R_g2c·Ry(tilt), 已知常數
        tilt = math.radians(cfg.GIMBAL_DIVE_TILT_BODY_DEG)
        self.R_bc = R_G2C @ rot_y_np(tilt)
        self.gimbal_lock = False

        self._pitch_cmd = 0.0
        self._heading_cmd = 0.0
        self._dive_frozen_heading = 0.0   # 版本 B: 進 DIVE 凍結的航向

        self.tecs_resend = False   # PULLOUT 觸發時設起, run_flight 消費
        # ★ PULLOUT 成績快照 (觸發瞬間填, 呼叫端讀)
        self.pullout_snapshot = None

    # ------------------------------------------------------------------
    def arm(self):
        self.mission_armed = True
        self._log("mission ARMED — 條件滿足即進入 PRE_DIVE")

    def operator_abort(self, t=None):
        self._trigger_abort('OPERATOR', t)

    def _log(self, msg):
        if self.verbose:
            print(f"[Mission/{self.mode}] {msg}")

    def _trigger_abort(self, reason, t):
        if self.mode in (self.ABORT, self.DONE):
            return
        self._abort_reason = reason
        self._abort_t0 = t
        self.mode = self.ABORT
        self._log(f"★★ ABORT: {reason} — 拉起 {self.cfg.ABORT_HOLD_S}s 後釋放控制權")

    # ------------------------------------------------------------------
    def update(self, inp):
        cfg = self.cfg
        t = inp['t']
        guided = bool(inp.get('guided', False))
        agl = inp.get('agl_m')
        tgt_valid = bool(inp.get('tgt_valid', False))

        if tgt_valid:
            self._valid_streak += 1
            self._last_target_valid_t = t
        else:
            self._valid_streak = 0

        # 非 GUIDED → 駕駛員奪權, 任務結束
        if not guided and self.mode in (self.PRE_DIVE, self.DIVE, self.PULLOUT):
            self._log("飛控離開 GUIDED — 駕駛員接手, 任務結束")
            self.mode = self.DONE

        if self.mode == self.SEARCH:
            self._update_search(inp, t, agl)
        if self.mode == self.PRE_DIVE:
            self._update_predive(inp, t, agl)
        if self.mode == self.DIVE:
            self._update_dive(inp, t, agl)
        if self.mode == self.PULLOUT:
            self._update_pullout(inp, t)
        if self.mode == self.ABORT:
            self._update_abort(t)

        cmd_active = (guided and (
            self.mode in (self.PRE_DIVE, self.DIVE, self.PULLOUT,
                          self.ABORT, self.DONE)
            or (self.mode == self.SEARCH and self.mission_armed
                and self._search_agl_ref is not None)))

        return {
            'mode': self.mode,
            'cmd_active': cmd_active,
            'is_abort': int(self.mode == self.ABORT),
            'is_pullout': int(self.mode == self.PULLOUT),
            'tecs_level': int(self.mode == self.DONE),
            'pitch_cmd_deg': round(self._pitch_cmd, 2),
            'heading_cmd_deg': round(wrap_deg(self._heading_cmd) % 360.0, 2),
            'gimbal_lock': self.gimbal_lock,
            # ── PID 記錄 (DIVE 段有效, 其他段為上次值/0) ──
            'pid_u_sp': round(self._sp_u, 1),
            'pid_v_sp': round(self._sp_v, 1),
            'pid_u_err_deg': round(self.pid.info['u_err_deg'], 3),
            'pid_v_err_deg': round(self.pid.info['v_err_deg'], 3),
            'pid_lat_p': round(self.pid.lat.p, 3),
            'pid_lat_i': round(self.pid.lat.i, 3),
            'pid_lat_d': round(self.pid.lat.d, 3),
            'pid_lon_p': round(self.pid.lon.p, 3),
            'pid_lon_i': round(self.pid.lon.i, 3),
            'pid_lon_d': round(self.pid.lon.d, 3),
            'pid_dpsi_deg': round(self.pid.info['dpsi'], 3),
            # ★ 2026/09/11: 版本 B 的語意載體.
            #   舊架構下 yaw 欄位 = heading_cmd, 凍結 heading 就等於關掉橫向.
            #   新架構的 DIVE 方向舵直接吃 pid_dpsi_deg, 會繞過 heading_cmd,
            #   所以必須另外把「這趟是不是版本 B」告訴 attitude_bridge,
            #   否則 --dive-hold-heading 形同虛設, A/B 對照失去意義.
            'dive_hold_heading': int(bool(self.cfg.DIVE_HOLD_HEADING)),
            'pid_dtheta_deg': round(self.pid.info['dtheta'], 3),
            'pid_compute_ms': round(self.pid.last_compute_ms, 3),
            'mission_armed': int(self.mission_armed),
            'abort_reason': self._abort_reason,
        }

    # ------------------------------------------------------------------
    def _predive_slant_threshold(self, agl):
        s = math.sin(math.radians(self.cfg.DIVE_ANGLE_DEG))
        return self.cfg.PREDIVE_SLANT_MARGIN * agl / max(s, 0.1)

    def _update_search(self, inp, t, agl):
        cfg = self.cfg
        if not self.mission_armed or not inp.get('guided', False):
            return

        # ── armed 後的 SEARCH: 主動發平飛保持命令 ──
        # ArduPlane GUIDED 沒收到 setpoint 會原地 loiter (壓坡度轉圈),
        # 會把進場航跡弄歪 → armed 起就持續命令「定高 + 朝目標直飛」.
        if self._search_agl_ref is None and agl is not None:
            self._search_agl_ref = agl          # 定高基準 = arm 當下高度
        if self._search_agl_ref is not None and agl is not None:
            alt_err = self._search_agl_ref - agl
            self._pitch_cmd = max(-5.0, min(8.0,
                                            cfg.ALPHA_NOM_DEG + 0.5 * alt_err))
        tgt_n, tgt_e = inp.get('tgt_n'), inp.get('tgt_e')
        if cfg.SEARCH_HOLD_HEADING:
            # 地面測試: 保持當下機頭, 切 GUIDED 不打 yaw
            yaw_now = inp.get('yaw_true_deg')
            if yaw_now is not None:
                self._heading_cmd = yaw_now
        elif tgt_n is not None:
            # 實飛: 朝目標方位角直飛
            ac_n = inp.get('ac_n', 0.0) or 0.0
            ac_e = inp.get('ac_e', 0.0) or 0.0
            self._heading_cmd = math.degrees(
                math.atan2(tgt_e - ac_e, tgt_n - ac_n))

        if agl is None or agl < cfg.ENTRY_MIN_AGL_M:
            return

        # ── 觸發模式 A: 距離門檻 (gps_point=到固定點 / target_dist=到目標) ──
        if cfg.PREDIVE_TRIGGER_MODE in ('gps_point', 'target_dist'):
            d = inp.get('predive_dist_m')   # 來源由 virtual_source 決定
            thr = (cfg.PREDIVE_REACH_RADIUS_M
                   if cfg.PREDIVE_TRIGGER_MODE == 'gps_point'
                   else cfg.PREDIVE_DIST_M)
            if d is not None and d <= thr:
                self.mode = self.PRE_DIVE
                self._predive_t0 = t
                # ★ 2026/09/11: TECS 高度目標在【進 PRE_DIVE 時】就切到拉平
                #   高度 (原本等 PULLOUT 才送, 只剩 0.5s 就交棒).
                #   PRE_DIVE/DIVE 期間 pitch 與油門都是強制的, TECS 輸出
                #   會被丟掉 —— 但 TECS 內部的積分器與高度需求濾波器照常
                #   在跑, 提早給目標可以讓它在 DONE 放手前就收斂完畢.
                #   附帶好處: 俯衝途中若強制命令逾時, 飛控的退路變成
                #   「往 70m」而不是「往巡航高度爬」, 比較合理.
                self.tecs_resend = True
                p = inp.get('pitch_deg')
                self._pitch_cmd = p if p is not None else self._pitch_cmd
                self._predive_agl0 = agl
                tag = ('到達 PRE_DIVE 點' if cfg.PREDIVE_TRIGGER_MODE == 'gps_point'
                       else '到目標水平距')
                self._log(f"→ PRE_DIVE ({tag} {d:.0f}m ≤ {thr:.0f}m, "
                          f"AGL={agl:.0f}m) 開始壓機頭")
            return

        # ── 觸發模式 B: 估距 + 居中 streak (舊, slant) ──
        slant = inp.get('slant_m')
        off = inp.get('bbox_center_offset_norm')
        if (self._valid_streak >= cfg.ENTRY_VALID_STREAK
                and slant is not None
                and off is not None and off <= cfg.ENTRY_CENTER_NORM_MAX
                and slant <= self._predive_slant_threshold(
                    max(agl, cfg.BENCH_GEOMETRY_AGL_M))):
            self.mode = self.PRE_DIVE
            self._predive_t0 = t
            # ★ 2026/09/11: TECS 高度目標在【進 PRE_DIVE 時】就切到拉平
            #   高度 (原本等 PULLOUT 才送, 只剩 0.5s 就交棒).
            #   PRE_DIVE/DIVE 期間 pitch 與油門都是強制的, TECS 輸出
            #   會被丟掉 —— 但 TECS 內部的積分器與高度需求濾波器照常
            #   在跑, 提早給目標可以讓它在 DONE 放手前就收斂完畢.
            #   附帶好處: 俯衝途中若強制命令逾時, 飛控的退路變成
            #   「往 70m」而不是「往巡航高度爬」, 比較合理.
            self.tecs_resend = True
            p = inp.get('pitch_deg')
            self._pitch_cmd = p if p is not None else self._pitch_cmd
            self._predive_agl0 = agl
            self._log(f"→ PRE_DIVE (slant={slant:.0f}m ≤ 門檻"
                      f"{self._predive_slant_threshold(max(agl, cfg.BENCH_GEOMETRY_AGL_M)):.0f}m, "
                      f"AGL={agl:.0f}m) 開始壓機頭")

    def _update_predive(self, inp, t, agl):
        cfg = self.cfg
        if t - self._predive_t0 > cfg.PREDIVE_TIMEOUT_S:
            self._trigger_abort('PREDIVE_TIMEOUT', t)
            return
        if (self._last_target_valid_t is None
                or t - self._last_target_valid_t > cfg.ABORT_TARGET_LOST_S):
            self._trigger_abort('TARGET_LOST_PREDIVE', t)
            return

        # 低 AGL 保護 (原版只有 DIVE 有, 這裡補上)
        if (cfg.ABORT_MIN_AGL_M > 0 and agl is not None
                and agl < cfg.ABORT_MIN_AGL_M):
            self._trigger_abort(f'LOW_AGL_PREDIVE_{agl:.0f}m', t)
            return

        ground = inp.get('ground_m')
        dep_now = None
        if agl is not None and ground is not None and ground > 1.0:
            dep_now = math.degrees(math.atan2(agl, ground))

        theta_target = -(cfg.DIVE_ANGLE_DEG) + cfg.ALPHA_DIVE_DEG

        # ── 距離到 → 直接 -30, 壓到位 → 交棒 ──
        dive_dep = cfg.DIVE_ANGLE_DEG - cfg.DIVE_ENTRY_DEPRESSION_TOL_DEG
        # ★ 2026/09/11: 這裡原本用生 agl, 沒吃 BENCH_GEOMETRY_AGL_M, 導致
        #   掛機時 d_cone≈0 → 推頭門檻只有 20m → 永遠不推頭 → PREDIVE_TIMEOUT.
        #   SEARCH 的 slant 門檻與 PULLOUT 門檻都已用 max(), 這裡補齊一致性.
        #   實飛時 BENCH_GEOMETRY_AGL_M=0, max() 等同原行為, 無影響.
        _agl_geom = (max(agl, cfg.BENCH_GEOMETRY_AGL_M)
                     if agl is not None else None)
        d_cone = ((_agl_geom / max(math.tan(math.radians(dive_dep)), 0.05))
                  if _agl_geom is not None else None)
        if cfg.BENCH_FAKE_CLOSURE_MPS > 0 and ground is not None:
            # 掛機假逼近: 距離以固定速率倒數 (只影響 PUSH 觸發判斷)
            ground = max(0.0, ground - cfg.BENCH_FAKE_CLOSURE_MPS
                         * (t - self._predive_t0))
        if not self._pushover_started:
            if (ground is not None and d_cone is not None
                    and ground <= d_cone + cfg.PREDIVE_PUSH_MARGIN_M):
                self._pushover_started = True
                self._log(f"PRE_DIVE PUSH (ground={ground:.0f}m ≤ "
                          f"錐界{d_cone:.0f}+{cfg.PREDIVE_PUSH_MARGIN_M:.0f}m)"
                          f" 直接命令 {theta_target:.1f}°")
        if not self._pushover_started:
            # 定高逼近
            alt_err = self._predive_agl0 - (agl if agl is not None
                                            else self._predive_agl0)
            self._pitch_cmd = max(-5.0, min(8.0,
                                            cfg.ALPHA_NOM_DEG + 0.5 * alt_err))
        else:
            self._pitch_cmd = theta_target      # 壓多快飛控決定
        # 橫向: 指向瞄準點
        tgt_n, tgt_e = inp.get('tgt_n'), inp.get('tgt_e')
        if tgt_n is not None:
            an, ae = tgt_n, tgt_e
            if cfg.DIVE_TARGET_LEAD and agl is not None:
                an, ae = lead_aim_point(inp, agl)
            self._heading_cmd = math.degrees(math.atan2(
                ae - inp.get('ac_e', 0.0), an - inp.get('ac_n', 0.0)))
        # 交棒: pitch 壓到容差內 (且 bbox 有效)
        p_now = inp.get('pitch_deg')
        # ★ BENCH_HANDOFF_ANY_PITCH: 掛機時飛機不會真的低頭, 若仍要求實際
        #   pitch 壓到容差內就永遠交棒不了 DIVE. 只有 --bench 會打開.
        _pitch_ok = (cfg.BENCH_HANDOFF_ANY_PITCH
                     or (p_now is not None
                         and p_now <= theta_target
                         + cfg.PREDIVE_HANDOFF_PITCH_TOL_DEG))
        if (self._pushover_started and _pitch_ok
                and inp.get('bbox_detected', True)):
            self._enter_dive(inp, t, dep_now if dep_now is not None else 0.0)
        return

    def _enter_dive(self, inp, t, depression):
        cfg = self.cfg
        self.gimbal_lock = True                 # 虛擬相機切 body-fixed
        self._fpv_settled = (cfg.GIMBAL_FPV_SETTLE_S <= 0.0)
        self._drift_frames = 0
        self.mode = self.DIVE
        self._dive_t0 = t
        self._settle_pitch = -(cfg.DIVE_ANGLE_DEG) + cfg.ALPHA_DIVE_DEG
        # PID 從當下命令無縫接手 (積分/濾波歸零, 變化率限幅以此為起點)
        self.pid.reset(pitch0=self._pitch_cmd, heading0=self._heading_cmd)
        self._spu_filt = Filter1st(cfg.PID_SP_FILT_TAU_S)
        self._spv_filt = Filter1st(cfg.PID_SP_FILT_TAU_S)
        self._pid_last_t = t
        self._sp_u, self._sp_v = cfg.CX, cfg.CY
        # ★ 版本 B (DIVE_HOLD_HEADING): 凍結進 DIVE 那刻的航向, DIVE 期間不修 yaw.
        #   優先用當下實際航向 (course, 退化用 yaw); 拿不到就用進入時的 heading_cmd.
        vn, ve = inp.get('vn'), inp.get('ve')
        if vn is not None and (vn * vn + ve * ve) > 4.0:
            self._dive_frozen_heading = math.degrees(math.atan2(ve, vn))
        else:
            self._dive_frozen_heading = (inp.get('yaw_true_deg')
                                         if inp.get('yaw_true_deg') is not None
                                         else self._heading_cmd)
        self._log(f"→ DIVE (視線俯角 {depression:.1f}°) 影像 PID 接管"
                  + (f"; ★ yaw 鎖定 {self._dive_frozen_heading:.1f}°"
                     if cfg.DIVE_HOLD_HEADING else ""))

    def _update_dive(self, inp, t, agl):
        cfg = self.cfg

        # ── ★ PULLOUT 觸發 (正門, 先於一切 ABORT 檢查) ──
        vd_now = inp.get('vd') or 0.0
        # 掛機: 門檻判斷用假 AGL 下限, 讓 DIVE 段能停留觀察 (實飛 BENCH=0 無效果)
        agl_gate = max(agl, cfg.BENCH_GEOMETRY_AGL_M) if agl is not None else None
        agl_pred = (agl_gate - vd_now * cfg.PULLOUT_LEAD_S) \
            if agl_gate is not None else None
        V_now = inp.get('airspeed_mps')
        if V_now is not None and V_now >= cfg.PULLOUT_VMAX_MPS:
            self._enter_pullout(inp, t, agl, reason=f"超速 {V_now:.1f}m/s")
            return
        if agl_gate is not None and min(agl_gate, agl_pred) <= cfg.PULLOUT_AGL_M:
            self._enter_pullout(inp, t, agl, reason=f"AGL {agl:.1f}m")
            return

        # ── ABORT 檢查 (逃生門) ──
        if (cfg.ABORT_MIN_AGL_M > 0 and agl is not None
                and agl < cfg.ABORT_MIN_AGL_M):
            self._trigger_abort(f'LOW_AGL_{agl:.0f}m', t)
            return
        if (self._last_target_valid_t is None
                or t - self._last_target_valid_t > cfg.ABORT_TARGET_LOST_S):
            self._trigger_abort('TARGET_LOST_DIVE', t)
            return

        # ── FPV settle (虛擬預設 0 → 跳過) ──
        if not self._fpv_settled:
            if t - self._dive_t0 < cfg.GIMBAL_FPV_SETTLE_S:
                if cfg.DIVE_SETTLE_HOLD_PITCH:
                    self._pitch_cmd = max(cfg.PITCH_CMD_MIN_DEG, self._settle_pitch)
                tgt_n, tgt_e = inp.get('tgt_n'), inp.get('tgt_e')
                ac_n = inp.get('ac_n', 0.0) or 0.0
                ac_e = inp.get('ac_e', 0.0) or 0.0
                if tgt_n is not None:
                    self._heading_cmd = math.degrees(
                        math.atan2(tgt_e - ac_e, tgt_n - ac_n))
                return
            self._fpv_settled = True

        # ── 目標嚴重偏出畫面 → ABORT (實飛防 FPV 沒 hold; 模擬防發散) ──
        if cfg.RBC_ABORT_IF_TARGET_DRIFT:
            off = inp.get('bbox_center_offset_norm')
            if off is not None and off > cfg.RBC_DRIFT_NORM_MAX:
                self._drift_frames += 1
                if self._drift_frames >= cfg.RBC_DRIFT_FRAMES:
                    self._trigger_abort('TARGET_DRIFT', t)
                    return
            else:
                self._drift_frames = 0

        # ── 影像 PID (每拍) ──
        # 進 DIVE 的當拍跳過: 這一拍的 bbox 是相機切 body-fixed 之前算的
        # (實飛: FPV 切換那一幀亦不可信), 沿用 PRE_DIVE 最後命令.
        if t <= self._dive_t0:
            return
        dt = (t - self._pid_last_t) if self._pid_last_t is not None else 0.0
        self._pid_last_t = t
        dt = max(1e-3, min(dt, 0.5))

        # setpoint: 靜止=畫面中心; 動態=中心+前置偏移 (幾何投影, 見 lead_setpoint_px)
        # ★ 低通 τ=0.6s: t_go=AGL/vd 使 setpoint 含機體狀態, 不濾會與縱向
        #   震盪耦合 (vd 震盪→sp 震盪→回餵 PID), 濾掉快速項、保留幾何漂移.
        sp_u_raw, sp_v_raw = lead_setpoint_px(inp, agl, self.R_bc, cfg)
        # ★ 攻角補償: 縱向 setpoint 下移 (實飛 FPV tilt≈0 時生效; 模擬設 0)
        sp_v_raw += cfg.DIVE_AIM_V_OFFSET_PX
        self._sp_u = self._spu_filt.step(sp_u_raw, dt)
        self._sp_v = self._spv_filt.step(sp_v_raw, dt)

        # 當下航向 (NED course, 退化用 yaw)
        vn, ve = inp.get('vn'), inp.get('ve')
        if vn is not None and (vn * vn + ve * ve) > 4.0:
            heading_now = math.degrees(math.atan2(ve, vn))
        else:
            heading_now = inp.get('yaw_true_deg', 0.0) or 0.0

        pitch_now = inp.get('pitch_deg')
        if pitch_now is None:
            pitch_now = self._pitch_cmd
        # ★ 掛機基準覆寫 (見 BENCH_DIVE_PITCH_BASE_DEG). 實飛為 None, 無影響.
        if cfg.BENCH_DIVE_PITCH_BASE_DEG is not None:
            pitch_now = float(cfg.BENCH_DIVE_PITCH_BASE_DEG)
        detected = bool(inp.get('bbox_detected', False))
        pitch, hdg = self.pid.update(
            inp.get('bbox_u'), inp.get('bbox_v'),
            self._sp_u, self._sp_v, detected, pitch_now, heading_now, dt)
        self._pitch_cmd = pitch
        # ★ 版本切換: 預設用 PID 修 yaw (版本 A); DIVE_HOLD_HEADING=True 則
        #   凍結進 DIVE 那刻的航向, 橫向不修 (版本 B). 縱向 pitch 兩版都照修.
        if cfg.DIVE_HOLD_HEADING:
            self._heading_cmd = self._dive_frozen_heading
        else:
            self._heading_cmd = hdg

    # ------------------------------------------------------------------
    def _enter_pullout(self, inp, t, agl, reason=''):
        """★ 拉平觸發 (AGL/超速): 凍結成績 → 交棒過渡 → TECS."""
        cfg = self.cfg
        self.pullout_snapshot = {
            't': t, 'agl': agl,
            'ac_n': inp.get('ac_n'), 'ac_e': inp.get('ac_e'),
            'vn': inp.get('vn'), 've': inp.get('ve'), 'vd': inp.get('vd'),
            'airspeed_mps': inp.get('airspeed_mps'),
            'tgt_n': inp.get('tgt_n'), 'tgt_e': inp.get('tgt_e'),
            'tgt_vn': inp.get('tgt_vn', 0.0), 'tgt_ve': inp.get('tgt_ve', 0.0),
        }
        self._pullout_t0 = t
        self.gimbal_lock = False
        self.mode = self.PULLOUT
        self.tecs_resend = True     # run_flight 讀到即重送 TECS 目標 (保險)
        self._log(f"→ PULLOUT ({reason}) 成績已凍結; 強制 pitch="
                  f"{cfg.PULLOUT_HANDOFF_PITCH_DEG:.0f}° "
                  f"{cfg.PULLOUT_HANDOFF_S}s 後交 TECS "
                  f"({cfg.TECS_HOLD_ALT_M:.0f}m/{cfg.TECS_SPEED_MPS:.0f}m/s)")

    def _update_pullout(self, inp, t):
        """交棒過渡: 強制 pitch=HANDOFF_PITCH 送 HANDOFF_S 秒 (飛控 hold 的
        「最後一筆」是安全值), 然後轉 DONE = TECS 平飛."""
        cfg = self.cfg
        self._pitch_cmd = cfg.PULLOUT_HANDOFF_PITCH_DEG
        # heading 保持 (self._heading_cmd 不更新 = 凍結)
        if t - self._pullout_t0 >= cfg.PULLOUT_HANDOFF_S:
            self.mode = self.DONE
            self._log(f"放手: pitch/油門歸 TECS "
                      f"({cfg.TECS_HOLD_ALT_M:.0f}m/{cfg.TECS_SPEED_MPS:.0f}m/s), "
                      f"只留 roll=0; 撥離 GUIDED 結束任務")

    def _update_abort(self, t):
        cfg = self.cfg
        self._pitch_cmd = cfg.ABORT_PITCH_DEG
        self.gimbal_lock = False
        if t - self._abort_t0 > cfg.ABORT_HOLD_S:
            self.mode = self.DONE
            self._log("ABORT 保持結束, 釋放姿態控制權")
