"""
Ego-Aware DKF + NED Reprojection Fallback - V3

相對 V2 (SimplifiedDKF) 的改動:
  (a) predict() 新增 projective ego compensation:
      把當前 pixel 反投影到 NED 假設目標位置, 用「新姿態」正投影回 pixel,
      差值 = 飛機/雲台運動造成的 pixel 漂移, 從預測中扣掉,
      讓 [u_dot, v_dot] 只承擔「目標的殘差像素速度」(地面車輛通常很小).
      → Q 可大幅下調 (150 → 20 px/s²), 預測精度顯著提升.

  (c) 新增 reproject_target_ned_to_pixel() 工具函數,
      把 controller.target_kf 的 NED 位置反投影回 pixel,
      在 YOLO 丟失或畫面嚴重延遲時當 fallback 量測.

API:
  - predict(dt=None, ac_state=None, gimbal_state=None, depth_m=None)
    所有參數可選, 全部 None 時退化為 V2 的純 CV model 行為.
  - update_fallback(z_pixel, r_std_pixel) 給反投影量測用,
    繞過 dynamic R 與動態 bbox 縮放, R 由呼叫端決定.
  - 既有的 update(z, delay_steps, bbox_size) 保留, 介面不變.

設計考量:
  - 與 V2 同檔同架構, 可直接 drop-in 替換 SimplifiedDKF.
  - reproject_target_ned_to_pixel 寫成獨立函數, 不依賴任何 class,
    方便在 main loop 直接呼叫 (參考 RayProjectionDistanceEstimator 的旋轉鏈).
"""

import math
import numpy as np
from collections import deque


# ════════════════════════════════════════════════════════════════════
#  旋轉矩陣工具 (與 gimbal_controller_pinhole.py 同款)
# ════════════════════════════════════════════════════════════════════

def _Rx(phi_deg):
    """繞 X 軸旋轉 (roll)"""
    phi = math.radians(phi_deg)
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _Ry(theta_deg):
    """繞 Y 軸旋轉 (pitch)"""
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

def _Rz(psi_deg):
    """繞 Z 軸旋轉 (yaw)"""
    psi = math.radians(psi_deg)
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


# 預設 Gimbal → Camera (CV convention: X 右, Y 下, Z 前) 旋轉矩陣
# 與 RayProjectionDistanceEstimator 一致
DEFAULT_R_G2C = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
], dtype=float)


# ════════════════════════════════════════════════════════════════════
#  (c) NED → Pixel 反投影工具
# ════════════════════════════════════════════════════════════════════

def reproject_target_ned_to_pixel(
    target_x_ned, target_y_ned,
    ac_x_ned, ac_y_ned, ac_altitude_agl,
    ac_roll_deg, ac_pitch_deg, ac_yaw_true_deg,
    gimbal_pitch_deg, gimbal_yaw_deg,
    fx, fy, cx, cy,
    R_g2c=None,
    gimbal_pitch_offset_deg=0.0,
    gimbal_yaw_offset_deg=0.0,   # SIYI pan 對齊真北的啟動偏移(度); 須與 estimate_distance 同值
    target_altitude_msl=0.0,
):
    """
    把 NED 座標下的目標位置反投影成相機 pixel 座標.

    這是 RayProjectionDistanceEstimator.estimate_distance() 的反向運算:
        forward:  pixel → ray_cam → ray_ned → ground intersect → slant_range
        inverse:  target_ned → vec_ned → vec_cam → pixel

    Args:
        target_x_ned, target_y_ned: 目標在 NED 的位置 (m), 通常從 controller.target_kf.x[0:2] 取
        ac_x_ned, ac_y_ned, ac_altitude_agl: 飛機 NED 位置 + AGL 高度 (m)
        ac_roll/pitch_deg, ac_yaw_true_deg: 飛機姿態 (deg, 真航向 0=北 90=東)
        gimbal_pitch/yaw_deg: 雲台角度 (deg, gimbal_pitch 負=向下看)
        fx, fy, cx, cy: 相機內參 (pixel)
        R_g2c: Gimbal → Camera CV 旋轉, None 用預設
        gimbal_pitch_offset_deg: 同 RayProjectionDistanceEstimator 的常數補償
        target_altitude_msl: 若目標不在地面 (e.g. 山坡上), 給 MSL 高度 (m)

    Returns:
        (u, v, valid):
            u, v: 預測 pixel 座標 (float, 不限定在影像內)
            valid: True 表示目標在相機前方 (Pc_z > 0); False 表示在後方/側面無法成像
    """
    if R_g2c is None:
        R_g2c = DEFAULT_R_G2C

    # 飛機在 NED 的 z 座標 (NED 中 down 為正): -altitude_agl - target_altitude_msl
    # 若目標不在地面, target 的 NED z = -target_altitude_msl
    # vec_ned = target_ned - aircraft_ned
    dx = target_x_ned - ac_x_ned          # north
    dy = target_y_ned - ac_y_ned          # east
    dz = ac_altitude_agl - target_altitude_msl  # down: target 比飛機低多少
    vec_ned = np.array([dx, dy, dz], dtype=float)

    # SIYI A8 三軸 world-stabilized: 相機世界指向只由雲台世界角決定,
    # 與飛機 roll/pitch 無關(雲台自行穩定補償) → 不乘 R_w2b。
    # 注意: vec_ned 的「平移(視差)」仍用飛機 NED 位置, 只有「朝向」拿掉飛機姿態。
    # 此慣例必須與 RayProjectionDistanceEstimator.estimate_distance() 一致, 正反向才閉合。
    gimbal_pitch_eff = gimbal_pitch_deg - gimbal_pitch_offset_deg
    gimbal_yaw_world = gimbal_yaw_deg + gimbal_yaw_offset_deg
    R_ned2gimbal = _Ry(gimbal_pitch_eff) @ _Rz(gimbal_yaw_world)
    R_ned2cam = R_g2c @ R_ned2gimbal

    # 目標在 camera 座標
    Pc = R_ned2cam @ vec_ned

    if Pc[2] <= 0.1:
        # 目標在相機後方 (或太近), 無法成像
        return None, None, False

    # Pinhole projection (CV convention)
    u = fx * Pc[0] / Pc[2] + cx
    v = fy * Pc[1] / Pc[2] + cy

    return float(u), float(v), True


# ════════════════════════════════════════════════════════════════════
#  (a) Ego-Aware DKF
# ════════════════════════════════════════════════════════════════════

class EgoAwareDKF:
    """
    Ego-aware 延遲卡爾曼濾波器 (V3).

    State: [u, v, u_dot, v_dot]  (pixel 座標 + 像素速度)
    重點: u_dot, v_dot 現在只代表「目標自身運動造成的殘差像素速度」,
          飛機/雲台運動造成的 pixel 漂移在 predict() 用 IBVS Jacobian 補償掉.

    新功能:
      1. predict(omega_body_radps, v_world_mps, R_w2b, R_b2g, depth_m)
         在 CV propagation 之外, 加上 ego motion 預測位移.
      2. update_fallback(z, r_std_pixel)
         給 NED 反投影量測用, R 由外部決定.

    保留:
      - update(z, delay_steps, bbox_size)
      - dynamic R / Mahalanobis gate / coasting mode (部分弱化, 因為 ego comp 已治本)
    """

    def __init__(self,
                 dt=0.01,
                 image_delay=0.08,
                 fx=1500.0, fy=1500.0, cx=960.0, cy=505.0,
                 enable_ego_comp=True,
                 max_history_steps=200,
                 max_update_delay_steps=30):
        """
        Args:
            ...
            max_history_steps: ✨ V3.3 history buffer 大小 (步數).
                必須 ≥ 預期最大延遲步數. 預設 200 步 = 2 秒, 對 100Hz dt=0.01 足夠.
            max_update_delay_steps: ✨ V3.3 update 補償深度上限 (步數).
                超過此值的延遲, update 只回去到此上限的位置做 KF update.
                預設 30 步 = 300ms. 太大 → forward propagation 放大 velocity correction 誤差.
                預設值是經驗值, dive scenario 大延遲應靠 NED fallback 補, 不靠 update 補.
        """
        self.dt = dt
        self.dt_nominal = dt      # ★ 初始 dt, delay replay 用 (固定步長)
        self.image_delay = image_delay
        self.default_delay_steps = max(1, int(image_delay / dt))
        self.max_update_delay_steps = max_update_delay_steps

        # 相機內參 (用於 IBVS Jacobian)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.enable_ego_comp = enable_ego_comp

        # 狀態
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0

        # CV propagation matrix (作用在 [u_dot, v_dot] 殘差速度上)
        self.F_base = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])

        # ✨ Q 大幅下調: 殘差速度 (target ground motion) 的不確定性
        #    地面車 ~10 m/s, 中遠距 ~10 px/s pixel 速度,
        #    accel 變化 ~5 px/s² 量級 → accel_noise=20 比 V2 的 150 合理得多
        accel_noise = 20.0  # px/s²
        self._accel_noise = accel_noise
        dt2 = dt * dt
        self.Q = (accel_noise ** 2) * np.array([
            [dt2*dt2/4, 0,         dt2*dt/2, 0        ],
            [0,         dt2*dt2/4, 0,        dt2*dt/2 ],
            [dt2*dt/2,  0,         dt2,      0        ],
            [0,         dt2*dt/2,  0,        dt2      ],
        ])

        # 量測噪聲 (動態 R, 同 V2)
        # ✨ V3.2: R_min_std 0.5 → 5.0 — 防止 DIVE 末段 bbox 大時 R 過小, gate 過嚴拒所有 YOLO
        self.R_base_std = 8.0
        self.R_bbox_ref = 32.0
        self.R_min_std = 5.0
        self.R_max_std = 25.0

        # ✨ Ego 補償的不確定性: 角速度量測誤差 + 深度估測誤差 → pixel 預測 noise
        # 經驗值: 角速度 1 deg/s 誤差 → fx * sin(1°) ≈ 26 px/s 預測誤差
        self.ego_uncertainty_std = 5.0  # px/s, 額外加到 Q 的速度項

        # ✨ V3.2: gate 50 → 200, 跟 fallback 對齊. 50 太緊, DIVE 末段 LoS 變化快會誤拒
        self.gate_threshold = 200.0
        self.gate_threshold_fallback = 200.0  # ✨ NED 反投影誤差大, gate 放寬

        max_history = max(self.default_delay_steps + 30, max_history_steps)
        self._history = deque(maxlen=max_history)
        self._max_history_size = max_history

        self.initialized = False
        self._predict_count = 0
        self._update_count = 0
        self._update_fallback_count = 0
        self._reject_count = 0

        # Coasting mode (削弱版: ego comp 已大幅減少 ego-induced drift, coast 不需要那麼激進)
        self._consecutive_miss = 0
        # ✨ 抗漂移調整 (依 log 實測: coast 期間殘差速度被外插造成框飄):
        #   coast 提早啟動 + 衰減加快, 讓「沒偵測時」殘差速度快速歸零,
        #   停止的目標其框不再滑走。只影響無偵測(coast)行為, 不動有偵測時的追蹤。
        self.coast_start = 5             # was 25; 多數 coast<25幀, 原值幾乎不啟動
        self.coast_q_inflation = 1.5
        self.coast_q_rate = 0.03
        self.coast_vel_decay = 0.88      # was 0.999; 殘差速度約 5 幀內衰減掉
        self.coast_vel_decay_fast = 0.70 # was 0.99; 長 coast 更快歸零
        # 殘差速度上限 (px/s): 防雜訊/reset 後虛假高速 (log 曾見 565) 把框甩飛
        self.max_residual_vel_pps = 120.0

        # 上次 ego predict 的內部狀態 (給 P 推進用)
        self._last_ego_dpx = np.zeros(2)
        self._last_ego_uncertainty_var = 0.0

        # ★ 暴露給上層 CSV 的 per-update 量測殘差 (每次 update() 後刷新)
        #   update() 時計算 innovation y_innov 和 Mahalanobis² 都只是內部變數,
        #   留下來才能畫 innovation vs time / DKF uncertainty 等分析圖.
        self._last_innovation = np.zeros(2)       # [u_innov, v_innov] (px)
        self._last_innovation_norm = 0.0          # ||y_innov|| (px)
        self._last_mahalanobis2 = 0.0             # mahal²
        self._last_update_was_accepted = False    # update() 是否通過 gate

        # ✨ 上一步飛機 + 雲台狀態 (給 projective predict 算 ego shift)
        self._prev_ac_state = None
        self._prev_gimbal_state = None

        # ✨ V3.6: dt 相關的 ego comp 閾值 (原本寫死, 針對 100Hz 調).
        # 改成可調屬性, 預設值 = 原本寫死的值 → 不影響任何既有呼叫端.
        # 低頻 (e.g. 相機 30Hz) 呼叫端應依 dt 放大這兩個值, 否則:
        #   - ego_gimbal_jump_deg 太小 → 雲台一追就被誤判跳變, 整段不做 ego comp
        #   - ego_dpx_max_px 太小 → 單步合理 ego 位移被 clip, ego comp 被低估
        # 建議縮放: 以 100Hz(dt=0.01) 為基準, 乘上 (dt / 0.01).
        self.ego_gimbal_jump_deg = 1.5     # 單步雲台角度跳變上限 (deg)
        self.ego_dpx_max_px = 30.0         # 單步 ego pixel 位移上限 (px)

        print(f"[DKF V3] 初始化:")
        print(f"  predict 頻率: {1/dt:.0f} Hz")
        print(f"  預設延遲: {image_delay*1000:.0f} ms ({self.default_delay_steps} 步)")
        print(f"  History buffer: {max_history} 步 = {max_history*dt*1000:.0f} ms (最大可補償延遲)")
        print(f"  Update 補償深度上限: {max_update_delay_steps} 步 = {max_update_delay_steps*dt*1000:.0f} ms "
              f"(超過此值的延遲靠 NED fallback)")
        print(f"  Ego 補償: {'ON' if enable_ego_comp else 'OFF'}")
        print(f"  相機內參: fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f}")
        print(f"  過程噪聲 (殘差): {accel_noise:.0f} px/s²  "
              f"(V2 對應值: 150)")
        print(f"  Coasting: start={self.coast_start}步 "
              f"Q_inflate={self.coast_q_inflation}x")

    # ─────────────────────────────────────────────────────────────
    #  Predict (三個版本: pure CV / IBVS Jacobian / Projective)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_F(dt):
        return np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def _build_Q(self, dt):
        """用實際 dt 重建過程噪聲. accel_noise 從 __init__ 存的值."""
        dt2 = dt * dt
        return (self._accel_noise ** 2) * np.array([
            [dt2*dt2/4, 0,         dt2*dt/2, 0        ],
            [0,         dt2*dt2/4, 0,        dt2*dt/2 ],
            [dt2*dt/2,  0,         dt2,      0        ],
            [0,         dt2*dt/2,  0,        dt2      ],
        ])

    def predict(self, dt=None,
                ac_state=None,         # dict: x_ned, y_ned, alt_agl, roll_deg, pitch_deg, yaw_true_deg
                gimbal_state=None,     # dict: pitch_deg, yaw_deg
                depth_m=None):
        """
        高頻預測 — 推薦用法 (projective ego comp).

        把當前 pixel 反投影到 NED 假設目標位置, 用「新姿態」正投影回 pixel,
        差值就是 ego motion 造成的 pixel shift. 自動包含飛機 + 雲台所有運動.

        三種呼叫方式:
          1. predict()
             退化為純 CV model (V2 行為), 不做 ego comp
          2. predict(ac_state, gimbal_state)
             不需要深度: 用上次的深度, 沒有就退化成純 CV
          3. predict(ac_state, gimbal_state, depth_m=slant_range)  ← 推薦
             完整 ego comp, 旋轉+平移都包

        ac_state dict 必須含:
            x_ned, y_ned (m), alt_agl (m),
            roll_deg, pitch_deg, yaw_true_deg
        gimbal_state dict 必須含:
            pitch_deg, yaw_deg
        depth_m: 當前估測的目標斜距 (m), 從 RayProjection 的 slant_range
        """
        if not self.initialized:
            return

        self._consecutive_miss += 1

        # ── 動態 dt ──
        if dt is not None:
            dt = float(np.clip(dt, 0.02, 0.15))
            self.dt = dt
        _F = self._build_F(self.dt)
        _Q = self._build_Q(self.dt)

        # ── Coasting mode ──
        if self._consecutive_miss > self.coast_start:
            coast_steps = self._consecutive_miss - self.coast_start
            q_factor = min(1.0 + coast_steps * self.coast_q_rate, self.coast_q_inflation)
            Q_eff = _Q * q_factor
            decay = self.coast_vel_decay if coast_steps < 30 else self.coast_vel_decay_fast
            self.x[2] *= decay
            self.x[3] *= decay
        else:
            Q_eff = _Q

        # ── (a) Projective ego compensation ──
        ego_dpx = np.zeros(2)
        ego_var = 0.0

        if (self.enable_ego_comp
                and ac_state is not None
                and gimbal_state is not None
                and self._prev_ac_state is not None
                and self._prev_gimbal_state is not None
                and depth_m is not None and depth_m > 1.0):
            # ✨ V3.1 防護: 偵測雲台角度跳變 (e.g. DIVE 觸發時 gimbal lock)
            # 100Hz 下正常雲台速率 < 60°/s → 單步 < 0.6°
            # 超過閾值就視為 lock/unlock 跳變, 跳過此拍 ego comp
            d_gimbal_pitch = abs(gimbal_state['pitch_deg'] - self._prev_gimbal_state['pitch_deg'])
            d_gimbal_yaw   = abs(gimbal_state['yaw_deg']   - self._prev_gimbal_state['yaw_deg'])
            _jump_thr = self.ego_gimbal_jump_deg
            gimbal_jumped = (d_gimbal_pitch > _jump_thr) or (d_gimbal_yaw > _jump_thr)

            if not gimbal_jumped:
                # Step 1: 反投影 (pixel + depth → NED), 用「上一步的姿態」
                P_ned = self._backproject_pixel_to_ned(
                    u_pixel=self.x[0], v_pixel=self.x[1],
                    depth=depth_m,
                    ac_state=self._prev_ac_state,
                    gimbal_state=self._prev_gimbal_state,
                )
                # Step 2: 用「當前姿態」把這個 NED 點正投影回 pixel
                u_new, v_new, valid = reproject_target_ned_to_pixel(
                    target_x_ned=P_ned[0],
                    target_y_ned=P_ned[1],
                    ac_x_ned=ac_state['x_ned'],
                    ac_y_ned=ac_state['y_ned'],
                    ac_altitude_agl=ac_state['alt_agl'],
                    ac_roll_deg=ac_state['roll_deg'],
                    ac_pitch_deg=ac_state['pitch_deg'],
                    ac_yaw_true_deg=ac_state['yaw_true_deg'],
                    gimbal_pitch_deg=gimbal_state['pitch_deg'],
                    gimbal_yaw_deg=gimbal_state['yaw_deg'],
                    fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                    R_g2c=DEFAULT_R_G2C,
                    target_altitude_msl=P_ned[2],   # 保留高度資訊
                )
                if valid:
                    raw_dpx_x = u_new - self.x[0]
                    raw_dpx_y = v_new - self.x[1]
                    raw_dpx_norm = math.sqrt(raw_dpx_x**2 + raw_dpx_y**2)

                    # ✨ V3.1 防護: 單步 ego shift 異常大 → 視為 depth 估測或姿態量測異常
                    # 100Hz 下正常 ego shift < 20 px (超快機動 30 px). 超過上限就 clip.
                    # V3.6: 上限改為可調 (低頻呼叫端會放大此值, 見 __init__ 註解)
                    EGO_DPX_MAX_PIXEL = self.ego_dpx_max_px
                    if raw_dpx_norm > EGO_DPX_MAX_PIXEL:
                        # 等比例 clip 到上限, 避免完全丟掉訊號
                        scale = EGO_DPX_MAX_PIXEL / raw_dpx_norm
                        ego_dpx[0] = raw_dpx_x * scale
                        ego_dpx[1] = raw_dpx_y * scale
                        # 不確定性放大 (因為我們不信任這個 ego comp)
                        ego_var = (raw_dpx_norm * 0.5) ** 2 + 100.0
                    else:
                        ego_dpx[0] = raw_dpx_x
                        ego_dpx[1] = raw_dpx_y
                        # 不確定性: ego_dpx 大時加大 (補償誤差跟移動量成比例)
                        ego_var = (0.05 * raw_dpx_norm) ** 2 + 1.0
            else:
                # Gimbal 跳變 → 跳過這拍 ego comp, 並把跳變記下
                # 下一拍會用新 prev_state, 自動恢復正常
                if not hasattr(self, '_jump_count'):
                    self._jump_count = 0
                self._jump_count += 1
                if self._jump_count % 10 == 1:
                    print(f"[DKF V3.1] ⚠ Gimbal 跳變偵測 "
                          f"(d_pitch={d_gimbal_pitch:.2f}° d_yaw={d_gimbal_yaw:.2f}°), "
                          f"跳過此拍 ego comp [count={self._jump_count}]")

        # ── 套用 ──
        # 注意: 套用順序很重要
        # x[0:2] (pixel pos) 先做 CV (用 x[2:4] 的殘差速度), 再加 ego 位移
        self.x = _F @ self.x
        self.x[0] += ego_dpx[0]
        self.x[1] += ego_dpx[1]
        # ✨ 殘差速度上限: 防止虛假高速把框甩飛 (見 __init__ max_residual_vel_pps)
        _vmag = math.hypot(self.x[2], self.x[3])
        if _vmag > self.max_residual_vel_pps:
            _sc = self.max_residual_vel_pps / _vmag
            self.x[2] *= _sc
            self.x[3] *= _sc

        # P propagation (用動態 F/Q)
        self.P = _F @ self.P @ _F.T + Q_eff
        self.P[0, 0] += ego_var
        self.P[1, 1] += ego_var

        # 記錄
        self._last_ego_dpx = ego_dpx
        self._last_ego_uncertainty_var = ego_var
        if ac_state is not None:
            self._prev_ac_state = dict(ac_state)
        if gimbal_state is not None:
            self._prev_gimbal_state = dict(gimbal_state)

        self._predict_count += 1
        # ✨ V3.3: history 多存 ego_dpx, update 往前推時可重放
        self._history.append((self.x.copy(), self.P.copy(), ego_dpx.copy()))

    def _backproject_pixel_to_ned(self, u_pixel, v_pixel, depth,
                                   ac_state, gimbal_state):
        """
        把 (pixel + depth) 反投影到 NED 座標.

        步驟:
          Pc = depth * normalize([(u-cx)/fx, (v-cy)/fy, 1])
          P_world = ac_pos_ned + R_cam2ned @ Pc
        """
        # 像素 → 相機座標方向向量
        x1 = (u_pixel - self.cx) / self.fx
        x2 = (v_pixel - self.cy) / self.fy
        Pc_dir = np.array([x1, x2, 1.0])
        Pc = depth * Pc_dir / np.linalg.norm(Pc_dir)

        # SIYI A8 world-stabilized: 相機朝向只由雲台世界角決定, 不乘 R_w2b。
        # 必須與 reproject_target_ned_to_pixel 同慣例, FULL_PROJECTIVE 正反向才閉合。
        R_ned2gimbal = (_Ry(gimbal_state['pitch_deg'])
                        @ _Rz(gimbal_state['yaw_deg']))
        R_ned2cam = DEFAULT_R_G2C @ R_ned2gimbal
        R_cam2ned = R_ned2cam.T   # 旋轉矩陣的逆 = 轉置

        # 在相機座標的點 → NED 座標 (相對於飛機)
        rel_ned = R_cam2ned @ Pc

        # 加上飛機 NED 位置 → 絕對 NED
        # 飛機在 NED 的 z = -alt_agl (NED z 朝下為正, 飛機在地面上)
        ac_ned = np.array([ac_state['x_ned'],
                            ac_state['y_ned'],
                            -ac_state['alt_agl']])
        return ac_ned + rel_ned

    #  Update (YOLO 量測, 含延遲補償)
    # ─────────────────────────────────────────────────────────────

    def update(self, z_image, delay_steps=None, bbox_size=None):
        """
        標準 YOLO 量測更新 (與 V2 同介面, 但延遲前推時也用 ego comp).

        注意: 因為歷史的 ego motion 已經在 _history 裡的 x 反映了,
              這裡的延遲前推用 F_base 即可 (歷史 x 已含 ego 修正).
        """
        z = np.asarray(z_image, dtype=float)

        if not self.initialized:
            self.x[0:2] = z
            self.x[2:4] = 0.0
            self.P = np.diag([20.0**2, 20.0**2, 50.0**2, 50.0**2])
            self.initialized = True
            self._history.append((self.x.copy(), self.P.copy(), np.zeros(2)))
            return

        # 動態 R
        if bbox_size is not None and bbox_size > 5:
            r_std = self.R_base_std * (self.R_bbox_ref / bbox_size)
            r_std = np.clip(r_std, self.R_min_std, self.R_max_std)
        else:
            r_std = self.R_base_std
        R = np.diag([r_std**2, r_std**2])

        if delay_steps is None:
            delay_steps = self.default_delay_steps
        delay_steps = min(delay_steps, len(self._history) - 1)
        delay_steps = max(delay_steps, 0)
        # ✨ V3.3: 明確限制 update 補償深度 (原本靠 buffer 小被動限制)
        # 超過 max_update_delay_steps 的延遲, 實際補償只到上限
        # 因為長距離 forward propagation 會放大 KF velocity correction 的誤差
        # 真正大延遲應該讓 NED fallback 接手
        delay_steps = min(delay_steps, self.max_update_delay_steps)

        # 取延遲時刻狀態
        if delay_steps > 0 and len(self._history) > delay_steps:
            x_d, P_d, _ = self._history[-(delay_steps + 1)]
            x_d, P_d = x_d.copy(), P_d.copy()
        else:
            x_d, P_d = self.x.copy(), self.P.copy()
            delay_steps = 0

        # Innovation + gate
        y_innov = z - self.H @ x_d
        S = self.H @ P_d @ self.H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        mahal2 = float(y_innov.T @ S_inv @ y_innov)
        # ★ 寫進實例變數讓上層讀 (給 CSV / 分析圖用)
        self._last_innovation = y_innov.copy()
        self._last_innovation_norm = float(np.linalg.norm(y_innov))
        self._last_mahalanobis2 = mahal2
        self._last_update_was_accepted = (mahal2 <= self.gate_threshold)

        if mahal2 > self.gate_threshold:
            self._reject_count += 1
            if self._reject_count % 10 == 1:
                print(f"[DKF V3] ⚠️ YOLO 量測被拒 (mahal²={mahal2:.1f}) "
                      f"z={z.astype(int)} vs pred={(self.H @ x_d).astype(int)}")
            return

        # KF update at t_{k-D}
        K = P_d @ self.H.T @ S_inv
        x_corrected = x_d + K @ y_innov
        P_corrected = (np.eye(4) - K @ self.H) @ P_d

        # ✨ V3.3: 前推 D 步 — 只用 F_base (純 CV)
        # NOT replay ego_dpx, 因為歷史 ego_dpx 是用舊 x 算的, KF update 後會 mismatch
        # 嚴重延遲時 (>300ms) 這個前推會丟掉部分 ego motion, 應靠 NED fallback 補
        for _ in range(delay_steps):
            x_corrected = self.F_base @ x_corrected
            P_corrected = self.F_base @ P_corrected @ self.F_base.T + self.Q

        # 覆寫
        self.x = x_corrected
        self.P = P_corrected
        self._update_count += 1
        self._consecutive_miss = 0
        if self._history:
            # 保留歷史 ego_dpx (不變動), 只更新 x, P
            _, _, last_ego = self._history[-1]
            self._history[-1] = (self.x.copy(), self.P.copy(), last_ego)

    # ─────────────────────────────────────────────────────────────
    #  (c) Update with NED reprojection fallback
    # ─────────────────────────────────────────────────────────────

    def update_fallback(self, z_pixel, r_std_pixel, gate_override=None):
        """
        NED 反投影 fallback 量測更新.

        與 update() 的差別:
          - R 由外部決定 (依 target_kf 位置 std + slant_range 推算)
          - 沒有 delay 補償 (反投影是「現在」的預測, 不是延遲量測)
          - gate 放寬 (預設用 gate_threshold_fallback)
          - 不重置 _consecutive_miss (因為這不是真實視覺量測)

        Args:
            z_pixel: array(2,), 反投影的 pixel 座標
            r_std_pixel: float, 量測不確定性 (pixel std)
            gate_override: 自訂 mahalanobis gate threshold, None 用預設
        """
        z = np.asarray(z_pixel, dtype=float)

        if not self.initialized:
            # 用 fallback 初始化 (緊急情況, 比沒有好)
            self.x[0:2] = z
            self.x[2:4] = 0.0
            self.P = np.diag([r_std_pixel**2, r_std_pixel**2,
                              100.0**2, 100.0**2])
            self.initialized = True
            self._history.append((self.x.copy(), self.P.copy(), np.zeros(2)))
            return

        R = np.diag([r_std_pixel**2, r_std_pixel**2])
        gate = gate_override if gate_override is not None else self.gate_threshold_fallback

        y_innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        mahal2 = float(y_innov.T @ S_inv @ y_innov)
        # ★ 寫進實例變數 (同 update())
        self._last_innovation = y_innov.copy()
        self._last_innovation_norm = float(np.linalg.norm(y_innov))
        self._last_mahalanobis2 = mahal2
        self._last_update_was_accepted = (mahal2 <= gate)
        if mahal2 > gate:
            self._reject_count += 1
            if self._reject_count % 20 == 1:
                print(f"[DKF V3] ⚠️ NED fallback 量測被拒 (mahal²={mahal2:.1f}, gate={gate:.0f})")
            return

        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self._update_fallback_count += 1
        # 注意: 不重置 _consecutive_miss, 因為 fallback 不算「真的看到了目標」
        if self._history:
            _, _, last_ego = self._history[-1]
            self._history[-1] = (self.x.copy(), self.P.copy(), last_ego)

    # ─────────────────────────────────────────────────────────────
    #  Getters
    # ─────────────────────────────────────────────────────────────

    def get_position(self):
        return self.x[0:2].copy()

    def get_velocity(self):
        """像素殘差速度 (px/s, 不含 ego motion)"""
        return self.x[2:4].copy()

    def get_position_std(self):
        return np.sqrt(np.diag(self.P)[:2])

    def get_stats(self):
        return {
            'predict_count': self._predict_count,
            'update_count': self._update_count,
            'update_fallback_count': self._update_fallback_count,
            'reject_count': self._reject_count,
            'consecutive_miss': self._consecutive_miss,
            'last_ego_dpx': self._last_ego_dpx.copy(),
        }

    def reset(self, initial_position=None):
        if initial_position is not None:
            self.x[0:2] = initial_position
            self.initialized = True
        else:
            self.x = np.zeros(4)
            self.initialized = False
        self.x[2:4] = 0.0
        self.P = np.eye(4) * 100.0
        self._history.clear()
        self._predict_count = 0
        self._update_count = 0
        self._update_fallback_count = 0
        self._reject_count = 0
        self._consecutive_miss = 0
        self._prev_ac_state = None
        self._prev_gimbal_state = None


# ════════════════════════════════════════════════════════════════════
#  自我測試 (跑一個合成情境驗證 ego comp 有效)
# ════════════════════════════════════════════════════════════════════

def _self_test():
    """
    Self-Test #1: 飛機低頭 + 雲台反向追 → 驗證 projective predict 的 ego comp.
    Self-Test #2: NED 反投影 sanity check.
    """
    fx, fy = 1500.0, 1500.0
    cx, cy = 960.0, 505.0
    dt = 0.01
    N = int(1.0 / dt)

    print("\n" + "=" * 70)
    print("V3 Self-Test #1: 雲台追蹤 → 測 projective predict")
    print("=" * 70)

    # 場景: 飛機以 30°/s 低頭, 雲台同步以 +30°/s 抬 (反向), 目標應該不動 in pixel
    Q_dps = 30.0
    print(f"\n飛機 pitch rate: {Q_dps:.0f} deg/s, 雲台 pitch rate: -{Q_dps:.0f} deg/s (反向追)")
    print("→ 目標 in pixel 應該幾乎不動\n")

    # 模擬: 飛機 NED (0,0,300), 目標 NED (1000, 0, 0)
    target_ned = np.array([1000.0, 0.0])
    ac_x_ned, ac_y_ned, ac_alt = 0.0, 0.0, 300.0

    # 用 reproject 算每一步真實 pixel
    p_true_proj = []
    ac_pitch_seq = []
    gim_pitch_seq = []
    for k in range(N + 1):
        ac_pitch = -Q_dps * (k * dt)         # 線性低頭
        gim_pitch = +Q_dps * (k * dt)         # 線性反向抬
        u, v, valid = reproject_target_ned_to_pixel(
            target_ned[0], target_ned[1],
            ac_x_ned, ac_y_ned, ac_alt,
            0.0, ac_pitch, 0.0,
            gim_pitch, 0.0,
            fx, fy, cx, cy)
        p_true_proj.append((u, v))
        ac_pitch_seq.append(ac_pitch)
        gim_pitch_seq.append(gim_pitch)

    print(f"真實 pixel 起點: ({p_true_proj[0][0]:.1f}, {p_true_proj[0][1]:.1f})")
    print(f"真實 pixel 終點: ({p_true_proj[-1][0]:.1f}, {p_true_proj[-1][1]:.1f})  "
          f"→ 偏移: dy={p_true_proj[-1][1]-p_true_proj[0][1]:+.1f}")

    # V3 projective DKF
    dkf3 = EgoAwareDKF(dt=dt, fx=fx, fy=fy, cx=cx, cy=cy, enable_ego_comp=True)
    dkf3.x[0:2] = np.array([p_true_proj[0][0], p_true_proj[0][1]])
    dkf3.initialized = True
    dkf3._history.append((dkf3.x.copy(), dkf3.P.copy(), np.zeros(2)))
    # 給初始 prev state, 這樣第一步就能跑 ego comp
    dkf3._prev_ac_state = dict(x_ned=ac_x_ned, y_ned=ac_y_ned, alt_agl=ac_alt,
                                roll_deg=0.0, pitch_deg=ac_pitch_seq[0],
                                yaw_true_deg=0.0)
    dkf3._prev_gimbal_state = dict(pitch_deg=gim_pitch_seq[0], yaw_deg=0.0)

    # V2 風格 (純 CV)
    dkf_cv = EgoAwareDKF(dt=dt, fx=fx, fy=fy, cx=cx, cy=cy, enable_ego_comp=False)
    dkf_cv.x[0:2] = np.array([p_true_proj[0][0], p_true_proj[0][1]])
    dkf_cv.initialized = True
    dkf_cv._history.append((dkf_cv.x.copy(), dkf_cv.P.copy(), np.zeros(2)))

    for k in range(1, N + 1):
        ac_state = dict(x_ned=ac_x_ned, y_ned=ac_y_ned, alt_agl=ac_alt,
                         roll_deg=0.0, pitch_deg=ac_pitch_seq[k], yaw_true_deg=0.0)
        gim_state = dict(pitch_deg=gim_pitch_seq[k], yaw_deg=0.0)

        # Projective: 用當前 ac+gimbal+depth
        slant = math.sqrt(target_ned[0]**2 + ac_alt**2)
        dkf3.predict(ac_state=ac_state, gimbal_state=gim_state, depth_m=slant)

        # CV: 純 predict
        dkf_cv.predict()

    true_end = np.array(p_true_proj[-1])
    err_proj = np.linalg.norm(dkf3.get_position() - true_end)
    err_cv   = np.linalg.norm(dkf_cv.get_position() - true_end)

    print(f"\nV3 projective predict:    終點估計 ({dkf3.get_position()[0]:.1f}, {dkf3.get_position()[1]:.1f})  "
          f"誤差 = {err_proj:.1f} px")
    print(f"V2 風格純 CV:             終點估計 ({dkf_cv.get_position()[0]:.1f}, {dkf_cv.get_position()[1]:.1f})  "
          f"誤差 = {err_cv:.1f} px")

    # ─── Self-test #2: NED 反投影 sanity ───
    print("\n" + "-" * 70)
    print("V3 Self-Test #2: NED 反投影 sanity check")
    print("-" * 70)

    # world-stabilized 慣例: 朝向只看雲台世界角, 飛機姿態不影響.
    # 所以是「雲台 pitch 對準目標」→ 期望投影在畫面中心.
    target_x_ned, target_y_ned = 800.0, 0.0
    ac_x_ned, ac_y_ned = 0.0, 0.0
    ac_alt_agl = 300.0
    ac_roll, ac_pitch, ac_yaw = 0.0, 0.0, 0.0
    gimbal_pitch = -math.degrees(math.atan2(300, 800))   # 負=向下看
    gimbal_yaw = 0.0

    u, v, valid = reproject_target_ned_to_pixel(
        target_x_ned, target_y_ned,
        ac_x_ned, ac_y_ned, ac_alt_agl,
        ac_roll, ac_pitch, ac_yaw,
        gimbal_pitch, gimbal_yaw,
        fx, fy, cx, cy)

    print(f"  雲台 pitch={gimbal_pitch:.3f}° (剛好對準目標), 反投影 pixel: ({u:.1f}, {v:.1f})")
    print(f"  期望畫面中心 ({cx:.0f}, {cy:.0f}), "
          f"偏差: dx={u-cx:+.2f} dy={v-cy:+.2f} pixel  "
          f"{'✓ PASS' if abs(u-cx) < 1 and abs(v-cy) < 1 else '✗ FAIL'}")


if __name__ == "__main__":
    _self_test()
