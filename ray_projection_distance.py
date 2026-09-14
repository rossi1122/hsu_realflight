#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ray_projection_distance.py - 實飛專用射線投影距離估測 (Real-Flight V1.0)
================================================================================

把模擬版 (gimbal_controller_pinhole.py 的 RayProjectionDistanceEstimator V7.0)
改成實飛可用版本. 三個關鍵差異:

  1. 相機內參: 模擬版用 X-Plane FOV/render/crop 反推 fx/fy/cx/cy.
     實飛版直接吃「相機標定值」(你檔頂的 CAMERA_FX / CAMERA_FY), 不再猜.

  2. 飛機姿態: 實飛從 MAVROS /mavros/imu/data 拿 (ENU 世界 + FLU 機體, 弧度).
     幾何鏈需要的是 NED 世界 + FRD 機體 + 真航向 (0=北, 90=東, 度).
     → convert_mavros_enu_to_ned_attitude() 負責換.

  3. 雲台角度: 實飛從 SIYI SDK getAttitude() 拿 (yaw, pitch, roll, 度),
     SIYI 的正負號慣例 ≠ 估測器要的慣例.
     → convert_siyi_gimbal_to_estimator() 負責換, 且把每個正負號/offset
       都做成「可調參數」, 因為這是實飛最容易出錯、必須地面校正的地方.

幾何鏈 (與 gimbal_controller_pinhole.py / vision_driver_car_dive_with_dkf_v3.py
完全一致, 故 _Rx/_Ry/_Rz 與 R_g2c 刻意逐字相同, 確保之後反投影對得起來):

    Ic           = normalize([(u-cx)/fx, (v-cy)/fy, 1])   # 像素 → 相機射線
    # SIYI A8 三軸 world-stabilized → 相機世界指向只由雲台世界角決定,
    # 不乘飛機姿態 R_w2b (飛機 roll/pitch 已被雲台自行穩定補償):
    R_ned2gimbal = Ry(gimbal_pitch_eff) @ Rz(gimbal_yaw_world)  # NED → Gimbal
    R_ned2cam    = R_g2c @ R_ned2gimbal                    # NED → Camera
    Iw           = inv(R_ned2cam) @ Ic                     # 射線在 NED 的方向
    slant        = altitude_agl / Iw[2]                    # 與地面交點

⚠️ 實飛安全須知 (務必先看):
  - slant_range (斜距) 幾乎不受「航向(yaw)誤差」影響: yaw 是繞 NED-Z(down)
    的旋轉, 不改變射線的 down 分量, 所以 alt/Iw[2] 對航向換算錯誤很魯棒.
    真正影響 slant 的是 roll / pitch / gimbal_pitch 的「正負號與 offset」.
  - target NED 方位 (北/東座標) 才需要正確航向.
  - 因此首飛前「一定要做地面校正」: 飛機與目標都不動, 雲台對準一個已知
    水平距離的目標, 看 estimate_distance 算出的 ground_range 對不對,
    用 gimbal_pitch_offset_deg 把常數偏差吃掉 (校正法見 __init__ docstring).
  - 本模組只「估測 + 記錄」, 不觸發任何動作. 數字沒地面驗證過前別拿來決策.

================================================================================
"""

import math
import numpy as np
from collections import deque


# ==================== 旋轉矩陣工具 ====================
# 與 gimbal_controller_pinhole.py / vision_driver_car_dive_with_dkf_v3.py
# 逐字相同 — 不要改, 改了反投影 (reproject_target_ned_to_pixel) 就對不上.

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


# ==================== MAVROS ENU → 估測器 NED 姿態轉換 ====================

def convert_mavros_enu_to_ned_attitude(roll_enu_rad, pitch_enu_rad, yaw_enu_rad):
    """
    把 MAVROS /mavros/imu/data 的 euler (ENU 世界 + FLU 機體, 弧度)
    轉成本估測器幾何鏈要的 (NED 世界 + FRD 機體 + 真航向, 度).

    依據 ROS REP-103 / MAVROS 慣例, 常用且工程上實證可用的對應:
        roll_ned_deg  =  deg(roll_enu)            (正 = 右翼下沉)
        pitch_ned_deg = -deg(pitch_enu)           (正 = 抬頭)
        yaw_true_deg  =  (90 - deg(yaw_enu))      (0=北, 90=東, 順時針為正)

    其中 yaw 轉換是「ENU 的 0=東、逆時針正」→「真航向 0=北、順時針正」.

    ⚠️ 這三個正負號是實飛最常見的 bug 來源. 上面是標準對應, 但不同
       飛控/IMU 安裝方向仍可能差一個符號. 首飛前務必地面驗證 (機頭朝
       已知方向看 yaw_true 對不對; 機身前傾看 pitch_ned 是否變正).

    Args:
        roll_enu_rad, pitch_enu_rad, yaw_enu_rad: euler_from_quaternion
            ([q.x,q.y,q.z,q.w]) 直接吐出來的三個值 (弧度).
    Returns:
        (roll_deg, pitch_deg, yaw_true_deg)
        yaw_true_deg 已 wrap 到 [0, 360).
    """
    roll_deg = math.degrees(roll_enu_rad)
    pitch_deg = -math.degrees(pitch_enu_rad)
    yaw_true_deg = (90.0 - math.degrees(yaw_enu_rad)) % 360.0
    return roll_deg, pitch_deg, yaw_true_deg


# ==================== SIYI 雲台 → 估測器慣例轉換 ====================

def convert_siyi_gimbal_to_estimator(siyi_yaw_deg, siyi_pitch_deg,
                                     yaw_sign=-1.0, pitch_sign=1.0,
                                     yaw_offset_deg=0.0, pitch_offset_deg=0.0):
    """
    SIYI A8 getAttitude() 回傳 (yaw, pitch, roll) 度. 依你 test 檔註解:
        SIYI yaw:   正 = 雲台往左轉      (估測器要: 正 = 往右)  → yaw_sign  = -1
        SIYI pitch: 正 = 往上, 負 = 往下  (估測器要: 負 = 往下看) → pitch_sign = +1

    把每個正負號/offset 都做成參數, 因為這正是要地面校正的地方.
    若校正發現方向相反, 改對應 sign 即可 (不用動幾何).

    Args:
        siyi_yaw_deg, siyi_pitch_deg: SIYI getAttitude() 的 yaw, pitch.
        yaw_sign, pitch_sign: ±1, 校正方向.
        yaw_offset_deg, pitch_offset_deg: 機械零位常數偏差.
    Returns:
        (gimbal_pitch_deg, gimbal_yaw_deg) — 餵給 estimate_distance().
    """
    gimbal_yaw_deg = yaw_sign * siyi_yaw_deg + yaw_offset_deg
    gimbal_pitch_deg = pitch_sign * siyi_pitch_deg + pitch_offset_deg
    return gimbal_pitch_deg, gimbal_yaw_deg


# ==================== 射線投影距離估測器 (實飛標定內參版) ====================

class RayProjectionDistanceEstimator:
    """
    射線投影距離估測器 — 實飛標定內參版.

    與模擬版 V7.0 的唯一原理差別: 內參直接吃標定值, 不從 FOV 反推.
    幾何鏈、平滑、信心度算法完全相同.

    原理:
      1. 像素 (u,v) → 相機座標歸一化射線 Ic
      2. 旋轉鏈 Camera → Gimbal → Body → World(NED)
      3. 假設目標在地面, 用 AGL 高度反算 slant = altitude / Iw[Z]
      不需要目標尺寸, 只要相機標定 + 飛機姿態 + 雲台角度 + 高度.
    """

    def __init__(self,
                 fx, fy, cx, cy,
                 ema_alpha=0.3,
                 outlier_factor=2.0,
                 gimbal_pitch_offset_deg=0.0,
                 gimbal_yaw_offset_deg=0.0,
                 history_len=15,
                 min_iw_down=0.01,
                 conf_ref_iw_down=0.30,
                 verbose=True):
        """
        Args:
            fx, fy: 相機焦距 (pixel), 用你標定的 CAMERA_FX / CAMERA_FY.
            cx, cy: 主點 (pixel). 沒標定就傳 影像寬/2, 高/2.
            ema_alpha: EMA 平滑係數 (越小越平滑, 但 dive 時會 lag —
                       下游若用 last_raw_slant 可避開 lag, 見屬性說明).
            outlier_factor: 離群倍率, raw 偏離 median 超過此倍就 clip.
            gimbal_pitch_offset_deg: ✨ 雲台報告角度 vs 實際光軸的常數偏差(度).
                正值 = 實際光軸比報告值「更朝下」.
                === 地面校正法 ===
                飛機水平靜止 + 目標放正前方已知水平距離 D, 量到的 AGL=H,
                雲台對準目標讓它在畫面正中央. 此時理論俯角 = atan2(H, D).
                記下程式算出的 ground_range, 微調此 offset 直到 ground_range≈D.
            history_len: 平滑用的歷史窗格 (median + outlier 用).
            min_iw_down: Iw[2] 小於此值視為射線沒朝下, 無法估測.
            conf_ref_iw_down: 信心度 = clip(Iw[2]/此值, 0, 1).
                Iw[2]=sin(俯角), 0.30≈俯角17°時信心滿.
        """
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.gimbal_pitch_offset_deg = float(gimbal_pitch_offset_deg)
        # SIYI A8 三軸 world-stabilized: yaw 相對上電朝向, 用此 offset 對齊真北
        # (只影響目標方位/NED 座標, 不影響斜距與地面距離). 啟動時設
        #  gimbal_yaw_offset_deg = aircraft_yaw_true_at_start - gimbal_yaw_at_start.
        self.gimbal_yaw_offset_deg = float(gimbal_yaw_offset_deg)
        self.ema_alpha = float(ema_alpha)
        self.outlier_factor = float(outlier_factor)
        self.min_iw_down = float(min_iw_down)
        self.conf_ref_iw_down = float(conf_ref_iw_down)

        # Gimbal → Camera (CV: X右 Y下 Z前). 與兩個既有檔逐字相同.
        self.R_g2c = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=float)

        self.distance_history = deque(maxlen=int(history_len))
        self.ema_distance = None

        # Debug / 下游用 (與模擬版同名, drop-in 相容):
        self.last_ray_world = None      # 射線在 NED 的單位方向 (np.array(3,))
        self.last_raw_slant = None      # 未平滑斜距 (m) — dive 時餵 KF 用這個
        self.last_ground_range = None   # 未平滑地面距 (m)
        self.last_target_ned = None     # 目標推估 NED 絕對座標 [N, E, D] (m)
        self.last_confidence = 0.0

        if verbose:
            print(f"[RayProjectionDistanceEstimator RealFlight V1.0] 初始化:")
            print(f"  相機內參(標定): fx={self.fx:.2f} fy={self.fy:.2f} "
                  f"cx={self.cx:.1f} cy={self.cy:.1f}")
            print(f"  gimbal_pitch_offset={self.gimbal_pitch_offset_deg:+.2f}° "
                  f"(0=未校正, 首飛前務必地面校正)")
            print(f"  EMA α={self.ema_alpha} outlier×{self.outlier_factor} "
                  f"history={history_len}")

    # --------------------------------------------------------------
    def estimate_distance(self, pixel_u, pixel_v,
                          aircraft_roll, aircraft_pitch, aircraft_yaw_true,
                          gimbal_pitch, gimbal_yaw,
                          altitude_agl,
                          ac_x_ned=0.0, ac_y_ned=0.0):
        """
        射線投影估測到地面目標的距離.

        Args:
            pixel_u, pixel_v: 目標像素座標 (YOLO bbox 中心 或 DKF 預測點).
            aircraft_roll: 飛機滾轉 (度, 正=右翼下沉) — 已轉好的 NED 值.
            aircraft_pitch: 飛機俯仰 (度, 正=抬頭) — 已轉好的 NED 值.
            aircraft_yaw_true: 真航向 (度, 0=北 90=東) — 已轉好的值.
            gimbal_pitch: 雲台俯仰 (度, 負=向下看) — 已轉好的估測器慣例值.
            gimbal_yaw: 雲台偏航 (度, 正=向右看) — 已轉好的估測器慣例值.
            altitude_agl: AGL 高度 (m). 必須 > 0.
            ac_x_ned, ac_y_ned: 飛機 NED 北/東座標 (m), 用來算 target 絕對
                NED. 沒有就傳 0 (此時 last_target_ned 是「相對飛機」座標).

        Returns:
            (slant_range, ground_range, confidence)
            無法估測時回 (last_estimate_or_None, None, 0.0).
            另外設好 last_ray_world / last_raw_slant / last_ground_range /
            last_target_ned / last_confidence 供下游 (DKF NED fallback) 取用.
        """
        if altitude_agl is None or altitude_agl <= 0.5:
            self.last_confidence = 0.0
            return self._get_current_estimate(), None, 0.0

        # ---- Step 1: 像素 → 相機座標歸一化射線 ----
        x1 = (pixel_u - self.cx) / self.fx
        x2 = (pixel_v - self.cy) / self.fy
        Pc = np.array([x1, x2, 1.0])
        Ic = Pc / np.linalg.norm(Pc)

        # ---- Step 2: 旋轉鏈 (SIYI A8 三軸 world-stabilized) ----
        # SIYI A8 是三軸穩定雲台: 回報 pitch/roll 是「相對水平面」的世界角,
        # yaw 是「相對上電朝向」。相機在世界(NED)的指向只由雲台世界角決定,
        # 與飛機 roll/pitch 無關(雲台已自行穩定補償) → 不可再乘 R_w2b。
        # (此版淨結果等價 test_Visual_Main.py 的 R_g2c @ R_enu2gimbal。)
        # 註: aircraft_roll/pitch 不再進入幾何; aircraft_yaw_true 僅供啟動時
        #     計算 gimbal_yaw_offset_deg 用 (見 __init__)。
        gimbal_pitch_eff = gimbal_pitch - self.gimbal_pitch_offset_deg
        gimbal_yaw_world = gimbal_yaw + self.gimbal_yaw_offset_deg
        R_ned2gimbal = _Ry(gimbal_pitch_eff) @ _Rz(gimbal_yaw_world)
        R_ned2cam = self.R_g2c @ R_ned2gimbal
        R_cam2ned = R_ned2cam.T   # 正交矩陣的逆 = 轉置

        Iw = R_cam2ned @ Ic
        self.last_ray_world = Iw.copy()

        # ---- Step 3: 與地面交叉 ----
        # NED: 飛機在 z=-altitude_agl, 地面 z=0.
        # 射線 P = [acx, acy, -alt] + t*Iw, 解 -alt + t*Iw[2]=0 → t=alt/Iw[2]
        if Iw[2] <= self.min_iw_down:
            # 射線幾乎不朝下 (水平或朝上) → 無法估測
            self.last_confidence = 0.0
            self.last_target_ned = None
            return self._smooth(None), None, 0.0

        slant_range = altitude_agl / Iw[2]
        self.last_raw_slant = slant_range

        horizontal = math.sqrt(Iw[0] ** 2 + Iw[1] ** 2)
        ground_range = slant_range * horizontal
        self.last_ground_range = ground_range

        # 目標絕對 NED 座標 (供 vision_driver_v3 的 reproject / DKF fallback)
        # 飛機 NED = [ac_x, ac_y, -alt]; target = ac_ned + slant*Iw
        ac_ned = np.array([float(ac_x_ned), float(ac_y_ned),
                           -float(altitude_agl)])
        self.last_target_ned = ac_ned + slant_range * Iw

        # ---- Step 4: 信心度 ----
        # Iw[2]=sin(俯角): 射線越陡 → 高度反算越穩 → 信心越高
        confidence = float(np.clip(Iw[2] / self.conf_ref_iw_down, 0.0, 1.0))
        self.last_confidence = confidence

        # ---- Step 5: 平滑 ----
        smoothed_slant = self._smooth(slant_range)
        if smoothed_slant is not None and smoothed_slant > altitude_agl:
            smoothed_ground = math.sqrt(smoothed_slant ** 2 - altitude_agl ** 2)
        else:
            smoothed_ground = ground_range

        return smoothed_slant, smoothed_ground, confidence

    # --------------------------------------------------------------
    def _smooth(self, raw_distance):
        """EMA + median outlier rejection (與模擬版同算法)"""
        if raw_distance is None or raw_distance <= 0:
            return self._get_current_estimate()

        if len(self.distance_history) >= 5:
            median = float(np.median(list(self.distance_history)))
            if median > 0:
                ratio = raw_distance / median
                if ratio > self.outlier_factor or ratio < (1.0 / self.outlier_factor):
                    raw_distance = float(np.clip(
                        raw_distance,
                        median / self.outlier_factor,
                        median * self.outlier_factor))

        self.distance_history.append(raw_distance)
        median_dist = float(np.median(list(self.distance_history)))

        if self.ema_distance is None:
            self.ema_distance = median_dist
        else:
            self.ema_distance = (self.ema_alpha * median_dist +
                                 (1.0 - self.ema_alpha) * self.ema_distance)
        return self.ema_distance

    def _get_current_estimate(self):
        if self.ema_distance is not None:
            return self.ema_distance
        if len(self.distance_history) > 0:
            return float(np.median(list(self.distance_history)))
        return None

    def reset(self):
        self.distance_history.clear()
        self.ema_distance = None
        self.last_ray_world = None
        self.last_raw_slant = None
        self.last_ground_range = None
        self.last_target_ned = None
        self.last_confidence = 0.0


# ==================== Self-test ====================

def _self_test():
    """
    兩個 sanity check:
      #1 飛機平飛、雲台俯角剛好對準已知水平距離的目標
         → 估出的 ground_range 應 ≈ 已知水平距離.
      #2 MAVROS ENU → NED 航向轉換點檢.
    """
    print("=" * 70)
    print("RayProjection RealFlight Self-Test")
    print("=" * 70)

    # SIYI A8 1080p 標定值 (與 test 檔頂一致)
    fx, fy = 1057.652416, 1051.784178
    W, H = 1920, 1080
    cx, cy = W / 2.0, H / 2.0

    est = RayProjectionDistanceEstimator(fx=fx, fy=fy, cx=cx, cy=cy,
                                         verbose=True)

    print("\n--- #1 幾何 sanity: 目標在畫面正中央 ---")
    H_alt = 100.0   # AGL 100 m
    for D in (100.0, 200.0, 500.0):
        # 飛機平飛 (roll=pitch=0), 機頭朝北 (yaw_true=0).
        # 雲台只用 pitch 對準: 目標水平距 D, 高度差 H → 俯角 atan2(H,D).
        # 估測器要 gimbal_pitch 負=向下, 所以雲台 pitch = -atan2(H,D).
        gimbal_pitch = -math.degrees(math.atan2(H_alt, D))
        slant, ground, conf = est.estimate_distance(
            pixel_u=cx, pixel_v=cy,            # 畫面正中央
            aircraft_roll=0.0, aircraft_pitch=0.0, aircraft_yaw_true=0.0,
            gimbal_pitch=gimbal_pitch, gimbal_yaw=0.0,
            altitude_agl=H_alt)
        true_slant = math.sqrt(D ** 2 + H_alt ** 2)
        ok = (slant is not None
              and abs(ground - D) < 1.0
              and abs(slant - true_slant) < 1.0)
        print(f"  D={D:5.0f}m gimbal_pitch={gimbal_pitch:7.2f}° → "
              f"slant={slant:7.1f}m (理論 {true_slant:7.1f}) "
              f"ground={ground:6.1f}m (理論 {D:5.0f}) conf={conf:.2f} "
              f"{'✓' if ok else '✗ FAIL'}")
        est.reset()

    print("\n--- #2 yaw 不影響 slant (對航向換算錯誤魯棒) ---")
    base = None
    for yaw_true in (0.0, 45.0, 90.0, 180.0, 270.0):
        gimbal_pitch = -math.degrees(math.atan2(H_alt, 300.0))
        slant, ground, conf = est.estimate_distance(
            pixel_u=cx, pixel_v=cy,
            aircraft_roll=0.0, aircraft_pitch=0.0, aircraft_yaw_true=yaw_true,
            gimbal_pitch=gimbal_pitch, gimbal_yaw=0.0,
            altitude_agl=H_alt)
        if base is None:
            base = slant
        drift = abs(slant - base)
        print(f"  yaw_true={yaw_true:6.1f}° → slant={slant:7.2f}m "
              f"(漂移 {drift:.4f}m) {'✓' if drift < 1e-6 else '✗'}")
        est.reset()

    print("\n--- #3 MAVROS ENU→NED 航向轉換點檢 ---")
    # ENU yaw: 0=東(逆時針正). 真航向: 0=北(順時針正).
    cases = [
        (0.0,   90.0,  "機頭朝東 → 真航向 90"),
        (90.0,  0.0,   "ENU yaw=90(朝北) → 真航向 0"),
        (180.0, 270.0, "ENU yaw=180(朝西) → 真航向 270"),
        (-90.0, 180.0, "ENU yaw=-90(朝南) → 真航向 180"),
    ]
    for yaw_enu_deg, expect_true, desc in cases:
        r, p, y = convert_mavros_enu_to_ned_attitude(
            0.0, 0.0, math.radians(yaw_enu_deg))
        ok = abs((y - expect_true + 180) % 360 - 180) < 1e-6
        print(f"  {desc}: 得 {y:6.1f}° {'✓' if ok else '✗ FAIL'}")

    print("\n--- #4 SIYI 雲台慣例轉換 ---")
    gp, gy = convert_siyi_gimbal_to_estimator(siyi_yaw_deg=10.0,
                                              siyi_pitch_deg=-30.0)
    print(f"  SIYI(yaw=+10 左, pitch=-30 下) → "
          f"估測器(gimbal_yaw={gy:+.1f} 右為正, gimbal_pitch={gp:+.1f} 下為負)")
    print(f"  預期: gimbal_yaw=-10 (左轉→負/右為正), gimbal_pitch=-30 (向下)")

    print("\n" + "=" * 70)
    print("Self-test 完成. 注意: #1/#2 過 = 幾何對; 但實飛 roll/pitch/雲台")
    print("正負號仍須地面校正 (見模組頂 docstring 與 offset 校正法).")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()
