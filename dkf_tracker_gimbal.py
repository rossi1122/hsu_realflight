#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
dkf_tracker_gimbal.py - YOLO 漏偵測補位用 DKF wrapper (純 CV mode)
================================================================================

把 EgoAwareDKF (vision_driver_car_dive_with_dkf_v3) 包成一個簡單介面,
專門解決「YOLO 偶發漏偵測 / 短暫遮擋 / 目標跳出框」時, 短時間靠 DKF 預測位置補上.

純 CV mode 意思:
  - 不需要 MAVROS 飛行狀態
  - 不需要雲台姿態
  - 不需要相機內參標定
  - 不需要 depth / AGL
  → 退化為「constant velocity + 動態 R + delay 補償 + coast decay」的 pixel KF
  → 適合: 四旋翼地面/低空測試, YOLO 補漏的初步驗證

使用方式:
    from dkf_tracker_gimbal import DKFTrackerGimbal

    tracker = DKFTrackerGimbal(fps=30, screen_width=1920, screen_height=1080)

    # 主迴圈 (每幀):
    u_used, v_used, dkf_used, status = tracker.step(
        detected=det['detected'] == 1,
        u_yolo=float(det['bbox_x']) if det['detected'] else None,
        v_yolo=float(det['bbox_y']) if det['detected'] else None,
        bbox_w=float(det['bbox_w']) if det['detected'] else None,
    )
    state = tracker.get_state()   # 給 CSV / overlay 用的內部狀態

回傳值:
    u_used, v_used:  給下游 (gimbal/控制/顯示) 用的座標
                     - YOLO 有時用 YOLO raw (避免濾波延遲)
                     - YOLO miss 時用 DKF 預測
                     - 從沒偵測過時 = None, None
    dkf_used:        bool, True 表示這幀靠 DKF 補位
    status:          'YOLO' / 'DKF_COAST' / 'RESET_INIT' / 'WAITING'

================================================================================
"""

import math
import numpy as np
from vision_driver_car_dive_with_dkf_v3 import EgoAwareDKF


class DKFTrackerGimbal:
    """
    EgoAwareDKF 的精簡 wrapper (純 CV mode).

    State: [u, v, u_dot, v_dot] (pixel)
    每幀 predict() + (有偵測就 update / 沒偵測就 get_position).

    對長 miss 後重新偵測有保護: 連續 miss 超過閾值, 下次 YOLO 偵測時強制
    reset DKF 而不是 update, 避免 Mahalanobis gate 把新偵測誤判為 outlier.
    """

    def __init__(self,
                 fps=30,
                 screen_width=1920,
                 screen_height=1080,
                 reset_after_miss_frames=None,
                 use_yolo_when_available=True,
                 fx=None, fy=None, cx=None, cy=None,
                 gate_threshold=1000.0,
                 coast_mode='cv',
                 outlier_max_jump_px=200.0,
                 outlier_high_conf=0.4,
                 outlier_hard_jump_px=600.0,
                 outlier_bbox_ratio=None,
                 outlier_max_age_frames=None,
                 hold_timeout_frames=None,
                 min_updates_to_coast=5,
                 enable_full_ego_comp=False,
                 verbose=True):
        """
        Args:
            ...
            outlier_high_conf: 信心高於此值的偵測:
                - 連續追蹤場景: 永遠接受 (不管跳多遠)
                - 初始化 / 長 miss 後重新出現: 必須通過此門檻才接受
                預設 0.4 - 平衡「擋 false positive」vs「讓真實低信心目標進得來」.
                你的 YOLO 模型對真實目標 conf 普遍偏低時可以調更低.
            hold_timeout_frames: DKF hold 超過此幀數沒收到新 YOLO confirm → 主動清空,
                允許重新初始化. 預設 = outlier_max_age_frames (3 秒).
                避免系統永遠卡在 false positive 初始位置.
        """
        """
        Args:
            fps: 主迴圈預期執行頻率, DKF 固定 dt = 1/fps.
                 實際 FPS 抖動 ±30% 內可接受.
            screen_width, screen_height: 影像尺寸 (pixel).
            reset_after_miss_frames: 連續 miss 多少幀後, 下次 YOLO 偵測強制 reset DKF.
                                     None = 自動取 fps 數 (= 1 秒).
                                     太大 → DKF 飄太遠後 gate 拒絕新偵測.
                                     太小 → 失去 DKF 平滑效果.
            use_yolo_when_available: 偵測到時:
                                     True  = 用 YOLO raw 座標 (推薦, 避免濾波延遲)
                                     False = 用 DKF 平滑後座標
            fx, fy, cx, cy: 相機內參. 純 CV mode 不會用到, 傳 None 給 placeholder.
                            升級到 ego comp mode 時才需要真實標定值.
            verbose: 是否印 DKF 初始化訊息.
        """
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.screen_w = int(screen_width)
        self.screen_h = int(screen_height)
        self.use_yolo_when_available = bool(use_yolo_when_available)

        if reset_after_miss_frames is None:
            reset_after_miss_frames = int(fps)
        self.reset_after_miss_frames = int(reset_after_miss_frames)

        # 純 CV mode: 內參只用作 placeholder, 不會被 predict() 用到
        _fx = float(fx) if fx is not None else 1500.0
        _fy = float(fy) if fy is not None else 1500.0
        _cx = float(cx) if cx is not None else self.screen_w / 2.0
        _cy = float(cy) if cy is not None else self.screen_h / 2.0

        # ★ 新增: 給「雲台 ego comp」用的內參副本
        # 公式: 純旋轉相機, 同一個世界點在前後兩幀的 pixel 位置差:
        #   du = -fx * Δyaw_rad     (雲台往右轉 yaw+ → 像素往左移)
        #   dv = -fy * Δpitch_rad   (雲台往上轉 pitch+ → 像素往下移)
        self._ego_fx = _fx
        self._ego_fy = _fy
        # 上一步雲台姿態 (給 ego comp 算 Δyaw, Δpitch 用)
        self._prev_gimbal_yaw_deg = None
        self._prev_gimbal_pitch_deg = None
        # Ego comp 統計 (給 debug 看實際補多少 pixel)
        self._total_ego_shift_u = 0.0
        self._total_ego_shift_v = 0.0
        self._ego_comp_applied_frames = 0
        # 上一次 update 用的 delay_steps (給 get_state / CSV 看實際延遲補償了幾步)
        self._last_delay_steps = 0

        # ★ V3.6: full projective ego comp 開關
        # False (預設) = 原本行為: DKF enable_ego_comp=False, 只做雲台粗近似.
        # True          = DKF enable_ego_comp=True, step() 收到完整飛機姿態 +
        #                 雲台(估測器慣例) + depth 時, 走 EgoAwareDKF 的
        #                 projective predict (反投影→正投影扣 ego 漂移,
        #                 飛機本體 roll/pitch/yaw 全包). 資料不齊的幀自動退回
        #                 雲台粗近似 / 純 CV, 不會壞.
        self.enable_full_ego_comp = bool(enable_full_ego_comp)
        # full ego comp 統計
        self._full_ego_frames = 0          # 實際走 projective 的幀數
        self._full_ego_skipped_frames = 0  # 想走但資料不齊 → 退回粗近似的幀數

        _dkf_ego_flag = self.enable_full_ego_comp

        # 純 CV mode 也不用 image_delay (projective ego comp 不啟用)
        if verbose:
            self.dkf = EgoAwareDKF(
                dt=self.dt,
                image_delay=0.0,
                fx=_fx, fy=_fy, cx=_cx, cy=_cy,
                enable_ego_comp=_dkf_ego_flag,
            )
        else:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                self.dkf = EgoAwareDKF(
                    dt=self.dt,
                    image_delay=0.0,
                    fx=_fx, fy=_fy, cx=_cx, cy=_cy,
                    enable_ego_comp=_dkf_ego_flag,
                )

        # ★ V3.6: 依相機 dt 放大 projective 的兩個閾值.
        # EgoAwareDKF 預設值針對 100Hz(dt=0.01) 調. 相機 ~30Hz 一步是 3 倍多時間,
        # 不放大的話: 雲台一追就被當「跳變」整段不補, 合理 ego 位移也被 clip.
        # 以 dt/0.01 為比例, 並設下限避免 fps 很高時反而過鬆.
        if self.enable_full_ego_comp:
            _scale = max(1.0, self.dt / 0.01)
            self.dkf.ego_gimbal_jump_deg = 1.5 * _scale
            self.dkf.ego_dpx_max_px = 30.0 * _scale
            if verbose:
                print(f"[DKFTracker] full projective ego comp = ON "
                      f"(dt={self.dt*1000:.1f}ms, 閾值縮放×{_scale:.1f}: "
                      f"雲台跳變<{self.dkf.ego_gimbal_jump_deg:.1f}°/步, "
                      f"ego位移≤{self.dkf.ego_dpx_max_px:.0f}px/步)")

        # ★ 重要: 純 CV mode 下放寬 Mahalanobis gate
        # v3 預設 gate=200 對「自身會動的飛機 + 主動雲台」太緊,
        # 會把 ego motion 造成的 pixel 跳動誤判為 outlier 拒絕 YOLO update,
        # DKF 因此永遠跟不上目標. 用更大的 gate 接受這些「看起來像 outlier 但
        # 其實是 ego motion」的量測.
        self.dkf.gate_threshold = float(gate_threshold)
        if verbose:
            print(f"[DKFTracker] gate_threshold 調整: 200.0 → {gate_threshold} "
                  f"(純 CV mode 飛機/雲台場景必要)")

        # 統計
        self._reset_count = 0
        self._total_dkf_coast_frames = 0
        self._total_yolo_frames = 0

        # 最後一次 YOLO 偵測到的 bbox 尺寸 (給 overlay 在補位時畫一樣大的框用)
        self.last_bbox_w = 0.0
        self.last_bbox_h = 0.0

        # Coast mode 設定
        if coast_mode not in ('hold', 'cv'):
            print(f"[DKFTracker] 警告: 未知的 coast_mode='{coast_mode}', 改用 'hold'")
            coast_mode = 'hold'
        self.coast_mode = coast_mode
        # 最後一次 update 後 DKF 位置 (給 hold mode 凍結用)
        self._last_update_pos = None

        # Outlier filter 設定
        self.outlier_max_jump_px = float(outlier_max_jump_px)
        self.outlier_high_conf = float(outlier_high_conf)
        # ★ 框大小一致性 gating: None=關閉. 設 1.8 → 單幀 bbox 面積比 >1.8x 或
        #   <1/1.8 且非高信心 → 拒絕 (跳到大小不同的鄰車). 俯衝漸進放大不會觸發.
        self.outlier_bbox_ratio = (float(outlier_bbox_ratio)
                                   if outlier_bbox_ratio is not None else None)
        self._last_good_bbox_area = None   # 上次接受的 bbox 面積 (px²)
        if outlier_max_age_frames is None:
            # 預設 90 幀 (3 秒 @ 30fps)
            outlier_max_age_frames = max(int(self.fps * 3.0), 30)
        self.outlier_max_age_frames = int(outlier_max_age_frames)
        # Hold timeout: DKF hold 太久沒新 confirm 就主動清空
        if hold_timeout_frames is None:
            hold_timeout_frames = self.outlier_max_age_frames
        self.hold_timeout_frames = int(hold_timeout_frames)
        # ★ 「物理上不可能的跳躍」硬上限: 連續追蹤中, 跳超過此距離
        #   一律視為 outlier, 不管 conf 多高 (high_conf 豁免在此失效).
        #   理由: 目標一幀內位移數百~上千 px 在物理上不可能, 必是 false
        #   positive / ID 跳到別物體 / YOLO 抽風. 預設 600px (≈ 1080p 畫面
        #   1/3 寬), 比 outlier_max_jump_px(200) 寬鬆, 只擋「離譜」等級.
        #   設 None 可關閉此硬門檻 (退回原行為).
        self.outlier_hard_jump_px = (float(outlier_hard_jump_px)
                                     if outlier_hard_jump_px is not None
                                     else None)
        # Outlier filter 狀態
        self._last_good_yolo_pos = None
        self._frames_since_good_yolo = 0
        self._total_outlier_rejected = 0

        # ★ 成熟度門檻: 連續成功 YOLO update (含 RESET_INIT) 數要 ≥ 此值,
        #   DKF 才會在 miss 時對外輸出 coast 預測.
        #   未成熟前 miss 一律回傳 (None, None, False, 'IMMATURE')
        #   ─ 觀測太少時 DKF 速度估計不可信, 不應該讓雲台跟著預測軌跡走.
        #   ─ reset 後計數歸零, 要重新累積.
        #   調整指引:
        #     min_updates_to_coast=1 → 等同舊行為 (一個 sample 就 coast)
        #     min_updates_to_coast=3 → 至少看到 3 幀才信任 (對 7 FPS ≈ 0.4 秒)
        #     min_updates_to_coast=5 → 較保守 (≈ 0.65 秒)
        #     min_updates_to_coast=10→ 保守 (≈ 1.3 秒, 短遮擋 < 1 秒就會被擋)
        self.min_updates_to_coast = int(min_updates_to_coast)
        self._consecutive_yolo_updates = 0

        # ★ Per-frame step info (給 CSV / 分析圖用; 每次 step() 開頭重置)
        #   把 step() 內原本只是區域變數的「這一幀做了什麼」全部寫進來,
        #   讓 get_state() 可以一次拿到, 不用上層拼湊.
        self._last_step_info = self._fresh_step_info()

        if verbose:
            print(f"[DKFTracker] coast_mode = '{coast_mode}' "
                  f"({'YOLO miss 時凍結在最後位置' if coast_mode=='hold' else 'YOLO miss 時 CV 推進'})")
            print(f"[DKFTracker] outlier filter: jump>{outlier_max_jump_px}px "
                  f"AND conf<{outlier_high_conf} → reject; "
                  f"硬上限 jump>{self.outlier_hard_jump_px}px → reject(不豁免)")

    @staticmethod
    def _fresh_step_info():
        """每幀重置成這個空白範本, 然後 step() 過程中填上實際值."""
        return {
            # Ego compensation per-frame
            'ego_comp_mode': 'NONE',          # NONE/GIMBAL_ROTATION/FULL_PROJECTIVE/FALLBACK_CV
            'ego_comp_applied': 0,            # 0/1 — 此幀有沒有實際套用 ego shift
            'ego_du_px': 0.0,                 # 此幀套用的 u 方向 shift (px)
            'ego_dv_px': 0.0,                 # 此幀套用的 v 方向 shift (px)
            'full_ego_used': 0,               # 0/1 — 此幀有沒有走 full projective
            'full_ego_skipped': 0,            # 0/1 — 想用 full 但跳過了
            'full_ego_skip_reason': 'NONE',   # 跳過原因
            # Outlier per-frame
            'outlier_rejected_this_frame': 0, # 0/1
            'outlier_reason': 'NONE',         # NONE/LOW_CONF_JUMP/HARD_JUMP/LOW_CONF_RESET/...
            'yolo_jump_px': 0.0,              # 此幀 YOLO 距上次 good YOLO 的距離
            # Innovation (從底層 DKF 拿)
            'innovation_u': 0.0,
            'innovation_v': 0.0,
            'innovation_norm': 0.0,
            'mahalanobis2': 0.0,
            # Status
            'status': 'WAITING',
        }

    # ----------------------------------------------------------------
    #  主介面
    # ----------------------------------------------------------------
    def step(self, detected, u_yolo=None, v_yolo=None, bbox_w=None, bbox_h=None,
             dt=None,
             confidence=None, gimbal_yaw_deg=None, gimbal_pitch_deg=None,
             measurement_delay_ms=0.0,
             ac_roll_ned_deg=None, ac_pitch_ned_deg=None, ac_yaw_true_deg=None,
             ac_x_ned=None, ac_y_ned=None, ac_alt_agl=None,
             gimbal_pitch_est_deg=None, gimbal_yaw_est_deg=None,
             target_depth_m=None):
        """
        每幀呼叫一次. 順序: predict → (gimbal ego comp) → outlier → (update or coast).

        Args:
            detected: YOLO 是否偵測到此幀的目標 (bool).
            u_yolo, v_yolo: YOLO bbox 中心 pixel (detected=True 時必傳).
            bbox_w: bbox 寬度 (pixel), 給 DKF 動態 R 用. None 用預設 R.
            bbox_h: bbox 高度 (pixel), 給 overlay 畫框尺寸用. None 不更新.
            confidence: YOLO 偵測信心 (0~1). 給 outlier filter 用.
            gimbal_yaw_deg: 雲台當前 yaw (度, SIYI 原始). 給「雲台粗近似」ego
                comp 用 (full ego comp 關閉, 或下面 full 參數不齊時的後援).
            gimbal_pitch_deg: 雲台當前 pitch (度, SIYI 原始). 同上.
            measurement_delay_ms: 此次量測延遲 (ms). 建議傳 inference_ms.

            ───── 以下為 V3.6 full projective ego comp 參數 (全部選填) ─────
            只有 enable_full_ego_comp=True 且「飛機姿態三軸 + AGL +
            雲台估測慣例角度 + depth」全部有值, 這一幀才會走 projective
            predict (飛機本體 roll/pitch/yaw 漂移也扣掉). 缺任何一項 →
            這一幀自動退回雲台粗近似 / 純 CV, 不會壞.

            ac_roll_ned_deg, ac_pitch_ned_deg, ac_yaw_true_deg:
                飛機姿態, NED 慣例, deg. yaw 為真航向 (0=北, 90=東).
                由 MAVROS ENU 經 convert_mavros_enu_to_ned_attitude() 轉出,
                必須跟距離估測器用同一套轉換, 投影鏈才一致.
            ac_x_ned, ac_y_ned: 飛機 NED 北/東座標 (m). 沒有可給 0
                (ego comp 只看前後兩幀差, 常數偏移會抵銷).
            ac_alt_agl: 飛機 AGL 高度 (m).
            gimbal_pitch_est_deg, gimbal_yaw_est_deg:
                雲台角度, 估測器慣例 (經 convert_siyi_gimbal_to_estimator()
                轉出, 與距離估測器同一套). 注意: 這跟 gimbal_*_deg 的
                SIYI 原始值不同, 不可混用.
            target_depth_m: 目標斜距 (m), 通常傳上一幀的 slant_range
                (depth 變化遠慢於單幀, 一幀延遲可接受, 與主控制器
                depth_ema 同作法).

        Returns:
            (u_used, v_used, dkf_used, status):
                status: 'YOLO' / 'DKF_COAST' / 'RESET_INIT' / 'WAITING' /
                        'OUTLIER' / 'IMMATURE'
                IMMATURE = DKF 已初始化但連續 YOLO update 數還不夠
                         (< min_updates_to_coast). 此時 miss 不對外輸出 coast,
                         u_used/v_used=None, dkf_used=False, 上層應視為無目標.
        """
        # ★ 重置 per-frame 紀錄
        self._last_step_info = self._fresh_step_info()
        info = self._last_step_info   # 本次 step 的紀錄 (跟外層共享, 直接改)

        # ★ v1.5: 動態 dt — 用實際幀間隔取代固定 1/fps
        if dt is not None:
            import numpy as _np
            self.dt = float(_np.clip(dt, 0.02, 0.15))

        # ── 判斷這一幀能不能走 full projective ego comp ──
        _full_ego_ready = (
            self.enable_full_ego_comp
            and ac_roll_ned_deg is not None
            and ac_pitch_ned_deg is not None
            and ac_yaw_true_deg is not None
            and ac_alt_agl is not None
            and gimbal_pitch_est_deg is not None
            and gimbal_yaw_est_deg is not None
            and target_depth_m is not None
            and target_depth_m > 1.0
        )
        # ★ 如果 enable_full_ego_comp=True 但本幀條件不齊, 找出哪個沒帶
        if self.enable_full_ego_comp and not _full_ego_ready:
            if ac_roll_ned_deg is None or ac_pitch_ned_deg is None or ac_yaw_true_deg is None:
                info['full_ego_skip_reason'] = 'NO_UAV_ATTITUDE'
            elif ac_alt_agl is None:
                info['full_ego_skip_reason'] = 'NO_AGL'
            elif gimbal_pitch_est_deg is None or gimbal_yaw_est_deg is None:
                info['full_ego_skip_reason'] = 'NO_GIMBAL_ATTITUDE'
            elif target_depth_m is None:
                info['full_ego_skip_reason'] = 'NO_DEPTH'
            elif target_depth_m <= 1.0:
                info['full_ego_skip_reason'] = 'INVALID_DEPTH'
        # 算 delay_steps (給 DKF 內部 update 用)
        if measurement_delay_ms > 0:
            delay_steps = int(round(measurement_delay_ms / 1000.0 / self.dt))
            # Clip 到合理範圍 (避免極端 outlier 把 KF 拉壞)
            delay_steps = max(0, min(delay_steps, 10))
        else:
            delay_steps = 0
        self._last_delay_steps = delay_steps

        # 1. Predict
        if _full_ego_ready:
            # ── V3.6: full projective ego comp ──
            # EgoAwareDKF 內部: 用「上一步姿態」反投影 pixel→NED, 再用
            # 「這一步姿態」正投影回 pixel, 差值就是飛機+雲台所有運動造成的
            # ego 漂移. 飛機本體 roll/pitch/yaw 全包, 不只雲台.
            ac_state = {
                'x_ned': float(ac_x_ned) if ac_x_ned is not None else 0.0,
                'y_ned': float(ac_y_ned) if ac_y_ned is not None else 0.0,
                'alt_agl': float(ac_alt_agl),
                'roll_deg': float(ac_roll_ned_deg),
                'pitch_deg': float(ac_pitch_ned_deg),
                'yaw_true_deg': float(ac_yaw_true_deg),
            }
            gimbal_state = {
                'pitch_deg': float(gimbal_pitch_est_deg),
                'yaw_deg': float(gimbal_yaw_est_deg),
            }
            self.dkf.predict(dt=self.dt,
                             ac_state=ac_state,
                             gimbal_state=gimbal_state,
                             depth_m=float(target_depth_m))
            self._full_ego_frames += 1

            # projective predict 已搬了 self.dkf.x, 但 tracker 自己維護的
            # _last_update_pos / _last_good_yolo_pos (hold mode 凍結點 +
            # outlier spatial 參考) 沒被搬. 用 DKF 這拍實際套用的 ego_dpx
            # 同步它們, 行為跟粗近似路徑一致.
            ego_dpx = getattr(self.dkf, '_last_ego_dpx', None)
            if ego_dpx is not None and self.dkf.initialized:
                _du, _dv = float(ego_dpx[0]), float(ego_dpx[1])
                if (_du != 0.0 or _dv != 0.0):
                    if self._last_update_pos is not None:
                        self._last_update_pos = (
                            self._last_update_pos[0] + _du,
                            self._last_update_pos[1] + _dv)
                    if self._last_good_yolo_pos is not None:
                        self._last_good_yolo_pos = (
                            self._last_good_yolo_pos[0] + _du,
                            self._last_good_yolo_pos[1] + _dv)
                    self._total_ego_shift_u += _du
                    self._total_ego_shift_v += _dv
                    self._ego_comp_applied_frames += 1
                    # ★ per-frame
                    info['ego_du_px'] = _du
                    info['ego_dv_px'] = _dv
                    info['ego_comp_applied'] = 1
            # ★ 不管 ego_dpx 是不是 0, 走進這支 = full projective 路徑
            info['ego_comp_mode'] = 'FULL_PROJECTIVE'
            info['full_ego_used'] = 1
        else:
            # ── 後援: 純 CV predict + 雲台粗近似 ego comp ──
            if self.enable_full_ego_comp and (
                    ac_roll_ned_deg is not None or target_depth_m is not None):
                # 想走 full 但資料不齊 (e.g. MAVROS 掉線 / 還沒估到 depth)
                self._full_ego_skipped_frames += 1
                info['full_ego_skipped'] = 1
            # 每幀都 predict (純 CV: x = F @ x)
            self.dkf.predict(dt=self.dt)

            # 2. ★ 雲台 Ego compensation (純旋轉近似)
            #    Predict 完後, 根據「上一步 → 這一步」的雲台姿態變化, 把 dkf state
            #    位置加上 pixel shift. 讓 DKF 認知「同一個世界點在新姿態下的 pixel」.
            #    SIYI A8 mini 約定: yaw 正 = 雲台往左, pitch 正 = 雲台往上
            #      du = +fx * Δyaw_rad
            #      dv = +fy * Δpitch_rad
            if (self.dkf.initialized
                    and gimbal_yaw_deg is not None and gimbal_pitch_deg is not None
                    and self._prev_gimbal_yaw_deg is not None):
                # ★★ 2026/09/12: 角度差必須 wrap 到 ±180.
                #   DIVE 時這裡塞的是【機身航向】(0~360 會繞圈), 不是雲台角
                #   (±150 不繞圈). 飛機每次穿越正北 359°→0°, 直接相減會得到
                #   -359° 而不是 +1°, DKF 會以為瞬間轉了一大圈.
                _dy = (gimbal_yaw_deg - self._prev_gimbal_yaw_deg + 180.0) \
                    % 360.0 - 180.0
                dy_rad = math.radians(_dy)
                dp_rad = math.radians(gimbal_pitch_deg - self._prev_gimbal_pitch_deg)
                du = self._ego_fx * dy_rad
                dv = self._ego_fy * dp_rad
                # 加到 DKF state 上 (predict 之後, 屬於 prior)
                self.dkf.x[0] += du
                self.dkf.x[1] += dv
                # 同步把 last_update_pos 跟 last_good_yolo_pos 也 shift
                # (這樣 hold mode 凍結時跟著畫面 ego shift 跑, outlier filter
                #  spatial 參考點也更新)
                if self._last_update_pos is not None:
                    self._last_update_pos = (
                        self._last_update_pos[0] + du,
                        self._last_update_pos[1] + dv,
                    )
                if self._last_good_yolo_pos is not None:
                    self._last_good_yolo_pos = (
                        self._last_good_yolo_pos[0] + du,
                        self._last_good_yolo_pos[1] + dv,
                    )
                # 統計
                self._total_ego_shift_u += du
                self._total_ego_shift_v += dv
                self._ego_comp_applied_frames += 1
                # ★ per-frame
                info['ego_du_px'] = du
                info['ego_dv_px'] = dv
                info['ego_comp_applied'] = 1
                info['ego_comp_mode'] = ('FALLBACK_CV' if self.enable_full_ego_comp
                                         else 'GIMBAL_ROTATION')
            else:
                # 沒套用任何 ego comp (沒雲台資料 / 第一幀)
                info['ego_comp_mode'] = ('FALLBACK_CV' if self.enable_full_ego_comp
                                         else 'NONE')

        # 更新「上一步雲台姿態」(粗近似路徑用; 無條件更新, 這樣 full ego 中途
        # 缺資料退回粗近似時 prev 值是新鮮的)
        if gimbal_yaw_deg is not None and gimbal_pitch_deg is not None:
            self._prev_gimbal_yaw_deg = float(gimbal_yaw_deg)
            self._prev_gimbal_pitch_deg = float(gimbal_pitch_deg)

        # 3. Outlier filter (在 update 之前)
        is_outlier = False
        if detected and u_yolo is not None and v_yolo is not None:
            high_conf = (confidence is not None
                         and confidence >= self.outlier_high_conf)
            has_recent_good = (
                self._last_good_yolo_pos is not None
                and self._frames_since_good_yolo <= self.outlier_max_age_frames
            )
            # 算 jump (給 CSV 用)
            if self._last_good_yolo_pos is not None:
                _dx = u_yolo - self._last_good_yolo_pos[0]
                _dy = v_yolo - self._last_good_yolo_pos[1]
                info['yolo_jump_px'] = float((_dx*_dx + _dy*_dy) ** 0.5)
            # 判斷接下來會走 RESET 還是 update
            will_reset = (
                not self.dkf.initialized
                or self.dkf._consecutive_miss >= self.reset_after_miss_frames
            )

            if will_reset:
                # ★ RESET 或初始化 = 重新開始追蹤新軌跡, 必須 high_conf 才信任.
                # 不管 spatial check, 防止 false positive 把系統 RESET 到錯誤位置
                # 後形成「同位置低信心 false positive 連坐通過」的死循環.
                if not high_conf:
                    is_outlier = True
                    self._total_outlier_rejected += 1
                    info['outlier_reason'] = 'LOW_CONF_RESET'
            elif has_recent_good:
                # 連續 update 場景: 跳躍 + high_conf 任一通過即可
                dx = u_yolo - self._last_good_yolo_pos[0]
                dy = v_yolo - self._last_good_yolo_pos[1]
                jump = (dx*dx + dy*dy) ** 0.5
                if jump > self.outlier_max_jump_px and not high_conf:
                    is_outlier = True
                    self._total_outlier_rejected += 1
                    info['outlier_reason'] = 'LOW_CONF_JUMP'
                # ★ 硬上限: 跳到「物理上不可能」的程度時, high_conf 不再豁免.
                #   解掉原本「跳 1200px 但 conf 0.75 → 當成真目標放行」的 bug.
                elif (self.outlier_hard_jump_px is not None
                      and jump > self.outlier_hard_jump_px):
                    is_outlier = True
                    self._total_outlier_rejected += 1
                    info['outlier_reason'] = 'HARD_JUMP'
                # ★ 框大小一致性: 單幀 bbox 面積比突變 (跳到大小不同的鄰車).
                #   俯衝放大是漸進的 (單幀面積比通常 <1.2), 跳到鄰車才會突變.
                #   非 high_conf 才擋, 避免誤殺真目標被部分遮擋時的尺寸抖動.
                elif (self.outlier_bbox_ratio is not None
                      and bbox_w is not None and bbox_h is not None
                      and self._last_good_bbox_area is not None
                      and self._last_good_bbox_area > 1.0
                      and not high_conf):
                    area = float(bbox_w) * float(bbox_h)
                    r = area / self._last_good_bbox_area
                    if r > self.outlier_bbox_ratio or r < 1.0 / self.outlier_bbox_ratio:
                        is_outlier = True
                        self._total_outlier_rejected += 1
                        info['outlier_reason'] = 'BBOX_SIZE_JUMP'
                        info['bbox_area_ratio'] = float(r)
            else:
                # 安全網 (理論上 will_reset 已 cover)
                if not high_conf:
                    is_outlier = True
                    self._total_outlier_rejected += 1
                    info['outlier_reason'] = 'LOW_CONF_NO_RECENT'
            info['outlier_rejected_this_frame'] = 1 if is_outlier else 0

        # 4. 接受的偵測 → update or reset
        if detected and not is_outlier and u_yolo is not None and v_yolo is not None:
            self._total_yolo_frames += 1
            z = np.array([float(u_yolo), float(v_yolo)])

            # 記住最後 bbox 尺寸 (給 overlay 在補位時畫框用)
            if bbox_w is not None:
                self.last_bbox_w = float(bbox_w)
            if bbox_h is not None:
                self.last_bbox_h = float(bbox_h)
            # ★ 記住最後良好 bbox 面積 (給下一幀 bbox 大小 gating 比對)
            if bbox_w is not None and bbox_h is not None:
                self._last_good_bbox_area = float(bbox_w) * float(bbox_h)

            # 處理長 miss 後重新偵測: 避免 KF gate 把新偵測當 outlier
            if (self.dkf.initialized
                    and self.dkf._consecutive_miss >= self.reset_after_miss_frames):
                self.dkf.reset(initial_position=z)
                self.dkf._history.append((self.dkf.x.copy(), self.dkf.P.copy(), np.zeros(2)))
                self._reset_count += 1
                status = 'RESET_INIT'
            else:
                # 標準 KF update with 延遲補償
                # delay_steps > 0: 內部回退 N 步做 update 再前推到當前
                self.dkf.update(
                    z_image=z,
                    delay_steps=delay_steps,
                    bbox_size=float(bbox_w) if bbox_w is not None else None,
                )
                status = 'YOLO'
                # ★ 把底層 update() 算的 innovation 拉出來 (給 CSV)
                info['innovation_u'] = float(getattr(self.dkf, '_last_innovation',
                                                     np.zeros(2))[0])
                info['innovation_v'] = float(getattr(self.dkf, '_last_innovation',
                                                     np.zeros(2))[1])
                info['innovation_norm'] = float(getattr(self.dkf,
                                                        '_last_innovation_norm', 0.0))
                info['mahalanobis2'] = float(getattr(self.dkf,
                                                     '_last_mahalanobis2', 0.0))

            # 強制 DKF 位置 = YOLO bbox 位置 (繞過 KF gain<1 的位置落後)
            # ★ 只在無延遲補償的純 CV 場景套用. 有延遲補償時, KF update 結果
            # 已經是「N 步前 YOLO + propagation 到現在」, 強制覆蓋會抵銷補償.
            if delay_steps == 0:
                self.dkf.x[0] = float(u_yolo)
                self.dkf.x[1] = float(v_yolo)

            # 記住 update / reset 後的位置 (給 hold mode 凍結用)
            self._last_update_pos = (float(u_yolo), float(v_yolo))
            # 更新「上次良好偵測」(給 outlier filter 用)
            self._last_good_yolo_pos = (float(u_yolo), float(v_yolo))
            self._frames_since_good_yolo = 0

            # ★ 成熟度計數: RESET_INIT 重新累積, 否則 +1
            if status == 'RESET_INIT':
                self._consecutive_yolo_updates = 1
            else:
                self._consecutive_yolo_updates += 1

            if self.use_yolo_when_available:
                info['status'] = status
                return float(u_yolo), float(v_yolo), False, status
            else:
                pos = self.dkf.get_position()
                info['status'] = status
                return float(pos[0]), float(pos[1]), False, status

        # 5. YOLO miss 或 outlier → DKF hold/coast
        else:
            # 計算 outlier filter 的「上次良好偵測距現在多久」
            self._frames_since_good_yolo += 1

            # ★ Hold timeout auto-reset: 太久沒新 YOLO confirm 就「忘掉」舊位置
            # 避免系統永遠卡在 false positive 初始位置
            if (self.dkf.initialized
                    and self._frames_since_good_yolo > self.hold_timeout_frames):
                self.dkf.initialized = False
                self._last_good_yolo_pos = None
                self._last_update_pos = None
                self.dkf.x = np.zeros(4)
                self.dkf.P = np.eye(4) * 100.0
                self.dkf._consecutive_miss = 0
                self._reset_count += 1
                self._consecutive_yolo_updates = 0  # ★ 全清, 下次要重新累積

            if not self.dkf.initialized:
                info['status'] = 'WAITING'
                return None, None, False, 'WAITING'

            # ★ 成熟度檢查: 累積 update 數不夠不對外輸出 coast 預測.
            #   理由: 觀測量太少時 DKF 內部速度估計沒收斂, 預測會把
            #   雲台帶到錯誤方向. 不如直接返回「沒目標」讓上層決定.
            if self._consecutive_yolo_updates < self.min_updates_to_coast:
                info['status'] = 'IMMATURE'
                return None, None, False, 'IMMATURE'

            self._total_dkf_coast_frames += 1

            # Hold mode: 凍結在最後 update 位置, 速度歸零
            if self.coast_mode == 'hold' and self._last_update_pos is not None:
                self.dkf.x[0] = self._last_update_pos[0]
                self.dkf.x[1] = self._last_update_pos[1]
                self.dkf.x[2] = 0.0
                self.dkf.x[3] = 0.0

            pos = self.dkf.get_position()
            u = float(np.clip(pos[0], 0.0, self.screen_w - 1.0))
            v = float(np.clip(pos[1], 0.0, self.screen_h - 1.0))
            status = 'OUTLIER' if is_outlier else 'DKF_COAST'
            info['status'] = status
            return u, v, True, status

    # ----------------------------------------------------------------
    #  Getter
    # ----------------------------------------------------------------
    def get_state(self):
        """
        回傳 DKF 內部狀態 (給 CSV / overlay 用).

        Returns:
            dict:
                u, v: 估計 pixel 座標 (None 表示還沒初始化)
                vu, vv: 估計 pixel 速度 (px/s)
                pos_std_u, pos_std_v: 位置標準差 (pixel)
                consecutive_miss: 連續沒收到 update 的幀數
                latency_ms: consecutive_miss * dt * 1000 (DKF 距上次 update 的時間)
                initialized: bool
                predict_count, update_count, reject_count: 累計次數
                reset_count: 強制 reset 次數
                dkf_coast_frames: 累計靠 DKF 補位的幀數
                yolo_frames: 累計 YOLO 偵測到的幀數
        """
        stats = self.dkf.get_stats()

        if not self.dkf.initialized:
            out = {
                'u': None, 'v': None,
                'vu': 0.0, 'vv': 0.0,
                'pos_std_u': 0.0, 'pos_std_v': 0.0,
                'pos_std_norm': 0.0,
                'consecutive_miss': 0,
                'latency_ms': 0.0,
                'initialized': False,
                'predict_count': stats['predict_count'],
                'update_count': stats['update_count'],
                'reject_count': stats['reject_count'],
                'reset_count': self._reset_count,
                'dkf_coast_frames': self._total_dkf_coast_frames,
                'yolo_frames': self._total_yolo_frames,
                'last_bbox_w': self.last_bbox_w,
                'last_bbox_h': self.last_bbox_h,
                'outlier_rejected': self._total_outlier_rejected,
                'ego_comp_frames': self._ego_comp_applied_frames,
                'delay_steps': self._last_delay_steps,
                'total_ego_shift_u': self._total_ego_shift_u,
                'total_ego_shift_v': self._total_ego_shift_v,
                'full_ego_frames': self._full_ego_frames,
                'full_ego_skipped_frames': self._full_ego_skipped_frames,
                'frames_since_good_yolo': self._frames_since_good_yolo,
            }
            out.update(self._last_step_info)
            return out

        std = self.dkf.get_position_std()
        miss = stats['consecutive_miss']
        out = {
            'u': float(self.dkf.x[0]),
            'v': float(self.dkf.x[1]),
            'vu': float(self.dkf.x[2]),
            'vv': float(self.dkf.x[3]),
            'pos_std_u': float(std[0]),
            'pos_std_v': float(std[1]),
            'pos_std_norm': float((std[0]**2 + std[1]**2) ** 0.5),
            'consecutive_miss': int(miss),
            'latency_ms': miss * self.dt * 1000.0,
            'initialized': True,
            'predict_count': stats['predict_count'],
            'update_count': stats['update_count'],
            'reject_count': stats['reject_count'],
            'reset_count': self._reset_count,
            'dkf_coast_frames': self._total_dkf_coast_frames,
            'yolo_frames': self._total_yolo_frames,
            'last_bbox_w': self.last_bbox_w,
            'last_bbox_h': self.last_bbox_h,
            'outlier_rejected': self._total_outlier_rejected,
            'ego_comp_frames': self._ego_comp_applied_frames,
            'delay_steps': self._last_delay_steps,
            'total_ego_shift_u': self._total_ego_shift_u,
            'total_ego_shift_v': self._total_ego_shift_v,
            'full_ego_frames': self._full_ego_frames,
            'full_ego_skipped_frames': self._full_ego_skipped_frames,
            'frames_since_good_yolo': self._frames_since_good_yolo,
            'consecutive_yolo_updates': self._consecutive_yolo_updates,
            'min_updates_to_coast': self.min_updates_to_coast,
        }
        # ★ 把 per-frame step_info 一起 merge 進來
        out.update(self._last_step_info)
        return out

    def reset(self):
        """完全重置 (用於任務切換 / target 切換)"""
        self.dkf.reset()
        self._reset_count = 0
        self._total_dkf_coast_frames = 0
        self._total_yolo_frames = 0
        self._consecutive_yolo_updates = 0  # ★ 成熟度計數歸零
        self._last_good_yolo_pos = None
        self._frames_since_good_yolo = 0


# ════════════════════════════════════════════════════════════════════
#  Self-test
# ════════════════════════════════════════════════════════════════════
def _self_test():
    """
    模擬場景:
      Phase 1 (30 幀): YOLO 正常, 目標等速移動
      Phase 2 (25 幀): YOLO 全 miss (約 0.83 秒)
      Phase 3 (10 幀): YOLO 恢復, 但目標已遠離 DKF 預測位置 (機動)
    驗證: DKF 補位誤差是否合理, RESET_INIT 是否正常觸發.
    """
    import random
    random.seed(42)

    print("=" * 70)
    print("DKFTracker Self-Test")
    print("=" * 70)

    tracker = DKFTrackerGimbal(fps=30, screen_width=1920, screen_height=1080,
                               reset_after_miss_frames=20,  # 短一點方便觸發 RESET
                               verbose=True)

    def true_pos(idx):
        """目標真實軌跡: 前 55 幀等速, 之後突然加速 (模擬機動 + 長 miss)"""
        if idx < 55:
            return (500.0 + idx * 6.67, 400.0 + idx * 3.33)
        else:
            # 機動: 速度變大且方向改變
            base_u = 500.0 + 55 * 6.67
            base_v = 400.0 + 55 * 3.33
            return (base_u + (idx - 55) * 20.0, base_v - (idx - 55) * 5.0)

    print("\n--- Phase 1: YOLO 正常 30 幀 ---")
    for i in range(30):
        u_true, v_true = true_pos(i)
        u_n = u_true + random.uniform(-2, 2)
        v_n = v_true + random.uniform(-2, 2)
        u, v, used, st = tracker.step(detected=True, u_yolo=u_n, v_yolo=v_n,
                                      bbox_w=30, confidence=0.9)
        if i % 10 == 0 or i == 29:
            print(f"  [{i:3d}] true=({u_true:6.1f},{v_true:6.1f}) "
                  f"yolo=({u_n:6.1f},{v_n:6.1f}) used=({u:6.1f},{v:6.1f}) {st}")

    print("\n--- Phase 2: YOLO miss 25 幀 (目標 idx 30~54, 仍等速) ---")
    for i in range(25):
        idx = 30 + i
        u_true, v_true = true_pos(idx)
        u, v, used, st = tracker.step(detected=False)
        if i % 5 == 0 or i == 24:
            err = ((u - u_true) ** 2 + (v - v_true) ** 2) ** 0.5
            state = tracker.get_state()
            print(f"  [{idx:3d}] true=({u_true:6.1f},{v_true:6.1f}) "
                  f"DKF=({u:6.1f},{v:6.1f}) err={err:5.1f}px miss={state['consecutive_miss']:3d} {st}")

    print("\n--- Phase 3: YOLO 恢復 10 幀, 但目標已機動 (idx 55+) ---")
    for i in range(10):
        idx = 55 + i
        u_true, v_true = true_pos(idx)
        u_n = u_true + random.uniform(-2, 2)
        v_n = v_true + random.uniform(-2, 2)
        u, v, used, st = tracker.step(detected=True, u_yolo=u_n, v_yolo=v_n,
                                      bbox_w=30, confidence=0.9)
        print(f"  [{idx:3d}] true=({u_true:6.1f},{v_true:6.1f}) "
              f"yolo=({u_n:6.1f},{v_n:6.1f}) used=({u:6.1f},{v:6.1f}) {st}")

    final = tracker.get_state()
    print("\n" + "=" * 70)
    print(f"  Summary:")
    print(f"    predict_count    = {final['predict_count']}")
    print(f"    update_count     = {final['update_count']}")
    print(f"    reset_count      = {final['reset_count']}")
    print(f"    reject_count     = {final['reject_count']}")
    print(f"    dkf_coast_frames = {final['dkf_coast_frames']}")
    print(f"    yolo_frames      = {final['yolo_frames']}")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()
