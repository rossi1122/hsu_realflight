# -*- coding: utf-8 -*-
"""
config.py — vision_dive 全部參數集中在這裡
═══════════════════════════════════════════════════════════════════
任務: AUTO 繞圈對準 → 撥 GUIDED 開始 → SEARCH(雲台鎖定目標) → PRE_DIVE
      → 30° DIVE(影像 PID, 雲台 FPV body-fixed) → PULLOUT 拉平 → TECS 平飛
感知: 單目相機 (SIYI A8 mini, 開機固定 2x) + YOLOv8 + DKF + 射線投影測距
      目標由視覺提供 (真實 bbox → NED 座標), 不依賴 GPS 目標先驗.
機體: HERO2180 VTOL 固定翼 (巡航 17-22 m/s, 失速 12 m/s, MTOW 7kg)

★ 一個目標假設: 全程只鎖一台車. 選框規則 = 「離 DKF 預測點最近」+ 地理閘門,
  別的目標 conf 再高也搶不走. 切 GUIDED 前雲台不追 (FOLLOW yaw 跟機頭), 對準
  即撥 GUIDED, 目標就在畫面中央 → 首次截獲取「離畫面中心最近」.
"""

import math


class Config:
    # ════════════════════════════════════════════════════════════
    #  任務幾何
    # ════════════════════════════════════════════════════════════
    CRUISE_AGL_M = 120.0        # 巡航/進場高度 (AUTO 繞圈高度)
    DIVE_ANGLE_DEG = 30.0       # 俯衝角 (正值)
    PULLOUT_AGL_M = 70.0
    PULLOUT_VMAX_MPS = 30.0     # 超速即提前拉平 (安全上限, 幾乎每趟由此觸發)
    TECS_HOLD_ALT_M = 70.0      # 拉平後 TECS 保持高度 (進 PRE_DIVE 時就先送,
                                #   讓 TECS 內部積分器/濾波器有時間收斂)
    TECS_SPEED_MPS = 18.0       # 拉平 / DONE 的 TECS 目標空速
    # ★ 2026/09/11: SEARCH 段的空速獨立出來, 與 AUTO 巡航一致.
    #   撥 GUIDED 後 SEARCH 的 pitch/高度/速度全交 TECS, 若沿用 18 會與
    #   AUTO 的巡航速度不同, 一撥進去就開始加速, 白白擾動配平.
    #   拉平仍用 TECS_SPEED_MPS (較高的空速對拉起的能量比較有利).
    #   ⚠ 43000 CHANGE_SPEED 若超出飛控的空速上下限會被拒 (result=2).
    #     參數名依韌體版本: ArduPlane 4.4 以後是 AIRSPEED_MIN / AIRSPEED_MAX,
    #     之前叫 ARSPD_FBW_MIN / ARSPD_FBW_MAX. 必須 AIRSPEED_MIN ≤ 這個值.
    #   建議對齊飛控的 AIRSPEED_CRUISE (舊名 TRIM_ARSPD_CM), 也就是 AUTO
    #     實際在飛的巡航空速 — 這樣撥 GUIDED 不會造成加減速擾動.
    SEARCH_SPEED_MPS = 15.0     # SEARCH 巡航空速 (與 AUTO 一致)

    # 攻角 (2026/08/18 pitch step log 校準, AOA 隨速度變)
    ALPHA_NOM_DEG = 3.5         # 巡航攻角
    ALPHA_DIVE_DEG = 1.3        # ★ 俯衝攻角 — 瞄準鏈用這個

    # PRE_DIVE: 定高逼近 → 距錐界+margin 直接 -30 → pitch 壓到容差內交棒
    PREDIVE_PUSH_MARGIN_M = 20.0
    PREDIVE_HANDOFF_PITCH_TOL_DEG = 6.0
    PREDIVE_SLANT_MARGIN = 1.30       # SEARCH→PRE_DIVE: slant ≤ margin·h/sin(dive)

    # ════════════════════════════════════════════════════════════
    #  空速 / 油門 (HERO2180)
    # ════════════════════════════════════════════════════════════
    V_CRUISE_MPS = 20.0
    V_STALL_MPS = 12.0
    THR_TRIM = 0.45
    THR_MIN = 0.05             # DIVE 怠速
    THR_MAX = 0.85

    # ════════════════════════════════════════════════════════════
    #  安全 (與實飛 SOP 對齊)
    # ════════════════════════════════════════════════════════════
    ABORT_MIN_AGL_M = 55.0
    ABORT_TARGET_LOST_S = 1.5   # DIVE 段目標丟失多久 → ABORT

    # ════════════════════════════════════════════════════════════
    #  相機內參 — ★ 1x 標定值 (2026/08 t2 機上 A8 mini)
    #  程式啟動時依 ZOOM_LEVEL 一次縮放成有效內參 (見 effective_intrinsics)
    # ════════════════════════════════════════════════════════════
    IMG_W = 1920
    IMG_H = 1080
    FX_1X = 1066.7
    FY_1X = 1059.5
    CX_1X = 959.4932
    CY_1X = 545.0292

    # ── 變焦 (開機固定, 不做 auto-zoom) ──
    ZOOM_LEVEL = 2.0            # ★ 開機送 requestAbsoluteZoom(2); 內參乘 2 一次算死
                               #   A8 mini 4K sensor 出 1080p, 2x 內幾乎無損裁切.

    # ════════════════════════════════════════════════════════════
    #  YOLO
    # ════════════════════════════════════════════════════════════
    YOLO_MODEL_PATH = '/home/hsu/catkin_ws/src/hsu_realflight/hsu_visual/weights/best_car9.pt'
    YOLO_CONF = 0.4            # 偵測信心門檻
    YOLO_TARGET_CLASS_IDS = None   # None = 不過濾類別 (模型只有 1 類車)

    # ════════════════════════════════════════════════════════════
    #  影像延遲補償 (餵 DKF; DKF 內部 retrodiction 回溯補償)
    # ════════════════════════════════════════════════════════════
    # DKF 吃「延遲補償後的當前時刻估測」餵給 PID (use_yolo_when_available=False).
    # 量測延遲 = YOLO 推論時間 + 擷取鏈延遲(下方常數). 前者每幀量, 後者校一次填.
    CAPTURE_LATENCY_MS = 0.0   # ★ HDMI 擷取卡 glass-to-frame 延遲 (校一次填;
                               #   0=只補 YOLO 推論. 校法: 對相機閃燈量時間差)

    # ════════════════════════════════════════════════════════════
    #  相機 / 主迴圈頻率
    # ════════════════════════════════════════════════════════════
    CAM_INDEX = 0
    # ★ 2026/09/11 拆成兩個值 (test_camera_bench 實測結論):
    #   CAM_FPS        = 標稱值, 用在 raw 影片檔頭 + DKF 的幀數型參數
    #   CAM_REQUEST_FPS = 向 V4L2 要求的值, 要【刻意設高】
    #   原因: BUFFERSIZE=1 時消費端稍慢就會固定錯過下一幀 ->
    #         要求 30 拿到 15 (剛好一半). 要求 60 則量化變細, 損失較小.
    #   單獨測試取像路徑可達 60 fps; 完整程式跑起來因 CPU 競爭會下降,
    #   實際值每趟都會印在收尾的「相機實際 fps」.
    CAM_FPS = 30              # 標稱 fps (raw 影片時基 / DKF 幀數參數)
    CAM_REQUEST_FPS = 60      # 向 V4L2 要求的 fps (留排擠餘裕)
    CONTROL_HZ = 50.0        # 控制迴路頻率 (MAVROS + bridge)

    # ════════════════════════════════════════════════════════════
    #  雲台 (SIYI A8 mini)
    # ════════════════════════════════════════════════════════════
    SIYI_IP = '192.168.144.25'
    SIYI_PORT = 37260
    # STANDBY (未 GUIDED, AUTO 繞圈中): FOLLOW 模式, yaw=0 跟機頭, 只低頭看前方
    GIMBAL_STANDBY_PITCH_DEG = -22.0   # atan(120/300)≈22°; 對準 300m 前方地面
    GIMBAL_STANDBY_HOLD_HZ = 2.0       # STANDBY 重送 setAngles 頻率
    # SEARCH/PRE_DIVE 追蹤 (FOLLOW + 分區 P, 與 test_gimbal 一致)
    GIMBAL_TRACK_CMD_HZ = 5.0
    GIMBAL_LOST_TIMEOUT_S = 5.0        # 追蹤丟失多久回 STANDBY 指向
    GIMBAL_LOCK_TIME_S = 0.5           # 截獲後穩定多久才算 TRACKING
    # SIYI → 估測器慣例 (地面校正這幾個; 見 ray_projection_distance)
    SIYI_YAW_SIGN = -1.0
    SIYI_PITCH_SIGN = 1.0
    GIMBAL_PITCH_OFFSET_DEG = 0.0      # ★ 首飛前地面校正 (光軸 vs 報告角常數偏差)

    # ════════════════════════════════════════════════════════════
    #  地理閘門 (防跳目標: 偵測的 NED 位置須落在預期目標半徑內才接受)
    # ════════════════════════════════════════════════════════════
    GATE_RADIUS_M = 60.0       # 靜止目標基礎半徑
    GATE_MOVING_VMAX_MPS = 12.0  # 動態目標: 半徑隨丟失時間膨脹 R = R0 + vmax·t_lost
    # 首次截獲 (剛切 GUIDED): 取「離畫面中心最近」的偵測, 須連續穩定
    ACQUIRE_CENTER_NORM_MAX = 0.35   # 首次截獲: bbox 離中心 ≤ 此比例 (半對角歸一)
    ACQUIRE_STREAK = 5               # 連續幾幀滿足才建立鎖定
    # 鎖定後選框: 取「離 DKF 預測點最近」, 距離超過此像素 → 整幀視為無偵測
    TRACK_GATE_PX = 200.0            # 基礎像素閘門
    TRACK_GATE_PX_PER_S = 300.0      # 隨掉幀放寬 (px/s)

    # ════════════════════════════════════════════════════════════
    #  DIVE 段影像 PID (調校入口; 語意見 dive_guidance.GuidanceConfig)
    # ════════════════════════════════════════════════════════════
    PID_LAT_KP = 4.5
    PID_LAT_KI = 0.90
    PID_LAT_KD = 0.60
    PID_LON_KP = 1.3
    PID_LON_KI = 0.40
    PID_LON_KD = 0.30
    PID_SP_LEAD_S = 0.0
    PID_MEAS_TAU_S = 0.10
    PID_D_TAU_S = 0.20
    PID_OUT_TAU_S = 0.0

    # ════════════════════════════════════════════════════════════
    #  DIVE / PRE_DIVE 橫向: 方向舵迴路 (2026/09 掛機驗證後新增)
    # ════════════════════════════════════════════════════════════
    #  ★ 背景: ArduPlane 的 SET_ATTITUDE_TARGET, yaw 欄位【不是航向目標】,
    #    而是方向舵舵量 (度 ×100 = centidegree, 45° = 滿舵). 閉迴路要自己寫.
    #    掛機實測的伺服映射: SERVO4 MIN=1123 TRIM=1595 MAX=1900,
    #    正向 6.78 PWM/度, 負向 10.49 PWM/度 (TRIM 不在幾何中點).
    #
    #  ★ 限幅換算 (滿舵 45°):
    #      ±3°  =  6.7% 權限  → 正/負 ΔPWM  +20 / -31
    #      ±6°  = 13.3%       → +41 / -63
    #      ±15° = 33.3%       → +102 / -157
    #
    #  ⚠ V 尾 (ruddervator) 行程排擠: 兩片舵面 = 升降舵 ± 方向舵, 各自夾
    #    ±4500. 限幅越大, 俯衝時越容易把其中一片推到端點 —— 那會同時
    #    (a) 損失方向舵差動、(b) 產生不對稱的俯仰擾動. 實測門檻:
    #      ±6°  → 升降舵到 -4000 才開始排擠
    #      ±15° → 升降舵到 -3500 就損失 17% 差動, -4000 損失 33%
    #    實飛務必檢查 rcout2/rcout4 有沒有貼在端點.
    RUDDER_LIMIT_DEG = 15.0      # 方向舵總限幅 (度, 45=滿舵)
    K_PSI = 0.30                 # PRE_DIVE 用: 航向誤差(度) → 舵(度) — 目前
                                 #   PRE_DIVE 方向舵強制歸中, 此值暫未生效
    K_R = 0.15                   # PRE_DIVE 用: 偏航角速度阻尼
    K_LAT_TO_RUDDER = 0.50       # DIVE: 影像 PID 輸出 dpsi(度) → 舵(度)
    RUDDER_SLEW_DPS = 20.0       # 舵量變化率上限 (度/s)
    # 空速排程: 動壓 ∝ V², 越快權限越小. (上限空速, 權限倍率)
    RUDDER_VSCHED = ((22.0, 1.00), (26.0, 0.70), (999.0, 0.45))
    # roll 保護: 方向舵一定誘發滾轉, roll 太大就收手
    ROLL_PROT_FULL_DEG = 3.0     # |roll| ≤ 此值 → 全權限
    ROLL_PROT_ZERO_DEG = 8.0     # |roll| ≥ 此值 → 舵量歸零

    # ════════════════════════════════════════════════════════════
    #  虛擬目標 (控制路徑真相來源: 已知 GPS)
    # ════════════════════════════════════════════════════════════
    # ── 靜止版 (--static): 目標 GPS + PRE_DIVE 點 GPS, 進俯衝到固定點 ──
    #TARGET_GPS = (22.910215, 120.310426)        # 靜止目標 GPS (lat, lon)
    #PREDIVE_GPS = (22.9102299, 120.3108069) 
    TARGET_GPS = (22.913450, 120.312292)        # 靜止目標 GPS (lat, lon)
    PREDIVE_GPS = (22.911484, 120.312434)          # PRE_DIVE 點 GPS; 飛機水平到達即進 PRE_DIVE
    # ★ 2026/09/12: 30 → 15 m.
    #   這是【硬閘門】: 飛機的水平航跡必須真的進到這個半徑內才會進 PRE_DIVE.
    #   半徑越小, 進俯衝的位置越精準; 但 AUTO 航線的橫向誤差若大於這個值,
    #   飛機會擦身而過, 任務就【永遠不會觸發】(SEARCH 沒有逾時, 只會一直等).
    #   上機前請確認 AUTO 通過 PRE_DIVE 點的橫向精度優於 15 m.
    PREDIVE_REACH_RADIUS_M = 15.0

    # ── 移動版 (預設): 目標起點 GPS + 方位 + 速度, 進俯衝用「到目標距離」──
    #   方位由「起點→朝向點」兩座標算死 (= 馬路方向, 不用飛機航向).
    #   實際馬路 (2026 場地): 起點→朝向點 方位約 355.5°, 全長約 358m.
    MOVING_TARGET = True            # True=移動目標 (預設); False=靜止
    MOVING_START_GPS = (22.913647, 120.312259)   # 車起點 (切 GUIDED 時車在此)
    MOVING_HEADING_GPS = (22.9149299, 120.3121517)  # 朝此點開 → 算方位
    MOVING_HEADING_DEG = None       # 直接指定方位 (非 None 則覆寫上面兩點的算法)
    TARGET_SPEED_MPS = 10.0         # 車速 (等速)
    PREDIVE_DIST_M = 200.0          # 移動版進俯衝: 飛機到目標水平距 ≤ 此值
                                    #   30° 幾何: 120m 高對應 ~208m; 追擊需實飛調

    # ════════════════════════════════════════════════════════════
    #  輸出 / 斷電保護
    # ════════════════════════════════════════════════════════════
    OUT_DIR = 'flight_logs'    # log 根目錄
    VIDEO_FPS = 25             # YOLO 標註影片 fps
    FSYNC_HZ = 4.0             # 背景 fsync 頻率 (SD 卡防丟; NVMe 可調高)

    def __init__(self, **overrides):
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise KeyError(f"Config 沒有參數 {k}")
            setattr(self, k, v)

    # ------------------------------------------------------------------
    def effective_intrinsics(self):
        """依 ZOOM_LEVEL 縮放 1x 內參 → 有效內參 (fx, fy, cx, cy).
        zoom z: fx'=fx·z, cx'=(cx-W/2)·z + W/2 (中心不變, 焦距與偏移同縮放)."""
        z = float(self.ZOOM_LEVEL)
        fx = self.FX_1X * z
        fy = self.FY_1X * z
        cx = (self.CX_1X - self.IMG_W / 2.0) * z + self.IMG_W / 2.0
        cy = (self.CY_1X - self.IMG_H / 2.0) * z + self.IMG_H / 2.0
        return fx, fy, cx, cy

    @property
    def dive_aim_v_offset_px(self):
        """攻角補償: 縱向 setpoint 下移 fy·tan(ALPHA_DIVE) 像素 (FPV tilt≈0 用)."""
        fx, fy, cx, cy = self.effective_intrinsics()
        return fy * math.tan(math.radians(self.ALPHA_DIVE_DEG))

    @property
    def predive_ground_range_m(self):
        return self.CRUISE_AGL_M / math.tan(math.radians(self.DIVE_ANGLE_DEG))
