# -*- coding: utf-8 -*-
"""
attitude_bridge.py — 導引律姿態命令 (pitch/heading) → ArduPlane (virtual_dive_pid 版)
═══════════════════════════════════════════════════════════════════
來源: hsu_visual/ardupilot_attitude_bridge.py 複製, 加入三個修改:

★ 修改 1: MAVROS 座標轉換修正 (2026/08/12 實飛驗證的問題)
    MAVROS /mavros/setpoint_raw/attitude 會對 quaternion 套用
    Q_combined = ENU→NED × FLU→FRD (等效繞 Z 軸 90°), 導致:
        - pitch 反轉 (要 -20° 俯衝 → 變 +20° 爬升)
        - yaw 偏移 90° (88° 東 → 0° 北)
    修法 (與 MAVROS_座標轉換錯誤報告.md 一致):
        以 NED 慣例組 quaternion → 送出前左乘 Q_CONV_INV,
        MAVROS 內部再乘 Q_combined 後恰好抵消, 飛控收到正確 NED 姿態.
    Q_CONV_INV = quaternion_inverse([0, 0, 0.70710678, 0.70710678])
               = [0, 0, -0.70710678, 0.70710678]

★ 修改 2: 油門排程接 PULLOUT / DIVE
    DIVE:    THR_MIN 怠速 (30° 俯衝重力分量 +4.9 m/s² 壓不住, 只能收;
             峰值空速由模擬預先算出, 對照 VNE)
    PULLOUT: THR_PULLOUT 大油門 (拉起防掉速)
    其他:    空速保持 (V_REF)

★ 修改 3: 航向請用 /mavros/global_position/compass_hdg
    (run_flight.py 負責訂閱; 不要用 setpoint 座標系推的 yaw)

★ 修改 4 (2026/08/24 掛機 diag_yaw.py 確診):
    SET_ATTITUDE_TARGET 的 yaw 欄位 = **絕對航向目標** (NED, 經 Q_CONV_INV),
    不是原設計假設的 rudder 舵量! (yaw=0 → rudder 一直打想轉向北;
    yaw=當下機頭 → rudder 回中, 一翻兩瞪眼.)
    → 所有模式 yaw 一律填 heading_cmd, 航向環交給飛控閉;
      bridge 內建的 δ_r = K_ψ·e_ψ − K_r·r 手動 rudder 迴路廢除.
    → DIVE (rudder+roll0): roll=0 + yaw=heading_cmd, 飛控用 rudder/
      ground-steering 追航向. ⚠ 空中飛控是否真用 rudder 追 forced yaw
      尚無實飛資料 — rudder 校準飛行 (README) 同時回答這題.

其餘:
    - cmd_active=False → 停發 → GUIDED_TIMEOUT 交還飛手
"""

import math
import threading
import time

try:
    import rospy
    from mavros_msgs.msg import AttitudeTarget
    from mavros_msgs.srv import SetMode, ParamGet, CommandInt
    ROS_OK = True
except ImportError:
    ROS_OK = False


# ════════════════════════════════════════════════════════════════════
#  quaternion 工具 (x, y, z, w), 與 tf.transformations 慣例一致
# ════════════════════════════════════════════════════════════════════

def euler_to_quaternion(roll, pitch, yaw):
    """ZYX (yaw-pitch-roll) → quaternion (x, y, z, w). 弧度. NED 慣例."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_sxyz(ai, aj, ak):
    """tf.transformations.quaternion_from_euler(ai, aj, ak) 'sxyz' 等價式
    (靜態軸 X→Y→Z). 回傳 (x, y, z, w). 逐字對齊飛過的 dive_controller.py."""
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci, si = math.cos(ai), math.sin(ai)
    cj, sj = math.cos(aj), math.sin(aj)
    ck, sk = math.cos(ak), math.sin(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    return (cj * sc - sj * cs,
            cj * ss + sj * cc,
            cj * cs - sj * sc,
            cj * cc + sj * ss)


def quaternion_multiply(q1, q0):
    """q = q1 ⊗ q0, 皆為 (x, y, z, w). 與 tf.transformations 同."""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return (
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
    )


# ★ MAVROS 修正量: quaternion_inverse([0,0,0.70710678,0.70710678])
Q_CONV_INV = (0.0, 0.0, -0.70710678, 0.70710678)


class AttitudeBridgeConfig:
    # ══════════════════════════════════════════════════════════
    #  橫向架構 v2 (2026/09/09 重寫)
    # ══════════════════════════════════════════════════════════
    #  ★★★ 關鍵認知修正:
    #  ArduPlane GUIDED 的 SET_ATTITUDE_TARGET, yaw 欄位【不是航向目標】.
    #  Attitude.cpp / calc_nav_yaw_coordinated():
    #        commanded_rudder = guided_state.forced_rpy_cd.z
    #        steering_control.rudder = constrain(commanded_rudder, ±4500)
    #  而 forced_rpy_cd.z = (飛控抽到的 yaw 度) × 100.
    #  也就是 45° = 滿舵. 舊版把 0~360 的航向塞進去 → 幾乎永遠滿舵打死.
    #
    #  新分工:
    #    SEARCH   : 不送 SET_ATTITUDE_TARGET, 改送 43002 航向指令
    #               → 飛控用 GUIDED_* PID 自己算坡度 (協調轉彎)
    #    PRE_DIVE : roll=0 + pitch, 方向舵強制歸中        mask=128
    #    DIVE     : roll=0 + 方向舵 (影像 e_u 迴路)      mask=7
    #    PULLOUT  : roll=0, 方向舵【交還飛控】           mask=132
    #    ABORT    : roll=0, 方向舵【交還飛控】           mask=132
    #    DONE     : TECS 平飛                            mask=198
    #    HOLD(非任務): roll=0 + 配平 pitch, 方向舵交還   mask=132
    # ══════════════════════════════════════════════════════════
    LATERAL_ARCH = "rudder_v2"     # "rudder_v2" | "legacy_bank" (退回舊行為)

    # ── yaw 欄位 ↔ 方向舵 的映射 ─────────────────────────────
    #  飛控抽到的 yaw = YAW_FIELD_OFFSET + YAW_FIELD_SIGN × (我們送出的 yaw)
    #
    #  ★ 2026/09/11 掛機實測確認 (bench_logs/rudder_bench_20260911_1445*):
    #    Test A: 送 180→ΔPWM 0, 176→+28, 170→+71, 160→+142, 135→+305(滿).
    #            0/45/90 三點全部卡 1900 不動; 225/270/315 全部卡 1123.
    #            → 飛控抽到的 yaw = 180 − 送出值, 一個點都不差.
    #    Test B: δr=±60 的輸出與 ±45 完全相同 → 45° 確認是滿舵.
    #            ΔPWM 與 ArduPilot 伺服映射模型誤差 ≤10 PWM (1.3%).
    #    符號:   δr=+20 實際觀察到【右偏航】→ RUDDER_SIGN = +1.0
    #    副翼:   全程 1527~1528 不動 → yaw 不會洩漏到 roll.
    YAW_FIELD_OFFSET_DEG = 180.0
    YAW_FIELD_SIGN = -1.0
    RUDDER_NEUTRAL_YAW_DEG = 180.0   # 送這個值 → 飛控抽到 0 → 方向舵歸中

    # ★★ 安全閘: False 時所有方向舵命令強制為 0 (只送 roll=0 + pitch).
    RUDDER_VERIFIED = True         # ← 2026/09/11 掛機 Test A/B/C 驗畢
    RUDDER_SIGN = +1.0             # 正 δr = 右偏航 (2026/09/11 目視確認)

    #  實測伺服端點 (SERVO4): MIN=1123 TRIM=1595 MAX=1900
    #  幾何中點 1511.5, TRIM 偏 +83.5 → 左右行程不對稱:
    #     正向 6.78 PWM/度, 負向 10.49 PWM/度 (比值 1.55)
    #  這是機身配平, 不要去動 SERVO4_TRIM. 但同一個增益往一邊修會有效 1.5 倍,
    #  實飛若出現明顯單邊偏差, 再考慮在此補一個方向相依的增益.

    # ── 方向舵迴路增益 (第一版刻意保守) ──────────────────────
    # ★ 2026/09/11: 3.0 → 6.0.
    #   ±3° 在空速排程之後 (V>26 打 0.45 折) 只剩 1.35°, 約 9~14 PWM.
    #   那樣 A/B 若 miss 沒差, 無法分辨「影像橫向控制沒用」與「有用但被
    #   限幅掐死」—— 那是論文結論的效度問題, 比安全餘裕更該優先處理.
    #   ±6° 仍只有 13.3% 的方向舵權限, 加上空速排程與 roll 保護兩層收斂.
    #  ★ 2026/09/12: 以下這組的實際值由 config.py 覆寫 (run_vision_flight
    #    啟動時 setattr 進來). 這裡的數字只是 fallback, 改參數請改 config.py.
    RUDDER_LIMIT_DEG = 6.0         # 總限幅 (度, 45=滿舵) — config.py 覆寫
    K_PSI = 0.30                   # PRE_DIVE: 航向誤差(度) → 舵(度)
    K_R = 0.15                     # PRE_DIVE: 偏航角速度阻尼 (度)/(度/s)
    K_LAT_TO_RUDDER = 0.50         # DIVE: 影像 PID 輸出 dpsi(度) → 舵(度)
    RUDDER_SLEW_DPS = 20.0         # 舵量變化率上限 (度/s)

    # 空速排程: 動壓 ∝ V², 越快權限越要小. (上限空速, 權限倍率)
    RUDDER_VSCHED = ((22.0, 1.00), (26.0, 0.70), (999.0, 0.45))
    # roll 保護: 方向舵一定會誘發滾轉, roll 太大就收手
    ROLL_PROT_FULL_DEG = 3.0       # |roll| ≤ 此值 → 全權限
    ROLL_PROT_ZERO_DEG = 8.0       # |roll| ≥ 此值 → 舵量歸零

    # ── 主迴圈失聯保護 (stale-command watchdog) ─────────────
    #  發送執行緒是獨立的 20Hz thread, 只管重送「最後一次 submit 的內容」.
    #  主控制迴圈若卡住或每拍丟例外, 它會繼續以 20Hz 重送最後一筆
    #  (例如 pitch=-28.7 + 方向舵 ±6 + 怠速). 而飛控看到的是「有人一直在送
    #  新封包」, last_forced_rpy_ms 一直被刷新 → GUIDED_TIMEOUT 救不到.
    #  ★ 這個風險是 mask 修好之後才浮現的: 以前 mask=7 讓 roll/pitch 根本
    #    無效, 重送舊命令沒後果; 現在整條打通了, 重送是真的會執行的.
    #  超過門檻就【完全停止發送】(含 43002 與 HOLD 串流), 讓飛控自己的
    #  timeout / failsafe 接手. 不能退回 HOLD 串流 —— 那還是在發
    #  SET_ATTITUDE_TARGET, 時戳一樣被刷新, 等於沒有失效保護.
    COMMAND_STALE_S = 0.4          # 主迴圈 50Hz, 0.4s = 連漏 20 拍

    # ── SEARCH: MAV_CMD_GUIDED_CHANGE_HEADING (43002) ───────
    HEADING_CMD_HZ = 4.0           # 目標在動, 要持續更新 (非 keepalive)
    HEADING_ACCEL_LIMIT = 4.0      # m/s² 向心加速度上限 → 坡度上限 ≈22°
    HEADING_TYPE = 1               # 1 = 磁航向 (0 = 地速航跡向)
    HEADING_FAIL_MAX = 5           # 連續失敗這麼多次 → 退回 legacy bank

    # ── legacy_bank 退路 (43002 不可用時) ────────────────────
    K_HDG_TO_ROLL = 1.2            # roll_cmd = K·e_ψ
    ROLL_CMD_LIMIT_DEG = 30.0

    # ── 油門排程 (HERO2180) ──────────────────────────────────
    ENABLE_SPEED_HOLD = True
    V_REF_MPS = 20.0               # 巡航目標空速
    THR_TRIM = 0.45
    K_V = 0.05
    THR_MIN = 0.05                 # ★ DIVE 怠速
    THR_MAX = 0.85
    THR_PULLOUT = 0.75             # ★ ABORT 拉起油門
    TECS_HOLD_ALT_M = 70.0         # ★ 切 GUIDED 即預送給 TECS 的保持高度
    TECS_SPEED_MPS = 18.0          # ★ TECS 目標空速 (交棒過渡油門亦以此保持)
    HOLD_PITCH_DEG = 3.5           # ★ 修改6: 非任務 HOLD 串流的 pitch (≈巡航攻角)

    # ── 命令平滑 ─────────────────────────────────────────────
    PITCH_SLEW_DPS = 40.0
    HEADING_SLEW_DPS = 60.0

    SEND_HZ = 20.0
    ROLL_TRIM_DEG = 0.0

    REQUIRED_PITCH_LIM_MIN_DEG = -40.0   # ★ 與 dive_guidance.PITCH_CMD_MIN_DEG
                                         #   及飛控 PTCH_LIM_MIN_DEG 三者一致

    # ★ MAVROS 座標修正開關 (預設開; 若未來 MAVROS 版本行為改變,
    #   掛機測試發現 pitch/yaw 又不對, 先切這裡驗證)
    APPLY_MAVROS_FRAME_FIX = True


class ArduPilotAttitudeBridge:
    """
    用法 (與原版相同):
        bridge = ArduPilotAttitudeBridge(dive_angle_deg=30.0)
        bridge.start()
        每拍: bridge.submit(sm_out, yaw_true_deg, yaw_rate_dps, airspeed_mps)
        結束: bridge.stop()
    """

    def __init__(self, dive_angle_deg=30.0, cfg: AttitudeBridgeConfig = None):
        if not ROS_OK:
            raise ImportError("rospy / mavros_msgs 不可用")
        self.cfg = cfg or AttitudeBridgeConfig()
        self.dive_angle_deg = dive_angle_deg
        if self.cfg.REQUIRED_PITCH_LIM_MIN_DEG is None:
            self.cfg.REQUIRED_PITCH_LIM_MIN_DEG = -(dive_angle_deg + 15.0)

        self._pub = rospy.Publisher("/mavros/setpoint_raw/attitude",
                                    AttitudeTarget, queue_size=1)

        self._lock = threading.Lock()
        self._active = False
        self._pitch_cmd = 0.0
        self._heading_cmd = 0.0
        self._yaw_cmd = 0.0        # 實際送出的 yaw (bank=當下航向, 其他=heading_cmd)
        self._roll_cmd = self.cfg.ROLL_TRIM_DEG
        self._thr_cmd = self.cfg.THR_TRIM
        self._mode_tag = 'IDLE'

        self._pitch_sent = 0.0
        self._roll_sent = 0.0
        self._yaw_sent = None      # legacy_bank 才用 (絕對航向)

        # ── 橫向 v2 狀態 ──────────────────────────────────────
        self._rudder_cmd_deg = 0.0     # 限幅後的舵量命令 (度; 45=滿舵)
        self._rudder_raw_deg = 0.0     # 限幅前 (診斷用)
        self._rudder_sent_deg = 0.0    # 實際送出 (經 slew)
        self._rudder_limit_deg = 0.0   # 當下生效的限幅值
        self._lim_by_v = 1.0
        self._lim_by_roll = 1.0
        self._mask_sel = 7             # 當下選用的 type_mask
        self._roll_deg_in = None       # submit 傳進來的實際 roll
        self._hdg_srv = None           # 43002 ServiceProxy (快取)
        self._hdg_last_t = 0.0
        self._hdg_fail = 0
        self._hdg_ok_cnt = 0
        self._lateral_fallback = False  # 43002 失敗 → 退回 bank
        self._warned_no_roll = False
        self._last_submit_t = None     # 最後一次 submit 的時刻 (monotonic)
        self._stale = False            # 目前是否處於 stale 停送狀態
        self.stale_events = 0          # 累計進入 stale 的次數
        self.stale_total_s = 0.0       # 累計停送時間
        self._stale_t0 = None

        self._thread = None
        self._running = False
        self.sent_count = 0
        self.last_send_t = 0.0

    # ------------------------------------------------------------------
    def check_params(self):
        """啟動檢查: PTCH_LIM_MIN_DEG 夠深? GUIDED_TIMEOUT 多少?"""
        try:
            rospy.wait_for_service("/mavros/param/get", timeout=3.0)
            get = rospy.ServiceProxy("/mavros/param/get", ParamGet)
            r = get("PTCH_LIM_MIN_DEG")
            pmin = r.value.real if r.success else None
            if pmin is not None:
                need = self.cfg.REQUIRED_PITCH_LIM_MIN_DEG
                if pmin > need + 0.5:
                    print(f"[Bridge] ★★★ 警告: PTCH_LIM_MIN_DEG={pmin:.0f}° "
                          f"比俯衝需求 {need:.0f}° 淺! 實飛前請改深.")
                else:
                    print(f"[Bridge] PTCH_LIM_MIN_DEG={pmin:.0f}° OK "
                          f"(需求 {need:.0f}°)")
            for pn in ("ARSPD_FBW_MIN", "ARSPD_FBW_MAX"):
                rr = get(pn)
                if rr.success:
                    v = rr.value.real
                    bad = ((pn.endswith("MIN") and v > self.cfg.TECS_SPEED_MPS) or
                           (pn.endswith("MAX") and v < self.cfg.TECS_SPEED_MPS))
                    print(f"[Bridge] {pn}={v:.0f}"
                          + (f" ★ TECS 目標 {self.cfg.TECS_SPEED_MPS:.0f} 超出範圍,"
                             f" CHANGE_SPEED 會被拒!" if bad else " OK"))
            # ★ 43002 (GUIDED_CHANGE_HEADING) 需要韌體編進
            #   AP_PLANE_OFFBOARD_GUIDED_SLEW_ENABLED. 判斷方式: GUIDED_P 存在.
            rg = get("GUIDED_P")
            if rg.success:
                print(f"[Bridge] GUIDED_P={rg.value.real:.3f} → 43002 航向控制"
                      f"可用 (SEARCH 走 bank turn)")
            else:
                print("[Bridge] ★★★ 找不到 GUIDED_P — 此韌體沒有 43002 航向"
                      "控制器! SEARCH 將退回 legacy bank.")
                self._lateral_fallback = True
            r2 = get("GUIDED_TIMEOUT")
            if r2.success and r2.value.real > 0:
                gtv = r2.value.real
                print(f"[Bridge] GUIDED_TIMEOUT={gtv:.1f}s"
                      + (" ★ 太長! 交棒後飛控 hold 最後 pitch 這麼久才還 TECS,"
                         " 建議 0.5" if gtv > 1.0 else " OK"))
        except Exception as e:
            print(f"[Bridge] 參數檢查跳過 ({e})")

    def set_guided(self):
        try:
            rospy.wait_for_service("/mavros/set_mode", timeout=3.0)
            sm = rospy.ServiceProxy("/mavros/set_mode", SetMode)
            resp = sm(custom_mode="GUIDED")
            print(f"[Bridge] set GUIDED: {resp.mode_sent}")
            return resp.mode_sent
        except Exception as e:
            print(f"[Bridge] set GUIDED 失敗: {e}")
            return False

    # ------------------------------------------------------------------
    def submit(self, sm_out, yaw_true_deg, yaw_rate_dps, airspeed_mps,
               roll_deg=None):
        """
        主迴圈每拍呼叫.
        yaw_true_deg 請餵 /mavros/global_position/compass_hdg (★修改3).
        roll_deg = 實際滾轉角 (度), 給 roll protection 用. 沒給就關閉保護.
        """
        cfg = self.cfg
        # ★ stale-command watchdog: 記下主迴圈最後一次餵命令的時刻
        self._last_submit_t = time.monotonic()
        self._yaw_true = yaw_true_deg          # HOLD 串流用 (修改6)
        active = bool(sm_out.get('cmd_active', False))
        pitch_cmd = float(sm_out.get('pitch_cmd_deg', 0.0))
        heading_cmd = float(sm_out.get('heading_cmd_deg', 0.0))
        mode = sm_out.get('mode', 'IDLE')
        is_abort = bool(sm_out.get('is_abort', 0))
        is_pullout = bool(sm_out.get('is_pullout', 0))
        tecs_level = bool(sm_out.get('tecs_level', 0))

        e_psi = (heading_cmd - yaw_true_deg + 180.0) % 360.0 - 180.0

        with self._lock:
            self._active = active
            self._mode_tag = mode
            self._pitch_cmd = pitch_cmd
            self._heading_cmd = heading_cmd

            self._tecs_level = tecs_level
            self._roll_deg_in = roll_deg

            # ══════════════════════════════════════════════════
            #  橫向: 依模式決定 (roll 命令, 方向舵命令, type_mask)
            # ══════════════════════════════════════════════════
            legacy = (cfg.LATERAL_ARCH != "rudder_v2") or self._lateral_fallback
            rud_raw = 0.0                 # 舵量 (度), 45=滿舵
            use_43002 = False

            if tecs_level:
                # DONE: 只鎖 roll=0, pitch/油門/yaw 全歸 TECS
                self._roll_cmd = 0.0
                self._mask_sel = self.MASK_TECS
            elif is_pullout:
                # ★★ 2026/09/12: 拉平只做姿態歸零, 油門交 TECS (mask 192).
                #   原本這裡自己算 thr = THR_TRIM + K_V×(18−V), 理由是
                #   「避免飛控沿用 DIVE 的 THR_MIN=0.05 怠速」.
                #   但 K_V=0.05, 空速 28m/s 以上算出來就是 0 (夾限), 而俯衝
                #   末段必定超過 28 —— 我們強制出來的【就是怠速】, 跟殘留值
                #   差 0.05. 這一步是多餘的環節, 交 TECS 反而少一層.
                #   (TECS 此時看到超速, 本來就會把油門收到底)
                self._roll_cmd = 0.0
                rud_raw = 0.0
                self._mask_sel = self.MASK_RPY_NOTHR
            elif is_abort:
                # ★ 緊急拉起仍然強制大油門 —— 這裡要的是【推力】, 不是收油門,
                #   不能交給看到超速就收油門的 TECS.
                self._roll_cmd = 0.0
                rud_raw = 0.0
                self._mask_sel = self.MASK_RPY
                self._thr_cmd = cfg.THR_PULLOUT
            elif legacy:
                # 退路: 舊的 bank 攔截轉彎 (43002 不可用 / 手動指定)
                self._roll_cmd = max(-cfg.ROLL_CMD_LIMIT_DEG,
                                     min(cfg.ROLL_CMD_LIMIT_DEG,
                                         cfg.K_HDG_TO_ROLL * e_psi))
                self._mask_sel = self.MASK_RP
            elif mode == 'SEARCH':
                # ★ 不送姿態, 改用 43002 讓飛控自己壓坡度轉彎.
                #   roll / pitch / 方向舵【三個都不給】, 全交飛控.
                use_43002 = True
                self._roll_cmd = 0.0
                # ★ mask 記 0 = 「本段完全不發 SET_ATTITUDE_TARGET」.
                #   舊版在這裡留 132, log 看起來像有在送姿態, 會誤導判讀
                #   (實測 60 秒的 SEARCH 只發了撥 GUIDED 前的 90 筆暖機串流).
                self._mask_sel = self.MASK_NONE
            elif mode == 'PRE_DIVE':
                # ★ 2026/09/11 使用者定案: roll=0 + pitch=俯衝姿態,
                #   yaw【交還飛控】(mask 132) — PRE_DIVE 不做橫向修正.
                #   理由: 這一段只有 2~3 秒平飛, 方向舵能修的量本來就極小,
                #   交還飛控可保留它自己的協調轉彎/側滑抑制.
                #   舊版的 δr = Kψ·e_ψ − Kr·r 保留在下面註解, 要改回來時用.
                self._roll_cmd = 0.0
                rud_raw = 0.0
                # ★ 2026/09/11 再修: 從 MASK_RP(132, 不強制方向舵) 改成
                #   MASK_RPY(128) + 方向舵明確 0.
                #   132 是「停止刷新」, 飛控會沿用上一筆 forced 值直到逾時;
                #   128+0 是「明確歸中」, 飛控記住的最後一筆就是中立值.
                #   代價: PRE_DIVE 期間飛控自己的側滑抑制被取代 (機翼水平
                #   推頭, 側滑本來就小, 可接受). 好處是 PRE_DIVE→DIVE→
                #   PULLOUT 全程 mask 128, 不再有切換與殘留的模糊地帶.
                self._mask_sel = self.MASK_RPY
                # 舊版 (橫向 rudder 迴路):
                #   rud_raw = cfg.K_PSI * e_psi - cfg.K_R * (yaw_rate_dps or 0)
                #   self._mask_sel = self.MASK_RPY
            elif mode == 'DIVE':
                # roll=0 + 方向舵追影像橫向誤差 (版本 A)
                self._roll_cmd = cfg.ROLL_TRIM_DEG
                if sm_out.get('dive_hold_heading'):
                    # ★ 版本 B (--dive-hold-heading): DIVE 不修橫向.
                    #   方向舵交還飛控, 沿進入航向直線俯衝. 縱向照修.
                    rud_raw = 0.0
                    self._mask_sel = self.MASK_RP
                else:
                    rud_raw = cfg.K_LAT_TO_RUDDER * float(
                        sm_out.get('pid_dpsi_deg', 0.0) or 0.0)
                    self._mask_sel = self.MASK_RPY
            else:
                self._roll_cmd = 0.0
                self._mask_sel = self.MASK_RP

            self._use_43002 = use_43002
            self._rudder_raw_deg = rud_raw
            self._rudder_cmd_deg = self._limit_rudder(rud_raw, airspeed_mps,
                                                      roll_deg)
            self._yaw_cmd = heading_cmd     # 只給 log / legacy 用

            # ── 油門排程 (★修改2) ──
            if is_abort or is_pullout:
                pass                          # 上面已設 THR_PULLOUT
            elif mode == 'DIVE':
                self._thr_cmd = cfg.THR_MIN   # 怠速
            elif cfg.ENABLE_SPEED_HOLD and airspeed_mps is not None:
                thr = cfg.THR_TRIM + cfg.K_V * (cfg.V_REF_MPS - airspeed_mps)
                self._thr_cmd = max(cfg.THR_MIN, min(cfg.THR_MAX, thr))
            else:
                self._thr_cmd = cfg.THR_TRIM

    # ══════════════════════════════════════════════════════════════
    #  ArduPlane type_mask (XOR 語意, 見 GCS_MAVLink_Plane.cpp)
    #    收到後飛控做 type_mask ^= 0xFF, 再取 & 0b10000111 判哪幾軸強制.
    # ══════════════════════════════════════════════════════════════
    # ★★ 2026/09/11 掛機實測 (19:19 / 19:28 兩趟對照):
    #    mask=7   送 pitch -28.7 → 飛控 nav_pitch 毫無反應 (升降舵不動)
    #    mask=132 送 pitch -28.7 → 飛控 nav_pitch = -28.7, 穩定 13 秒
    #    差別只在 IGNORE_ATTITUDE (bit7). 照 ArduPlane 原始碼的 XOR 邏輯
    #    兩者都該生效, 但實測就是只有設了 bit7 的才行. 原因未明, 以實測為準.
    #    所以「全強制」改用 128 (= 132 保留 bit7, 再把 yaw 也強制):
    #      128 → attitude_mask 0b00000111 → roll+pitch+yaw+油門 全強制
    MASK_RPY = 128    # roll+pitch+yaw+油門 全強制 (DIVE 版本A)
    MASK_RPY_NOTHR = 192  # roll+pitch+yaw 強制, 油門交 TECS (PULLOUT)
                      #   = 128 + 64(IGNORE_THRUST), 保留必要的 bit7
    MASK_RPY_OLD7 = 7  # ★ 舊值, 實測 pitch/roll 無效 — 不要用
    MASK_RP = 132     # roll+pitch+油門 強制, yaw 交還飛控 (HOLD/PULLOUT/ABORT)
    MASK_P = 133      # 只強制 pitch+油門 (roll/yaw 都交還; 保留備用)
    MASK_TECS = 198   # 只強制 roll, pitch/yaw/油門 歸 TECS (DONE)
    MASK_NONE = 0     # ★ 不是真的 mask: log 用的標記, 代表這段完全不送姿態
                      #   (SEARCH 只送 43002, roll/pitch/方向舵全交飛控)

    def _limit_rudder(self, raw_deg, airspeed_mps, roll_deg):
        """總限幅 → 空速排程 → roll 保護 → 安全閘. 回傳實際可用舵量 (度)."""
        cfg = self.cfg
        # 空速排程 (動壓 ∝ V², 越快權限越小)
        k_v = 1.0
        if airspeed_mps is not None:
            for v_hi, k in cfg.RUDDER_VSCHED:
                if airspeed_mps <= v_hi:
                    k_v = k
                    break
        # roll 保護 (方向舵誘發滾轉; roll 越大越收手)
        k_roll = 1.0
        if roll_deg is None:
            if not self._warned_no_roll:
                self._warned_no_roll = True
                print("[Bridge] ⚠ submit() 未提供 roll_deg → roll protection 停用")
        else:
            a = abs(roll_deg)
            if a >= cfg.ROLL_PROT_ZERO_DEG:
                k_roll = 0.0
            elif a > cfg.ROLL_PROT_FULL_DEG:
                k_roll = 1.0 - (a - cfg.ROLL_PROT_FULL_DEG) / \
                    max(1e-6, cfg.ROLL_PROT_ZERO_DEG - cfg.ROLL_PROT_FULL_DEG)
        self._lim_by_v = k_v
        self._lim_by_roll = k_roll
        lim = cfg.RUDDER_LIMIT_DEG * k_v * k_roll
        self._rudder_limit_deg = lim
        out = max(-lim, min(lim, cfg.RUDDER_SIGN * raw_deg))
        # ★★ 安全閘: 掛機沒驗過就不准打舵
        if not cfg.RUDDER_VERIFIED:
            out = 0.0
        return out

    def _rudder_to_yaw_field(self, rudder_deg):
        """舵量(度) → 要塞進 quaternion yaw 欄位的值(度)."""
        cfg = self.cfg
        if not cfg.RUDDER_VERIFIED:
            return cfg.RUDDER_NEUTRAL_YAW_DEG
        # 飛控抽到 = OFFSET + SIGN × 送出  ⇒  送出 = (想要 − OFFSET) / SIGN
        return ((rudder_deg - cfg.YAW_FIELD_OFFSET_DEG)
                / cfg.YAW_FIELD_SIGN) % 360.0

    # ------------------------------------------------------------------
    MAV_CMD_GUIDED_CHANGE_HEADING = 43002

    def send_heading(self, heading_deg):
        """送 43002 給飛控, 讓它用 GUIDED_* PID 自己算坡度轉彎.

        param1 = 航向型態 (1=磁航向), param2 = 航向(度),
        param3 = 最大向心加速度 (m/s²) → 決定坡度上限.
        """
        cfg = self.cfg
        try:
            if self._hdg_srv is None:
                rospy.wait_for_service('/mavros/cmd/command_int', timeout=1.0)
                self._hdg_srv = rospy.ServiceProxy(
                    '/mavros/cmd/command_int', CommandInt, persistent=True)
            r = self._hdg_srv(broadcast=False, frame=0,
                              command=self.MAV_CMD_GUIDED_CHANGE_HEADING,
                              current=0, autocontinue=0,
                              param1=float(cfg.HEADING_TYPE),
                              param2=float(heading_deg % 360.0),
                              param3=float(cfg.HEADING_ACCEL_LIMIT),
                              param4=0.0, x=0, y=0, z=0.0)
            ok = bool(r.success)
        except Exception as e:
            self._hdg_srv = None
            print(f"[Bridge] 43002 例外: {e}")
            ok = False

        if ok:
            self._hdg_fail = 0
            self._hdg_ok_cnt += 1
        else:
            self._hdg_fail += 1
            if self._hdg_fail == 1:
                print("[Bridge] ★ 43002 (GUIDED_CHANGE_HEADING) 被拒 — "
                      "檢查韌體有無 GUIDED_P 參數")
            if self._hdg_fail >= cfg.HEADING_FAIL_MAX and not self._lateral_fallback:
                self._lateral_fallback = True
                print("[Bridge] ★★★ 43002 連續失敗 → SEARCH 橫向退回 "
                      "legacy bank (roll = K·e_ψ)")
        return ok

    # ------------------------------------------------------------------
    def _slew(self, target, current, rate_dps, dt):
        step = rate_dps * dt
        if target > current + step:
            return current + step
        if target < current - step:
            return current - step
        return target

    def _slew_wrap(self, target, current, rate_dps, dt):
        """航向平滑 (跨 ±180 走短邊)."""
        err = (target - current + 180.0) % 360.0 - 180.0
        step = rate_dps * dt
        err = max(-step, min(step, err))
        return (current + err) % 360.0

    def _send_loop(self):
        cfg = self.cfg
        rate = rospy.Rate(cfg.SEND_HZ)
        dt = 1.0 / cfg.SEND_HZ
        while self._running and not rospy.is_shutdown():
            with self._lock:
                active = self._active
                pitch = self._pitch_cmd
                heading_tgt = self._heading_cmd
                heading_legacy = getattr(self, '_yaw_cmd', self._heading_cmd)
                roll = self._roll_cmd
                thr = self._thr_cmd
                tecs_lv = getattr(self, '_tecs_level', False)
                mask = self._mask_sel
                rud_cmd = self._rudder_cmd_deg
                use_43002 = getattr(self, '_use_43002', False)
                legacy = ((cfg.LATERAL_ARCH != "rudder_v2")
                          or self._lateral_fallback)

            # ══ stale-command watchdog ══════════════════════════
            #  主迴圈太久沒餵新命令 → 完全停止發送 (含 43002 與 HOLD 串流),
            #  讓飛控自己的 GUIDED_TIMEOUT / failsafe 接手.
            #  ★ 不能退回 HOLD 串流 —— 那還是在發 SET_ATTITUDE_TARGET,
            #    飛控的 last_forced_rpy_ms 一樣會被刷新, 等於沒有失效保護.
            _age = (time.monotonic() - self._last_submit_t
                    if self._last_submit_t is not None else 0.0)
            if self._last_submit_t is not None and _age > cfg.COMMAND_STALE_S:
                if not self._stale:
                    self._stale = True
                    self._stale_t0 = time.monotonic()
                    self.stale_events += 1
                    print(f"\n[Bridge] ★★★ 主迴圈已 {_age:.2f}s 沒有更新命令 "
                          f"— 停止發送, 交還飛控 (第 {self.stale_events} 次)")
                rate.sleep()
                continue
            if self._stale:
                self._stale = False
                if self._stale_t0 is not None:
                    _d = time.monotonic() - self._stale_t0
                    self.stale_total_s += _d
                    print(f"[Bridge] 主迴圈恢復, 停送了 {_d:.2f}s")
                self._stale_t0 = None

            # ── SEARCH: 只送 43002, 完全不送 SET_ATTITUDE_TARGET ──
            #    (只要送了姿態, forced roll 就會蓋掉 43002 的航向控制器)
            if active and use_43002 and not legacy:
                now = time.time()
                if now - self._hdg_last_t >= 1.0 / max(1e-3, cfg.HEADING_CMD_HZ):
                    self._hdg_last_t = now
                    self.send_heading(heading_tgt)
                rate.sleep()
                continue

            if active:
                self._pitch_sent = self._slew(pitch, self._pitch_sent,
                                              cfg.PITCH_SLEW_DPS, dt)
                self._roll_sent = self._slew(roll, self._roll_sent,
                                             cfg.HEADING_SLEW_DPS, dt)
                if legacy:
                    # 舊路徑: yaw 欄位當絕對航向 (★已知是錯的, 只作退路)
                    if self._yaw_sent is None:
                        self._yaw_sent = heading_legacy
                    self._yaw_sent = self._slew_wrap(heading_legacy,
                                                     self._yaw_sent,
                                                     cfg.HEADING_SLEW_DPS, dt)
                    yaw_out = self._yaw_sent
                else:
                    # v2: yaw 欄位 = 方向舵. 用一般 rate limiter (不可 wrap!)
                    self._rudder_sent_deg = self._slew(
                        rud_cmd, self._rudder_sent_deg,
                        cfg.RUDDER_SLEW_DPS, dt)
                    yaw_out = self._rudder_to_yaw_field(self._rudder_sent_deg)

                # NED 慣例組 quaternion
                # ★★ 修改 5 v2 (2026/08/26 兩輪掛機定案) + v3 (2026/09/09):
                #  - roll/pitch: Q_CONV_INV 路徑 + pitch 取負, 掛機實證.
                #  - yaw: ★ 不是航向, 是方向舵命令 (見 AttitudeBridgeConfig).
                #    45° = 滿舵. 舊註解說的「176→回中」其實正是這件事的證據:
                #    180−176=4° → 舵量 400/4500 ≈ 回中, 不是航向編碼正確.
                self._publish_att(self._roll_sent, self._pitch_sent, yaw_out,
                                  thr, tecs_lv, mask)
            else:
                # ★ 修改 6 (2026/08/27 實飛): 非任務期間也持續串流「保持現況」
                # setpoint. GUIDED 以外飛控會忽略, 完全無害; 但撥入 GUIDED 的
                # 瞬間 forced attitude 已經是新鮮的 → 堵掉 ArduPlane GUIDED
                # 進入暫態 (~1.2s loiter 甩頭, F1 miss 125m 的根因).
                # ★ v3: yaw 改送中立值 + mask 132 (方向舵交還飛控).
                #   舊版送 yaw=當下機頭(0~360) → 撥 GUIDED 瞬間滿舵打死.
                self._pitch_sent = self._slew(cfg.HOLD_PITCH_DEG,
                                              self._pitch_sent,
                                              cfg.PITCH_SLEW_DPS, dt)
                self._roll_sent = 0.0
                self._rudder_sent_deg = 0.0
                self._publish_att(0.0, self._pitch_sent,
                                  cfg.RUDDER_NEUTRAL_YAW_DEG,
                                  cfg.THR_TRIM, False, self.MASK_RP)
            rate.sleep()

    def _publish_att(self, roll_deg, pitch_deg, yaw_deg, thr, tecs_lv,
                     mask=None):
        """組 quaternion (修改5 v2 編碼) 並發布. active 與 HOLD 串流共用."""
        cfg = self.cfg
        q_ned = euler_to_quaternion(
            math.radians(roll_deg),
            -math.radians(pitch_deg),          # ★ pitch 取負 (修改5 v2)
            math.radians(yaw_deg))
        if cfg.APPLY_MAVROS_FRAME_FIX:
            qx, qy, qz, qw = quaternion_multiply(Q_CONV_INV, q_ned)
        else:
            qx, qy, qz, qw = q_ned
        msg = AttitudeTarget()
        msg.header.stamp = rospy.Time.now()
        if mask is None:
            mask = self.MASK_TECS if tecs_lv else self.MASK_RPY
        if mask == self.MASK_TECS:
            thr = 0.0
        msg.type_mask = int(mask)
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        msg.thrust = thr
        try:
            self._pub.publish(msg)
        except Exception:
            # ★ 2026/09/11: Ctrl+C 時 rospy 會先關掉 publisher, 這裡會丟
            #   ROSException("publish() to a closed topic") 把發送執行緒炸掉
            #   (實測 Thread-37 traceback). 關機途中失敗無所謂, 安靜收工即可.
            self._running = False
            return
        self.sent_count += 1
        self.last_send_t = time.time()

    MAV_CMD_GUIDED_CHANGE_SPEED = 43000
    MAV_CMD_GUIDED_CHANGE_ALTITUDE = 43001
    MAV_FRAME_GLOBAL_RELATIVE_ALT = 3

    def send_tecs_targets(self, alt_m=None, speed_mps=None, tag=''):
        """送高度+空速目標給飛控 TECS (GUIDED 時有效).
        result=2 DENIED 多半是空速超出 ARSPD_FBW_MIN/MAX.

        ★★ 2026/09/11 修正 (嚴重回歸):
        舊版寫死 cfg.TECS_HOLD_ALT_M (=70m) 並在【撥 GUIDED 當下】就送出.
        舊架構下 SEARCH 自己壓 pitch 定高 (_search_agl_ref), TECS 的高度目標
        無所謂. 但 SEARCH 改成「不送姿態、pitch/高度全交 TECS」之後,
        這個 70m 就會【立刻把飛機從巡航 120m 往下帶】, 到 PRE_DIVE 點時高度
        全錯, 俯衝幾何整個垮掉.
        現在改成呼叫端指定:
          撥 GUIDED  → 送「當下高度」(維持巡航高度逼近)
          進 PULLOUT → 才送 TECS_HOLD_ALT_M (70m)
        """
        cfg = self.cfg
        if alt_m is None:
            alt_m = cfg.TECS_HOLD_ALT_M
        if speed_mps is None:
            speed_mps = cfg.TECS_SPEED_MPS

        def _cmd(command, frame=0, p1=0.0, p2=0.0, z=0.0):
            try:
                rospy.wait_for_service('/mavros/cmd/command_int', timeout=1.0)
                srv = rospy.ServiceProxy('/mavros/cmd/command_int', CommandInt)
                r = srv(broadcast=False, frame=frame, command=command,
                        current=0, autocontinue=0, param1=p1, param2=p2,
                        param3=0.0, param4=0.0, x=0, y=0, z=z)
                # 舊版 mavros CommandInt 回應只有 success 沒有 result
                # (CommandLong 才有) — 2026/08/26 掛機抓到, getattr 兼容
                return bool(r.success), int(getattr(r, 'result', 0))
            except Exception as e:
                print(f"[Bridge] command_int {command} 失敗: {e}")
                return False, -1
        ok_a, ra = _cmd(self.MAV_CMD_GUIDED_CHANGE_ALTITUDE,
                        frame=self.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                        z=float(alt_m))
        ok_s, rs = _cmd(self.MAV_CMD_GUIDED_CHANGE_SPEED,
                        p1=0.0, p2=float(speed_mps))
        print(f"[Bridge][TECS目標{tag}] ALT {alt_m:.0f}m ok={ok_a} "
              f"result={ra} | SPEED {speed_mps:.0f}m/s ok={ok_s} "
              f"result={rs}"
              + ("" if ok_s else " (result=2: 檢查 ARSPD_FBW_MIN/MAX)"))
        return ok_a, ok_s

    def start(self):
        self.check_params()
        self._running = True
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()
        c = self.cfg
        print(f"[Bridge] 發送執行緒啟動 @ {c.SEND_HZ:.0f}Hz, "
              f"lateral={c.LATERAL_ARCH}"
              + (" (已退回 legacy bank)" if self._lateral_fallback else "")
              + f", MAVROS frame fix="
              f"{'ON' if c.APPLY_MAVROS_FRAME_FIX else 'OFF'}")
        if c.LATERAL_ARCH == "rudder_v2":
            if c.RUDDER_VERIFIED:
                print(f"[Bridge] 方向舵迴路 ON — 限幅 ±{c.RUDDER_LIMIT_DEG:.1f}°, "
                      f"Kψ={c.K_PSI} Kr={c.K_R} K_lat={c.K_LAT_TO_RUDDER}, "
                      f"sign={c.RUDDER_SIGN:+.0f}")
            else:
                print("[Bridge] ★★★ RUDDER_VERIFIED=False → 方向舵命令一律 0.")
                print("[Bridge]     請先跑 test_rudder_bench.py 的 Test A/B, "
                      "確認映射與符號後再開.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        _st = (f", stale {self.stale_events} 次/共 {self.stale_total_s:.1f}s"
               if self.stale_events else "")
        print(f"[Bridge] 停止, 共發送 {self.sent_count} 筆姿態命令{_st}")

    def get_debug(self):
        with self._lock:
            # ★ brg_thr 誠實反映「實際發布給飛控」的油門:
            #   active=True  → 任務油門排程 (_thr_cmd)
            #   active=False → HOLD 串流實際發布的是 THR_TRIM (見送出迴圈 HOLD 分支),
            #                  非 _thr_cmd; 否則 log 會誤顯示上一次任務油門 (如 0.85).
            thr_report = self._thr_cmd if self._active else self.cfg.THR_TRIM
            # brg_roll 同理: HOLD 串流實際發布 roll=0, 非 _roll_cmd.
            roll_report = self._roll_cmd if self._active else 0.0
            return {
                'brg_active': int(self._active),
                'brg_mode': self._mode_tag,
                'brg_pitch_sent': round(self._pitch_sent, 2),
                'brg_yaw_sent': (round(self._yaw_sent, 1)
                                 if self._yaw_sent is not None else ''),
                'brg_roll': round(roll_report, 2),
                'brg_thr': round(thr_report, 3),
                'brg_sent_cnt': self.sent_count,
                # ── 橫向 v2 取證 ──
                'brg_mask': self._mask_sel,
                'rudder_raw_deg': round(self._rudder_raw_deg, 3),
                'rudder_cmd_deg': round(self._rudder_cmd_deg, 3),
                'rudder_sent_deg': round(self._rudder_sent_deg, 3),
                'rudder_limit_deg': round(self._rudder_limit_deg, 3),
                'lim_by_airspeed': round(self._lim_by_v, 2),
                'lim_by_roll': round(self._lim_by_roll, 2),
                'hdg_cmd_ok_cnt': self._hdg_ok_cnt,
                'lateral_fallback': int(self._lateral_fallback),
                'brg_stale': int(self._stale),
                'brg_stale_events': self.stale_events,
            }
