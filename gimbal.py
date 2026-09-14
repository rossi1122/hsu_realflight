# -*- coding: utf-8 -*-
"""
gimbal.py — SIYI A8 mini 雲台控制 (vision_dive)
═══════════════════════════════════════════════════════════════════
職責 (不動 siyi_sdk / siyi_message, 只呼叫):
  1. 連線 + 開機固定 2x zoom
  2. STANDBY (未 GUIDED): FOLLOW 模式, yaw=0 跟機頭, 低頭看前方
     → 對準後撥 GUIDED, 目標就在畫面中央
  3. SEARCH/PRE_DIVE: FOLLOW + 分區 P 追蹤 (bbox → 角度誤差 → setAngles)
  4. DIVE: 切 FPV, 雲台韌體貼齊機身 = 真 body-fixed, 不再送 setAngles

★ A8 mini 觀念: setAngles pitch 相對【水平面(重力)】不是機身. FOLLOW 下
  飛機壓頭時雲台反向補償保持俯角 (PRE_DIVE 免費 reverse compensation).
  DIVE 要 body-fixed 只能切 FPV (實測可行).

追蹤器 (SimpleTracker) 與角度誤差公式逐字沿用 test_gimbal_predive_dive.py,
確保雲台行為與已驗證的版本一致.
"""

import math
import time
import logging

import numpy as np

from siyi_sdk import SIYISDK


# ──────────────────────────────────────────────────────────────────
#  bbox → 雲台角度誤差 (SIYI 慣例: yaw+ 往左, pitch+ 往上)
#  沿用 test_gimbal_predive_dive.bbox_to_angle_error
# ──────────────────────────────────────────────────────────────────
def bbox_to_angle_error(cx, cy, w, h, fx, fy):
    dx = cx - w / 2.0
    dy = cy - h / 2.0
    return (-math.degrees(math.atan2(dx, fx)),   # 畫面右 → yaw 誤差負 (往右轉)
            -math.degrees(math.atan2(dy, fy)))   # 畫面下 → pitch 誤差負 (往下轉)


# ──────────────────────────────────────────────────────────────────
#  SimpleTracker — 分區 P 增量追蹤 (沿用 test_gimbal 邏輯)
# ──────────────────────────────────────────────────────────────────
class SimpleTracker:
    ZONE_CENTER, ZONE_NORMAL = 0.25, 0.60
    KP_CENTER, KP_NORMAL, KP_FAST = 0.0, 0.4, 0.8
    MAX_YAW_ERR, MAX_PITCH_ERR = 30.0, 20.0
    DEAD_ZONE_DEG = 0.5
    MAX_STEP_DEG = 30.0
    YAW_LIMIT = (-135.0, 135.0)
    PITCH_LIMIT = (-90.0, 10.0)   # 上限 -10 起: 防對天空 false positive 死循環

    def __init__(self, default_yaw, default_pitch, lost_timeout=5.0):
        self.default_yaw = default_yaw
        self.default_pitch = default_pitch
        self.lost_timeout = lost_timeout
        self.reset()

    def reset(self):
        self.last_seen = None
        self.target_yaw = self.default_yaw
        self.target_pitch = self.default_pitch
        self.state = "IDLE"

    @classmethod
    def _kp(cls, yaw_err, pitch_err):
        rel = min(max(abs(yaw_err) / cls.MAX_YAW_ERR,
                      abs(pitch_err) / cls.MAX_PITCH_ERR), 1.0)
        if rel < cls.ZONE_CENTER:
            return "CENTER", cls.KP_CENTER
        if rel < cls.ZONE_NORMAL:
            return "NORMAL", cls.KP_NORMAL
        return "FAST", cls.KP_FAST

    def update(self, has_target, yaw_err, pitch_err, cur_yaw, cur_pitch,
               lost_point=None):
        """回傳 (target_yaw, target_pitch, should_send, zone).
        lost_point: 丟失時要指向的 (yaw, pitch) — STANDBY 指向先驗用,
                    None 則丟失回 default. """
        now = time.time()
        if not has_target:
            fallback = lost_point if lost_point is not None \
                else (self.default_yaw, self.default_pitch)
            if self.last_seen is None:
                self.state = "IDLE"
                return fallback[0], fallback[1], False, "-"
            if now - self.last_seen >= self.lost_timeout:
                if self.state != "IDLE":
                    self.state = "IDLE"
                    self.target_yaw, self.target_pitch = fallback
                    return fallback[0], fallback[1], True, "-"
                return fallback[0], fallback[1], False, "-"
            self.state = "LOST"
            return self.target_yaw, self.target_pitch, False, "-"

        self.last_seen = now
        self.state = "TRACKING"
        if abs(yaw_err) < self.DEAD_ZONE_DEG:
            yaw_err = 0.0
        if abs(pitch_err) < self.DEAD_ZONE_DEG:
            pitch_err = 0.0
        if yaw_err == 0.0 and pitch_err == 0.0:
            return self.target_yaw, self.target_pitch, False, "CENTER"
        zone, kp = self._kp(yaw_err, pitch_err)
        if kp == 0.0:
            return self.target_yaw, self.target_pitch, False, zone
        yaw_err = float(np.clip(yaw_err, -self.MAX_STEP_DEG, self.MAX_STEP_DEG))
        pitch_err = float(np.clip(pitch_err, -self.MAX_STEP_DEG, self.MAX_STEP_DEG))
        self.target_yaw = float(np.clip(cur_yaw + kp * yaw_err, *self.YAW_LIMIT))
        self.target_pitch = float(np.clip(cur_pitch + kp * pitch_err, *self.PITCH_LIMIT))
        return self.target_yaw, self.target_pitch, True, zone


# ──────────────────────────────────────────────────────────────────
#  雲台連線 (沿用 test_gimbal 的 new-object-per-retry, 繞 SDK socket bug)
# ──────────────────────────────────────────────────────────────────
def _kill(cam):
    if cam is None:
        return
    try:
        logging.getLogger("SIYISDK").setLevel(logging.CRITICAL)
        cam._stop = True
        time.sleep(0.15)
        cam._socket.close()
    except Exception:
        pass


def connect_gimbal(ip, port, retries=3):
    for i in range(1, retries + 1):
        cam = SIYISDK(server_ip=ip, port=port, debug=False)
        try:
            if cam.connect():
                return cam
        except Exception as e:
            print(f"[Gimbal] 嘗試 {i} 例外: {e}")
        _kill(cam)
        print(f"[Gimbal] 嘗試 {i}/{retries} 失敗")
        time.sleep(1.0)
    return None


def set_gimbal_mode(cam, mode, repeat=2):
    """切 SIYI 模式 'lock'/'follow'/'fpv' (siyi_message builder 直送)."""
    builder = {
        'lock': cam._out_msg.lockModeMsg,
        'follow': cam._out_msg.followModeMsg,
        'fpv': cam._out_msg.fpvModeMsg,
    }.get(mode)
    if builder is None:
        return False
    ok = False
    for _ in range(repeat):
        ok = cam.sendMsg(builder()) or ok
        time.sleep(0.05)
    return ok


# ──────────────────────────────────────────────────────────────────
#  GimbalManager — 對主程式的單一介面
# ──────────────────────────────────────────────────────────────────
class GimbalManager:
    """
    對主程式暴露:
        connect() / standby_tick() / track_tick(...) / enter_fpv()
        get_attitude() → (yaw, pitch, roll) SIYI 原始度
    模式旗標由主程式的任務狀態驅動, 這裡只管雲台動作.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.cam = None
        self.mode = 'follow'          # 目前 SIYI 模式
        self._last_cmd_t = 0.0
        self.tracker = SimpleTracker(
            default_yaw=0.0,
            default_pitch=cfg.GIMBAL_STANDBY_PITCH_DEG,
            lost_timeout=cfg.GIMBAL_LOST_TIMEOUT_S)
        self.fpv_engaged = False

    # -- 連線 + 2x --
    def connect(self):
        c = self.cfg
        print(f"[Gimbal] 連線 {c.SIYI_IP}:{c.SIYI_PORT} ...")
        self.cam = connect_gimbal(c.SIYI_IP, c.SIYI_PORT)
        if self.cam is None:
            return False
        set_gimbal_mode(self.cam, 'follow')
        self.mode = 'follow'
        time.sleep(0.3)
        # 開機固定 zoom
        try:
            self.cam.requestAbsoluteZoom(float(c.ZOOM_LEVEL))
            print(f"[Gimbal] zoom → {c.ZOOM_LEVEL}x")
        except Exception as e:
            print(f"[Gimbal] zoom 命令失敗 (繼續): {e}")
        time.sleep(0.5)
        self.cam.requestSetAngles(0.0, c.GIMBAL_STANDBY_PITCH_DEG)
        time.sleep(1.0)
        print("[Gimbal] 連線成功, 進 STANDBY (FOLLOW, 看前方)")
        return True

    def get_attitude(self):
        try:
            y, p, r = self.cam.getAttitude()
            return float(y), float(p), float(r)
        except Exception:
            return 0.0, self.cfg.GIMBAL_STANDBY_PITCH_DEG, 0.0

    # -- STANDBY: 未 GUIDED, 只 hold 前方 (FOLLOW yaw 跟機頭, 不追目標) --
    def standby_tick(self):
        if self.mode != 'follow':
            set_gimbal_mode(self.cam, 'follow')
            self.mode = 'follow'
        now = time.time()
        if now - self._last_cmd_t >= 1.0 / self.cfg.GIMBAL_STANDBY_HOLD_HZ:
            try:
                self.cam.requestSetAngles(0.0, self.cfg.GIMBAL_STANDBY_PITCH_DEG)
                self._last_cmd_t = now
            except Exception:
                pass

    # -- SEARCH/PRE_DIVE: FOLLOW + 分區 P 追蹤鎖定的 bbox --
    def track_tick(self, has_target, bbox_u, bbox_v, img_w, img_h, fx, fy,
                   lost_point=None):
        """has_target=True 時用 (bbox_u,bbox_v) 追; False 時 LOST 保持/回指向."""
        if self.mode != 'follow':
            set_gimbal_mode(self.cam, 'follow')
            self.mode = 'follow'
        cur_y, cur_p, _ = self.get_attitude()
        ye = pe = 0.0
        if has_target and bbox_u is not None:
            ye, pe = bbox_to_angle_error(bbox_u, bbox_v, img_w, img_h, fx, fy)
        ty, tp, send, zone = self.tracker.update(
            has_target, ye, pe, cur_y, cur_p, lost_point=lost_point)
        now = time.time()
        if send and (now - self._last_cmd_t) >= 1.0 / self.cfg.GIMBAL_TRACK_CMD_HZ:
            try:
                self.cam.requestSetAngles(ty, tp)
                self._last_cmd_t = now
            except Exception:
                pass
        return zone

    # -- DIVE: 切 FPV (body-fixed), 之後不再送任何角度命令 --
    def enter_fpv(self):
        if not self.fpv_engaged:
            set_gimbal_mode(self.cam, 'fpv')
            self.mode = 'fpv'
            self.fpv_engaged = True
            print("[Gimbal] → FPV (body-fixed), 不再送 setAngles")

    def close(self):
        try:
            if self.cam is not None:
                self.cam.disconnect()
        except Exception:
            pass
