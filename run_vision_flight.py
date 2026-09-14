#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_vision_flight.py — vision_dive 實飛主程式 (虛擬目標版)
═══════════════════════════════════════════════════════════════════
這一版: 控制迴路吃「虛擬目標」(已知 GPS) 投影的影像誤差; 真實相機 + YOLO
照跑但只驅動雲台 + 錄影 + 記 log, 不進飛機 PID. 飛完拿 raw 影片離線重跑
YOLO, 用同一條飛行姿態序列重放 PID, 跟虛擬目標的命令逐幀比較 (見
postprocess_compare.py).

三條執行緒:
  1. 視覺線程 (~30Hz): FrameGrabber → YOLO → 雲台追蹤 (真實 bbox)
                       → 寫標註影片 + raw 影片 + 視覺 log (每幀釘飛機狀態)
                       ★ YOLO 影像誤差不進飛機 PID
  2. 控制線程 (50Hz):  MAVROS + 虛擬目標投影 → MissionStateMachine (影像 PID)
                       → AttitudeBridge → SET_ATTITUDE_TARGET + 寫主 log
  3. 主線程:           連線 / GUIDED 上升沿種點 / 收尾

目標與 PRE_DIVE 點 GPS 在命令列或 config 給定 (見 --target / --predive).
撥 GUIDED = 任務開始; 水平到達 PRE_DIVE 點 → 進 PRE_DIVE; 其餘同前.

用法:
  python3 run_vision_flight.py --target 23.001234,120.005678 \
                               --predive 23.002000,120.004000
  python3 run_vision_flight.py --no-video      # 不錄影
  python3 run_vision_flight.py --bench         # 掛機: 忽略 AGL 進入條件
"""

import argparse
import csv
import math
import os
import sys
import threading
import traceback
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dive_guidance import GuidanceConfig, MissionStateMachine
from attitude_bridge import ArduPilotAttitudeBridge, AttitudeBridgeConfig
from vision import VisionFrontend, FrameGrabber
from video_writer import VideoWriter
from virtual_source import VirtualSource
from virtual_target_geom import gps_to_ned, extrapolate_impact, miss_distance

try:
    import rospy
    from mavros_msgs.msg import (State, RCOut, AttitudeTarget,
                             NavControllerOutput)
    from sensor_msgs.msg import NavSatFix, Imu, BatteryState, TimeReference
    from std_msgs.msg import Float64
    from geometry_msgs.msg import TwistStamped
    from mavros_msgs.msg import VFR_HUD, HomePosition
    ROS_OK = True
except ImportError:
    ROS_OK = False


LOOP_HZ = 50.0

# ── 主 log 欄位 (50Hz 控制迴路, 逐拍): 論文畫圖 + 後處理對齊 ──
# virt_* = 虛擬目標算的 (實際餵 PID); real_* 在視覺 log, 不在這.
CSV_FIELDS = [
    't', 'gps_unix', 'frame_idx', 'mode', 'guided',
    # 飛機狀態 (NED)
    'ac_n', 'ac_e', 'agl', 'V', 'chi_deg', 'gamma_deg', 'theta_deg',
    'roll_deg', 'yaw_deg', 'vn', 've', 'vd', 'dt',
    # 虛擬目標 (控制路徑真相來源)
    'tgt_n', 'tgt_e', 'slant_m', 'ground_m', 'predive_dist_m',
    'virt_bbox_u', 'virt_bbox_v', 'virt_detected',
    # PID setpoint + 命令 (比較基準)
    'virt_u_sp', 'virt_v_sp', 'pitch_cmd', 'heading_cmd', 'throttle',
    # 影像 PID 各項 (DIVE 段)
    'pid_u_err_deg', 'pid_v_err_deg',
    'pid_lat_p', 'pid_lat_i', 'pid_lat_d',
    'pid_lon_p', 'pid_lon_i', 'pid_lon_d',
    'pid_dpsi_deg', 'pid_dtheta_deg', 'pid_compute_ms',
    # 橫向 v2 (方向舵迴路) 取證
    'brg_mask', 'rudder_raw_deg', 'rudder_cmd_deg', 'rudder_sent_deg',
    'rudder_limit_deg', 'lim_by_airspeed', 'lim_by_roll',
    'hdg_cmd_ok_cnt', 'lateral_fallback', 'brg_stale', 'brg_stale_events',
    # 影像誤差百分比 (論文圖)
    'img_err_u_pct', 'img_err_v_pct', 'glide_slope_err_deg',
    # ★ 2026/09/11: 逐幀外插彈著點 → miss distance.
    #   原本只在 PULLOUT 那一刻算一次並印進 events.txt, 畫不出收斂曲線,
    #   四個 case 也難比較. 現在每一拍都算, 直接畫 miss vs 剩餘高度.
    #   只有在下降 (vd > 0.1) 時才有值, 平飛/爬升時為空.
    'miss_pred_m', 'miss_pred_lon_m', 'miss_pred_lat_m', 'impact_t_fall_s',
    # 飛控取證
    'p_dps', 'q_dps', 'r_dps', 'nz_g',
    'fc_pitch_tgt_deg', 'fc_nav_pitch_deg', 'fc_nav_roll_deg',
    'fc_nav_bearing_deg', 'fc_mode', 'batt_v',
    'rcout1', 'rcout2', 'rcout3', 'rcout4',
    'lat', 'lon',
]

# ── 視覺 log 欄位 (每處理幀一列, 跟 raw 影片 1:1; 後處理重放 PID 用) ──
VISION_FIELDS = [
    # ★ raw_frame_idx = FrameGrabber 的相機幀序 = raw 影片的幀號.
    #   raw 錄影已與 YOLO 解耦, 兩者不再 1:1, 後處理請用這個對齊.
    'frame_idx', 'raw_frame_idx', 't', 'gps_unix', 'mode', 'dive_mode',
    # 真實 YOLO / DKF (機上即時; 交叉驗證 raw 影片離線重跑)
    'real_detected', 'real_bbox_u', 'real_bbox_v', 'real_bbox_w', 'real_bbox_h',
    'real_conf', 'dkf_u', 'dkf_v', 'dkf_status',
    # 飛機狀態快照 (釘在這一幀; 後處理重放 PID 直接取)
    'ac_n', 'ac_e', 'agl', 'theta_deg', 'chi_deg', 'vn', 've', 'vd', 'frame_dt',
    # 虛擬目標這幀的 setpoint (後處理用同一 setpoint 比較)
    'virt_u_sp', 'virt_v_sp',
    # 雲台
    'gimbal_yaw', 'gimbal_pitch', 'gimbal_mode',
    # 延遲補償取證 (VisionState 早就有, 之前漏記)
    'inf_ms', 'delay_ms',
]


#  MAVROS 訂閱 (照搬 run_flight.MavrosIO 慣例)
# ══════════════════════════════════════════════════════════════════
class MavrosIO:
    def __init__(self):
        self.lock = threading.Lock()
        self.guided = False
        self.connected = False
        self.lat = self.lon = None
        self.rel_alt = None
        self.compass_hdg = None
        self.vn = self.ve = self.vd = None
        self.yaw_rate_dps = 0.0
        self.airspeed = None
        self.pitch_deg = self.roll_deg = None
        self.roll_enu = self.pitch_enu = self.yaw_enu = None
        self.home_lat = self.home_lon = None
        self.last_pos_t = 0.0
        self.p_dps = self.q_dps = self.r_dps = None
        self.nz_g = None
        self.fc_mode = ''
        self.fc_pitch_tgt_deg = None
        # ★ 2026/09/11: ArduPlane 不發 ATTITUDE_TARGET (那是 Copter 的),
        #   所以 fc_pitch_tgt_deg 永遠是空的. 飛控真正在用的姿態目標要看
        #   NAV_CONTROLLER_OUTPUT 的 nav_pitch / nav_roll — 那就是 ArduPlane
        #   內部的 nav_pitch_cd / nav_roll_cd, 可以直接回答
        #   「我送的姿態命令飛控到底吃了沒有」.
        self.fc_nav_pitch_deg = None
        self.fc_nav_roll_deg = None
        self.fc_nav_bearing_deg = None
        self.batt_v = None
        self.rcout = [None] * 8
        self.gps_unix = None      # GPS 絕對時間 (unix epoch 秒; 跟車 BIN 對齊)

        rospy.Subscriber("/mavros/state", State, self._cb_state)
        rospy.Subscriber("/mavros/global_position/global", NavSatFix, self._cb_gps)
        rospy.Subscriber("/mavros/global_position/rel_alt", Float64, self._cb_alt)
        rospy.Subscriber("/mavros/global_position/compass_hdg", Float64, self._cb_hdg)
        rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self._cb_vel)
        rospy.Subscriber("/mavros/imu/data", Imu, self._cb_imu)
        rospy.Subscriber("/mavros/vfr_hud", VFR_HUD, self._cb_vfr)
        rospy.Subscriber("/mavros/home_position/home", HomePosition, self._cb_home)
        rospy.Subscriber("/mavros/rc/out", RCOut, self._cb_rcout)
        rospy.Subscriber("/mavros/battery", BatteryState, self._cb_batt)
        rospy.Subscriber("/mavros/setpoint_raw/target_attitude", AttitudeTarget, self._cb_fc_target)
        rospy.Subscriber("/mavros/nav_controller_output", NavControllerOutput,
                         self._cb_nav_out)
        # GPS 絕對時間 (跟車 BIN 對齊用): MAVROS 把 GPS 時間發在 time_reference
        rospy.Subscriber("/mavros/time_reference", TimeReference, self._cb_timeref)

    def _cb_state(self, m):
        with self.lock:
            self.connected = m.connected
            self.guided = (m.mode == 'GUIDED')
            self.fc_mode = m.mode

    def _cb_rcout(self, m):
        with self.lock:
            ch = list(m.channels[:8])
            self.rcout = ch + [None] * (8 - len(ch))

    def _cb_batt(self, m):
        with self.lock:
            self.batt_v = m.voltage

    def _cb_timeref(self, m):
        # time_reference.time_ref = GPS 絕對時間 (ROS Time). 轉 unix 秒.
        with self.lock:
            self.gps_unix = m.time_ref.to_sec()

    def _cb_nav_out(self, m):
        with self.lock:
            self.fc_nav_roll_deg = float(m.nav_roll)
            self.fc_nav_pitch_deg = float(m.nav_pitch)
            self.fc_nav_bearing_deg = float(m.nav_bearing)

    def _cb_fc_target(self, m):
        with self.lock:
            q = m.orientation
            sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
            self.fc_pitch_tgt_deg = -math.degrees(math.asin(sinp))

    def _cb_gps(self, m):
        with self.lock:
            self.lat, self.lon = m.latitude, m.longitude
            self.last_pos_t = time.time()

    def _cb_alt(self, m):
        with self.lock:
            self.rel_alt = m.data

    def _cb_hdg(self, m):
        with self.lock:
            self.compass_hdg = m.data

    def _cb_vel(self, m):
        with self.lock:
            self.vn = m.twist.linear.y
            self.ve = m.twist.linear.x
            self.vd = -m.twist.linear.z

    def _cb_imu(self, m):
        with self.lock:
            self.yaw_rate_dps = -math.degrees(m.angular_velocity.z)
            self.p_dps = math.degrees(m.angular_velocity.x)
            self.q_dps = -math.degrees(m.angular_velocity.y)
            self.r_dps = -math.degrees(m.angular_velocity.z)
            self.nz_g = m.linear_acceleration.z / 9.81
            q = m.orientation
            # ENU 四元數 → euler (給射線投影 ego 用) + pitch/roll NED
            sinr = 2.0 * (q.w * q.x + q.y * q.z)
            cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            self.roll_enu = math.atan2(sinr, cosr)
            sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
            self.pitch_enu = math.asin(sinp)
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.yaw_enu = math.atan2(siny, cosy)
            # ★ NED pitch (nose-up 正): 掛機實測 MAVROS ENU/FLU 此式解出
            #   nose-up 為負 → 取負轉 NED. 增量式 PID 直接用, 符號錯會發散.
            self.pitch_deg = -math.degrees(math.asin(sinp))
            self.roll_deg = math.degrees(math.atan2(sinr, cosr))

    def _cb_vfr(self, m):
        with self.lock:
            self.airspeed = m.airspeed

    def _cb_home(self, m):
        with self.lock:
            if self.home_lat is None:
                print(f"[IO] home = ({m.geo.latitude:.7f}, {m.geo.longitude:.7f})")
            self.home_lat = m.geo.latitude
            self.home_lon = m.geo.longitude

    def snapshot(self):
        with self.lock:
            return dict(self.__dict__)


# ══════════════════════════════════════════════════════════════════
#  共享視覺狀態 (視覺線程寫, 控制線程讀 frame_idx / 雲台角)
# ══════════════════════════════════════════════════════════════════
class SharedVision:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_idx = -1
        self.st = None                    # 最新 VisionState (真實 YOLO)
        self.gimbal_att = (0.0, 0.0, 0.0)
        self.gimbal_mode = 'follow'

    def put(self, frame_idx, st, gimbal_att, gimbal_mode):
        with self.lock:
            self.frame_idx = frame_idx
            self.st = st
            self.gimbal_att = gimbal_att
            self.gimbal_mode = gimbal_mode

    def get(self):
        with self.lock:
            return self.frame_idx, self.st, self.gimbal_att, self.gimbal_mode


def keyboard_thread(flags):
    while not flags['quit']:
        try:
            line = input().strip().lower()
        except EOFError:
            break
        if line == 'd':
            flags['arm'] = True
        elif line == 'x':
            flags['abort'] = True
        elif line == 'q':
            flags['quit'] = True


# ══════════════════════════════════════════════════════════════════
#  視覺線程 — 只驅動雲台 + 錄影 + 視覺 log; 不進飛機 PID
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
#  合成 bbox 注入 (掛機驗影像 PID → 舵面)
# ══════════════════════════════════════════════════════════════════
#  為什麼需要: 掛機時飛機不動, 虛擬目標在 AGL≈0 的幾何下:
#    · 目標只會【左右】移動 (視線永遠水平)
#    · 上下位置完全由機頭姿態決定 → 縱向 PID 驗不到
#  這裡直接覆寫 vfields 的 bbox, 讓目標自己在影像平面掃, 繞過幾何限制.
#  覆寫點在 virt.get_fields() 之後、送進狀態機之前, 所以下游 (狀態機 /
#  PID / bridge / mask / 錄影 / CSV) 全部走【正常路徑】.
#
#  幅度換算 (增益 PID_LAT_KP=4.5, K_LAT_TO_RUDDER=0.5):
#      50px → δr 3.0°   100px → 6.0°   200px → 12.1°   250px+ → 飽和
#  想看比例關係用 ≤200; 想看飽和行為才用 300.
INJECT_PATTERNS = ('lat', 'lon', 'step', 'corners')


def inject_bbox(name, t, cx, sp_v, amp, period):
    """回傳 (u, v). t 從注入開始算."""
    w = 2.0 * math.pi / max(period, 1e-3)
    if name == 'lat':                       # 只有 u 動 → 驗方向舵
        return cx + amp * math.sin(w * t), sp_v
    if name == 'lon':                       # 只有 v 動 → 驗升降舵
        return cx, sp_v + amp * math.sin(w * t)
    if name == 'step':                      # 對角階梯 → 看響應
        k = (0.0, 1.0, 0.0, -1.0)[int(t / (period / 2.0)) % 4]
        return cx + amp * k, sp_v + amp * k
    if name == 'corners':                   # 四角 → 驗四象限符號
        du, dv = ((+1, +1), (-1, +1), (-1, -1), (+1, -1))[
            int(t / (period / 2.0)) % 4]
        return cx + amp * du, sp_v + amp * dv
    return None, None


def vision_loop(cfg, io, vf, grabber, gimbal, shared, video_annot, video_raw,
                vwriter, vfile, flags, dive_flag, virt_sp_ref, t0, hb,
                err_log_path):
    """
    ~CAM_FPS Hz. 每處理一幀:
      YOLO → 雲台追蹤 (真實 bbox) → 標註影片
      → 視覺 log 釘飛機狀態 + 這幀虛擬 setpoint

    ★ 2026/09/09 兩項修正 (2026/09/08 實飛: 本執行緒在 754.7s 靜默死亡,
      主 log 卻跑到 1075.6s, raw 影片跟著缺了最後 5 分 21 秒):
      1. raw 錄影已搬到 FrameGrabber 取像執行緒, 本函式不再負責
      2. 逐幀 try/except: 單幀出錯只跳過該幀並記錄, 不再讓整條執行緒死
    hb: 心跳 dict, 主執行緒用它判斷本執行緒是否還活著.
    """
    frame_idx = -1
    last_raw_idx = -1
    last_t = time.time()
    consec_err = 0
    hb['alive'] = True
    while not flags['quit'] and not rospy.is_shutdown():
        frame, raw_idx = grabber.read_with_id()
        if frame is None:
            time.sleep(0.005)
            continue
        # ★★ 2026/09/11: 同一張相機影像不重複處理.
        #   FrameGrabber.read() 永遠回傳「最新幀」, 不會等新的.
        #   相機實測只有 15fps, 而本迴圈跑 23.7fps → 每張影像平均被
        #   處理 1.58 次. 後果不只是浪費 GPU:
        #     · DKF 會用同一筆量測連續 update → 共變異數被灌到過度自信
        #     · 標註影片與視覺 log 充滿重複幀, 統計出來的 fps 是假的
        #     · frame_dt 量到的是迴圈週期, 不是真正的量測間隔
        if raw_idx == last_raw_idx:
            hb['dup'] = hb.get('dup', 0) + 1
            time.sleep(0.002)
            continue
        last_raw_idx = raw_idx
        now = time.time()
        dt = now - last_t
        last_t = now
        frame_idx += 1
        try:
            _vision_step(cfg, io, vf, grabber, gimbal, shared, video_annot,
                         vwriter, dive_flag, virt_sp_ref, t0,
                         frame, raw_idx, frame_idx, now, dt)
            consec_err = 0
            hb['t'] = now
            hb['frame'] = frame_idx
            hb['errors'] = hb.get('errors', 0)
        except Exception:
            consec_err += 1
            hb['errors'] = hb.get('errors', 0) + 1
            tb = traceback.format_exc()
            hb['last_err'] = tb.strip().splitlines()[-1]
            msg = (f"\n[Vision] ★ 第 {frame_idx} 幀處理例外 "
                   f"(連續第 {consec_err} 次, 累計 {hb['errors']}):\n{tb}")
            print(msg)
            try:
                with open(err_log_path, 'a') as ef:
                    ef.write(f"=== t={now - t0:.3f}s frame_idx={frame_idx} "
                             f"raw_idx={raw_idx} consec={consec_err} ===\n{tb}\n")
            except Exception:
                pass
            # 連續錯太多次 = 不是偶發, 停止視覺但【不】影響控制與 raw 錄影
            if consec_err >= 50:
                hb['alive'] = False
                hb['fatal'] = f"連續 {consec_err} 幀例外, 視覺已停止"
                print(f"[Vision] ★★★ {hb['fatal']} (raw 錄影與控制不受影響)")
                return
            time.sleep(0.01)
    hb['alive'] = False
    hb['fatal'] = None       # 正常退出


def _vision_step(cfg, io, vf, grabber, gimbal, shared, video_annot,
                 vwriter, dive_flag, virt_sp_ref, t0,
                 frame, raw_idx, frame_idx, now, dt):
    """單幀處理 (原 vision_loop 迴圈本體). 任何例外由呼叫端捕捉."""
    s = io.snapshot()
    guided = s['guided']
    dive_mode = dive_flag.is_set()
    gy, gp, gr = gimbal.get_attitude()

    ac = {
        'roll_enu': s['roll_enu'] or 0.0,
        'pitch_enu': s['pitch_enu'] or 0.0,
        'yaw_enu': s['yaw_enu'] or 0.0,
        'body_pitch_deg': s['pitch_deg'] or 0.0,
        'body_yaw_true_deg': s['compass_hdg'] or 0.0,
        'agl': s['rel_alt'],
    }
    if s['lat'] is not None and s['home_lat'] is not None:
        ac['x_ned'], ac['y_ned'] = gps_to_ned(
            s['lat'], s['lon'], s['home_lat'], s['home_lon'])
    else:
        ac['x_ned'] = ac['y_ned'] = 0.0
    gimbal_dict = {'yaw': gy, 'pitch': gp}

    # YOLO + DKF + (測距, 這版 log 用) — 誤差不進飛機 PID
    st, annot = vf.step(frame, ac, gimbal_dict, dive_mode=dive_mode, dt=dt)

    # ── 雲台作動 (原本模式全保留) ──
    if not guided and not vf.locked:
        gimbal.standby_tick()
    elif dive_mode:
        gimbal.enter_fpv()
    else:
        has_t = st.detected or st.dkf_used
        u = st.u_used if st.u_used is not None else st.bbox_u
        v = st.v_used if st.v_used is not None else st.bbox_v
        gimbal.track_tick(has_t, u, v, cfg.IMG_W, cfg.IMG_H, vf.fx, vf.fy)

    # 標註影片 (節流 25fps, 給人看)
    if video_annot is not None:
        video_annot.write(annot)

    # ── 視覺 log (釘這一幀飛機狀態 + 虛擬 setpoint) ──
    vsp = dict(virt_sp_ref)   # 控制線程最新虛擬 setpoint
    vrow = {
        'frame_idx': frame_idx, 'raw_frame_idx': raw_idx,
        't': round(now - t0, 3),
        'gps_unix': (f"{s['gps_unix']:.3f}" if s['gps_unix'] is not None else ''),
        'mode': 'DIVE' if dive_mode else ('GUIDED' if guided else 'STANDBY'),
        'dive_mode': int(dive_mode),
        'real_detected': int(st.detected),
        'real_bbox_u': round(st.bbox_u, 1) if st.bbox_u is not None else '',
        'real_bbox_v': round(st.bbox_v, 1) if st.bbox_v is not None else '',
        'real_bbox_w': round(st.bbox_w, 1) if st.bbox_w is not None else '',
        'real_bbox_h': round(st.bbox_h, 1) if st.bbox_h is not None else '',
        'real_conf': round(st.conf, 3) if st.conf is not None else '',
        'dkf_u': round(st.u_used, 1) if st.u_used is not None else '',
        'dkf_v': round(st.v_used, 1) if st.v_used is not None else '',
        'dkf_status': st.dkf_status or '',
        'ac_n': round(ac['x_ned'], 2), 'ac_e': round(ac['y_ned'], 2),
        'agl': round(s['rel_alt'], 2) if s['rel_alt'] is not None else '',
        'theta_deg': round(s['pitch_deg'], 2) if s['pitch_deg'] is not None else '',
        'chi_deg': round(s['compass_hdg'], 2) if s['compass_hdg'] is not None else '',
        'vn': round(s['vn'], 2) if s['vn'] is not None else '',
        've': round(s['ve'], 2) if s['ve'] is not None else '',
        'vd': round(s['vd'], 2) if s['vd'] is not None else '',
        'frame_dt': round(dt, 4),
        'virt_u_sp': vsp.get('u', ''), 'virt_v_sp': vsp.get('v', ''),
        'gimbal_yaw': round(gy, 1), 'gimbal_pitch': round(gp, 1),
        'gimbal_mode': gimbal.mode,
        'inf_ms': round(st.inf_ms, 2) if getattr(st, 'inf_ms', None) is not None else '',
        'delay_ms': round(st.delay_ms, 2) if getattr(st, 'delay_ms', None) is not None else '',
    }
    vwriter.writerow(vrow)

    shared.put(frame_idx, st, (gy, gp, gr), gimbal.mode)


# ══════════════════════════════════════════════════════════════════
#  主程式 (控制線程)
# ══════════════════════════════════════════════════════════════════
def _parse_gps(s, name):
    try:
        a, b = s.split(',')
        return float(a), float(b)
    except Exception:
        print(f"[FATAL] --{name} 格式應為 lat,lon (例 23.0012,120.0056)")
        sys.exit(1)


def main():
    if not ROS_OK:
        print("rospy / mavros 不可用 — 本程式只能在機載電腦 (ROS 環境) 執行.")
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument('--moving', dest='moving', action='store_true', default=None,
                    help='移動目標 (預設由 config.MOVING_TARGET 決定)')
    ap.add_argument('--static', dest='moving', action='store_false',
                    help='靜止目標')
    # 移動版: 目標起點 + 車速 (方位由 config 兩座標算; 不用飛機航向)
    ap.add_argument('--start', type=str, default=None,
                    help='移動目標起點 GPS "lat,lon" (未給用 config.MOVING_START_GPS)')
    ap.add_argument('--speed', type=float, default=None,
                    help='移動目標車速 m/s (未給用 config.TARGET_SPEED_MPS)')
    # 靜止版: 目標 + PRE_DIVE 點
    ap.add_argument('--target', type=str, default=None,
                    help='靜止目標 GPS "lat,lon"')
    ap.add_argument('--predive', type=str, default=None,
                    help='靜止版 PRE_DIVE 點 GPS "lat,lon"')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--no-annot', action='store_true',
                    help='不錄標註影片 (省一路 NVENC + 一半磁碟寫入). '
                         '框的座標都在視覺 CSV 裡, 事後用 make_annot_video.py '
                         '重建即可 — 內容完全一樣.')
    ap.add_argument('--bench', action='store_true',
                    help='掛機: 忽略 AGL 進入條件 (仍需 GUIDED). '
                         '會停在 PRE_DIVE 持續命令俯衝姿態, 方便觀察舵面.')
    ap.add_argument('--tgt-heading', type=float, default=None, metavar='DEG',
                    help='移動目標的行進方位 (度), 覆寫 config 的 '
                         'MOVING_HEADING_DEG / MOVING_HEADING_GPS. '
                         '掛機驗橫向時設成【垂直於視線】的方位, 車子才會橫掃'
                         '過畫面; 用 config 的馬路方位會讓車子直接跑遠.')
    ap.add_argument('--inject-bbox', choices=INJECT_PATTERNS, default=None,
                    help='掛機: 用合成 bbox 覆寫虛擬目標的影像位置, 驗影像 '
                         'PID → 舵面. lat=橫向掃(方向舵) lon=縱向掃(升降舵) '
                         'step=對角階梯 corners=四角. 只在 --bench 有效.')
    ap.add_argument('--inject-amp', type=float, default=150.0, metavar='PX',
                    help='合成 bbox 幅度 (px, 預設 150). 50→δr 3° '
                         '100→6° 200→12° 250以上飽和.')
    ap.add_argument('--inject-period', type=float, default=8.0, metavar='SEC',
                    help='合成 bbox 週期 (秒, 預設 8)')
    ap.add_argument('--bench-pitch', type=float, default=None, metavar='DEG',
                    help='掛機: DIVE 縱向增量的基準姿態覆寫 (例 --bench-pitch -20). '
                         'DIVE 的 pitch 是 pitch_now+dtheta, 桌上 pitch_now≈+1 → '
                         '一進 DIVE 舵面就回平. 設成 -20 才看得到俯衝姿態. '
                         '只在 --bench 有效, 實飛不要加.')
    ap.add_argument('--bench-dive', action='store_true',
                    help='掛機時允許跳過 pitch 容差直接交棒 DIVE. '
                         '★ 注意: DIVE 的 pitch 命令是相對實際姿態的增量式, '
                         '桌上實際 pitch≈0 → 進 DIVE 後升降舵就不動了. '
                         '只有要驗狀態機轉換時才用, 看舵面請用 --bench.')
    ap.add_argument('--ground-test', action='store_true',
                    help='地面測試: SEARCH 保持機頭不朝目標 (切 GUIDED 不打 yaw)')
    ap.add_argument('--dive-hold-heading', action='store_true',
                    help='版本 B: DIVE 不修 yaw, 凍結進 DIVE 那刻的航向 (縱向仍修)')
    ap.add_argument('--case', type=str, default=None,
                    help='自訂 case 標籤加進檔名 (未給則自動用 移動/靜止+版本A/B)')
    args = ap.parse_args()

    cfg = Config()
    fx, fy, cx, cy = cfg.effective_intrinsics()

    moving = cfg.MOVING_TARGET if args.moving is None else args.moving
    # ★ 2026/09/12: 把實際模式寫回 cfg, 否則 --static 時 vision.py 的
    #   地理閘門還是會用移動目標的膨脹半徑 (兩邊模式不一致).
    cfg.MOVING_TARGET = moving

    if moving:
        start_gps = (_parse_gps(args.start, 'start') if args.start
                     else cfg.MOVING_START_GPS)
        speed = args.speed if args.speed is not None else cfg.TARGET_SPEED_MPS
        if start_gps is None:
            print("[FATAL] 移動版需目標起點 GPS (--start 或 config.MOVING_START_GPS)")
            sys.exit(1)
        # 方位優先序: --tgt-heading > config.MOVING_HEADING_DEG > 兩點算
        if args.tgt_heading is not None:
            heading = float(args.tgt_heading) % 360.0
            print(f"[flight] 目標方位由 --tgt-heading 指定: {heading:.1f}°")
        elif cfg.MOVING_HEADING_DEG is not None:
            heading = cfg.MOVING_HEADING_DEG
        else:
            hn, he = gps_to_ned(cfg.MOVING_HEADING_GPS[0], cfg.MOVING_HEADING_GPS[1],
                                start_gps[0], start_gps[1])
            heading = math.degrees(math.atan2(he, hn)) % 360.0
        print(f"[flight] 移動目標: 起點 {start_gps} 方位 {heading:.1f}° 車速 {speed} m/s")
    else:
        target_gps = (_parse_gps(args.target, 'target') if args.target
                      else getattr(cfg, 'TARGET_GPS', None))
        predive_gps = (_parse_gps(args.predive, 'predive') if args.predive
                       else getattr(cfg, 'PREDIVE_GPS', None))
        if target_gps is None or predive_gps is None:
            print("[FATAL] 靜止版需目標與 PRE_DIVE 點 GPS "
                  "(--target/--predive 或 config)")
            sys.exit(1)

    rospy.init_node('vision_dive_flight', anonymous=True)
    io = MavrosIO()

    gcfg = GuidanceConfig(
        DIVE_ANGLE_DEG=cfg.DIVE_ANGLE_DEG,
        ALPHA_NOM_DEG=cfg.ALPHA_NOM_DEG,
        ALPHA_DIVE_DEG=cfg.ALPHA_DIVE_DEG,
        PREDIVE_TRIGGER_MODE=('target_dist' if moving else 'gps_point'),
        PREDIVE_REACH_RADIUS_M=cfg.PREDIVE_REACH_RADIUS_M,
        PREDIVE_DIST_M=cfg.PREDIVE_DIST_M,
        PREDIVE_PUSH_MARGIN_M=cfg.PREDIVE_PUSH_MARGIN_M,
        PREDIVE_HANDOFF_PITCH_TOL_DEG=cfg.PREDIVE_HANDOFF_PITCH_TOL_DEG,
        PULLOUT_AGL_M=cfg.PULLOUT_AGL_M,
        PULLOUT_VMAX_MPS=cfg.PULLOUT_VMAX_MPS,
        TECS_HOLD_ALT_M=cfg.TECS_HOLD_ALT_M,
        TECS_SPEED_MPS=cfg.TECS_SPEED_MPS,
        ABORT_MIN_AGL_M=(0.0 if args.bench else cfg.ABORT_MIN_AGL_M),
        ABORT_TARGET_LOST_S=cfg.ABORT_TARGET_LOST_S,
        ENTRY_MIN_AGL_M=(0.0 if args.bench else 60.0),
        BENCH_GEOMETRY_AGL_M=(120.0 if args.bench else 0.0),
        BENCH_HANDOFF_ANY_PITCH=args.bench_dive,
        BENCH_DIVE_PITCH_BASE_DEG=(args.bench_pitch if args.bench else None),
        FX=fx, FY=fy, CX=cx, CY=cy, IMG_W=cfg.IMG_W, IMG_H=cfg.IMG_H,
        PID_LAT_KP=cfg.PID_LAT_KP, PID_LAT_KI=cfg.PID_LAT_KI, PID_LAT_KD=cfg.PID_LAT_KD,
        PID_LON_KP=cfg.PID_LON_KP, PID_LON_KI=cfg.PID_LON_KI, PID_LON_KD=cfg.PID_LON_KD,
        PID_SP_LEAD_S=cfg.PID_SP_LEAD_S,
        PID_MEAS_TAU_S=cfg.PID_MEAS_TAU_S, PID_D_TAU_S=cfg.PID_D_TAU_S,
        PID_OUT_TAU_S=cfg.PID_OUT_TAU_S,
        # 移動: 前置瞄準; 靜止: 不前置.
        # ★ 2026/09/12: --inject-bbox 時強制關閉前置瞄準.
        #   原因: lead_setpoint_px 依賴 t_go, 而掛機時飛機不動 →
        #     vd≈0  → 不走 AGL/vd
        #     接近率≈0 → t_go = 斜距/max(接近率,3) 被夾在 30s 上限
        #   → 前置點偏目標 60m → setpoint 偏畫面中心 850px 以上
        #   → 誤差 = bbox − setpoint 恆為大值 → PID 直接飽和 → 舵面貼死不動
        #   (2026/09/12 實測: u_sp 掉到 112, dpsi 全程飽和 30°)
        #   注入的目的是看「誤差變化 → 舵面修正」, 前置瞄準會把誤差直接
        #   推到飽和, 兩件事互相汙染. 前置瞄準的正確性只能實飛驗
        #   (它需要真實接近率), 掛機驗不了.
        DIVE_TARGET_LEAD=(moving and not (args.bench and args.inject_bbox)),
        DIVE_AIM_V_OFFSET_PX=cfg.dive_aim_v_offset_px,
        GIMBAL_DIVE_TILT_BODY_DEG=0.0,
        GIMBAL_FPV_SETTLE_S=0.3,
        SEARCH_HOLD_HEADING=args.ground_test,   # 地面測試不朝目標打 yaw
        DIVE_HOLD_HEADING=args.dive_hold_heading,  # 版本 B: DIVE 不修 yaw
    )
    sm = MissionStateMachine(gcfg, verbose=True)

    bcfg = AttitudeBridgeConfig()
    bcfg.TECS_HOLD_ALT_M = cfg.TECS_HOLD_ALT_M
    bcfg.TECS_SPEED_MPS = cfg.TECS_SPEED_MPS
    bcfg.V_REF_MPS = cfg.V_CRUISE_MPS
    bcfg.THR_TRIM = cfg.THR_TRIM
    bcfg.THR_MIN = cfg.THR_MIN
    # ★ 2026/09/12: 方向舵迴路參數改由 config.py 統一管理 (原本寫在
    #   AttitudeBridgeConfig 裡, 場地上要改得翻兩個檔案).
    #   用 getattr 取, 舊 config 沒有這些欄位就沿用 bridge 自己的預設值.
    for _k in ('RUDDER_LIMIT_DEG', 'K_PSI', 'K_R', 'K_LAT_TO_RUDDER',
               'RUDDER_SLEW_DPS', 'RUDDER_VSCHED',
               'ROLL_PROT_FULL_DEG', 'ROLL_PROT_ZERO_DEG'):
        if hasattr(cfg, _k):
            setattr(bcfg, _k, getattr(cfg, _k))
    bridge = ArduPilotAttitudeBridge(dive_angle_deg=cfg.DIVE_ANGLE_DEG, cfg=bcfg)

    from gimbal import GimbalManager
    gimbal = GimbalManager(cfg)
    if not gimbal.connect():
        print("[FATAL] 雲台連線失敗")
        sys.exit(1)

    grabber = FrameGrabber(cfg)
    if not grabber.start():
        print("[FATAL] 取像失敗")
        gimbal.close()
        sys.exit(1)

    vf = VisionFrontend(cfg)

    print("[flight] 等待 home_position 與 GPS fix ...")
    t_wait = time.time()
    while not rospy.is_shutdown():
        s = io.snapshot()
        if s['home_lat'] is not None and s['lat'] is not None:
            break
        if time.time() - t_wait > 30.0:
            print("[flight] ★ 30s 沒等到 home/GPS, 檢查 MAVROS")
            t_wait = time.time()
        rospy.sleep(0.5)

    s = io.snapshot()
    from virtual_source import StaticTarget, MovingTarget
    if moving:
        sn, se = gps_to_ned(start_gps[0], start_gps[1], s['home_lat'], s['home_lon'])
        # t0 佔位 (實際在 GUIDED 上升沿重設); 進俯衝用「到目標距離」故 predive_point=None
        target = MovingTarget(sn, se, heading, speed, t0=0.0)
        virt = VirtualSource(cfg, gcfg, s['home_lat'], s['home_lon'], target,
                             predive_point_ne=None)
    else:
        tn, te = gps_to_ned(target_gps[0], target_gps[1], s['home_lat'], s['home_lon'])
        pn, pe = gps_to_ned(predive_gps[0], predive_gps[1], s['home_lat'], s['home_lon'])
        target = StaticTarget(tn, te)
        virt = VirtualSource(cfg, gcfg, s['home_lat'], s['home_lon'], target,
                             predive_point_ne=(pn, pe))

    # ── log + 影片 ──
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.OUT_DIR)
    os.makedirs(log_dir, exist_ok=True)
    # ★ case 標籤加進檔名, 不用另外記哪趟是哪個 case.
    #   自訂 --case 優先; 否則自動組: 移動/靜止 + 版本A/B (yaw修/凍結)
    if args.case:
        tag = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in args.case)
    else:
        tag = ('moving' if moving else 'static') + \
              ('_B_yawhold' if args.dive_hold_heading else '_A_yawpid')
    stem = f'flight_{stamp}_{tag}'
    csv_path = os.path.join(log_dir, f'{stem}.csv')
    vis_path = os.path.join(log_dir, f'{stem}_vision.csv')
    ev_path = os.path.join(log_dir, f'{stem}_events.txt')
    vid_annot = os.path.join(log_dir, f'{stem}.mp4')
    vid_raw = os.path.join(log_dir, f'{stem}_raw.mp4')

    fcsv = open(csv_path, 'w', newline='', buffering=1)
    writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS, extrasaction='ignore')
    writer.writeheader()
    fvis = open(vis_path, 'w', newline='', buffering=1)
    vwriter = csv.DictWriter(fvis, fieldnames=VISION_FIELDS, extrasaction='ignore')
    vwriter.writeheader()
    fcsv.flush(); os.fsync(fcsv.fileno())
    fvis.flush(); os.fsync(fvis.fileno())
    fev = open(ev_path, 'a', buffering=1)
    t0 = time.time()

    def event(msg):
        line = f"{time.time() - t0:8.2f}s  {msg}"
        print(line)
        fev.write(line + '\n')
        fev.flush(); os.fsync(fev.fileno())

    video_annot = video_raw = None
    if not args.no_video:
        # ★ 2026/09/11: --no-annot 省掉一路 1080p NVENC.
        #   標註影片 = raw 影片 + 框, 而框的座標全在視覺 CSV 裡,
        #   事後 make_annot_video.py 可以完全重建 → 飛行中沒有必要編碼.
        #   Jetson 出現 over-current 降頻時, 這是最划算的減負.
        if args.no_annot:
            print("[flight] --no-annot: 不錄標註影片 "
                  "(事後 make_annot_video.py 重建)")
        else:
            # 節流關閉: 視覺迴圈本來就低於 VIDEO_FPS, 節流只會白丟幀
            video_annot = VideoWriter(vid_annot, cfg.VIDEO_FPS,
                                      cfg.IMG_W, cfg.IMG_H, throttle=False)
            if not video_annot.ok:
                video_annot = None
        # ★ raw: 不節流, queue 放大 (8 幀只有 0.3s 緩衝, 編碼器一頓就掉幀)
        video_raw = VideoWriter(vid_raw, cfg.CAM_FPS, cfg.IMG_W, cfg.IMG_H,
                                throttle=False, queue_len=120)
        if not video_raw.ok:
            print("[flight] ★ raw 影片啟動失敗 — 後處理需要它!")
            video_raw = None
        else:
            # ★★ raw 錄影掛到取像執行緒 (與 YOLO 解耦).
            #    YOLO/DKF/雲台/CSV 任何一步掛掉都不會再讓 raw 影片跟著停.
            grabber.recorder = video_raw
            print("[flight] raw 錄影已掛在取像執行緒 (與 YOLO 解耦)")

    # 背景 fsync (兩個 csv)
    _sync_run = {'on': True}

    def _fsync_loop():
        while _sync_run['on']:
            for f in (fcsv, fvis):
                try:
                    f.flush(); os.fsync(f.fileno())
                except (OSError, ValueError):
                    pass
            time.sleep(1.0 / cfg.FSYNC_HZ)
    threading.Thread(target=_fsync_loop, daemon=True).start()
    print(f"[flight] 主 log → {csv_path}")
    print(f"[flight] 視覺 log → {vis_path}")
    print(f"[flight] 影片 → {vid_annot} + {vid_raw} (raw)")

    flags = {'arm': False, 'abort': False, 'quit': False}
    threading.Thread(target=keyboard_thread, args=(flags,), daemon=True).start()

    shared = SharedVision()
    dive_flag = threading.Event()
    virt_sp_ref = {'u': cx, 'v': cy}   # 控制線程每拍更新, 視覺線程讀去記 log

    # ★ 視覺心跳 (主執行緒看門狗用) + 例外 log
    vision_hb = {'t': time.time(), 'frame': -1, 'alive': False,
                 'errors': 0, 'fatal': None, 'last_err': None, 'dup': 0}
    err_log_path = os.path.join(log_dir, f'{stem}_error.log')
    vision_thread = threading.Thread(
        target=vision_loop,
        args=(cfg, io, vf, grabber, gimbal, shared, video_annot, video_raw,
              vwriter, fvis, flags, dive_flag, virt_sp_ref, t0, vision_hb,
              err_log_path),
        daemon=True)
    vision_thread.start()

    bridge.start()
    print("[flight] ★ AUTO 對準 → 撥 GUIDED = 任務開始; 到 PRE_DIVE 點自動俯衝")
    print("[flight] 掛機備用鍵盤: d=arm x=abort q=quit")

    rate = rospy.Rate(LOOP_HZ)
    half_w, half_h = cfg.IMG_W / 2.0, cfg.IMG_H / 2.0
    guided_prev = False
    mode_prev = sm.mode
    last_print = 0.0
    # 合成 bbox 注入狀態 (只在 --bench 生效; 實飛 args.inject_bbox 為 None)
    _inject = args.inject_bbox if args.bench else None
    _inject_t0 = None
    _inject_sp_v = cy + cfg.dive_aim_v_offset_px
    if _inject and not args.bench:
        print("[flight] ⚠ --inject-bbox 只在 --bench 有效, 已忽略")
    if _inject and moving:
        print("[flight] ⚠ --inject-bbox + 移動目標: 前置瞄準 (DIVE_TARGET_LEAD) "
              "已自動關閉")
        print("[flight]   掛機時 t_go 會被夾在 30s 上限 → setpoint 偏 850px → "
              "誤差直接飽和")
        print("[flight]   前置瞄準的正確性請用 test_logic_sim.py --check 驗 "
              "(那裡飛機真的會飛)")
    _vid_check_t = 0.0
    _vid_warned = set()
    err_count = 0
    scored = {'done': False}
    _miss_warned = False
    pid_last_t = None

    def do_arm(s, t):
        # ★ 移動目標: 任務起點時刻 = 現在 (t), 目標從起點開始跑
        if moving:
            virt.target.t0 = t
            tn, te, _, _ = virt.target_now(t)
            event(f"種點(移動): 目標起點 NED=({tn:.1f},{te:.1f}) "
                  f"方位 {virt.target.heading_deg:.1f}° 車速 {virt.target.speed} m/s "
                  f"t0={t:.1f}s")
        else:
            tn, te, _, _ = virt.target_now(t)
            event(f"種點(靜止): 目標 NED=({tn:.1f},{te:.1f}) "
                  f"PRE_DIVE NED=({virt.pd_point[0]:.1f},{virt.pd_point[1]:.1f})")
        # ★ 2026/09/11: 撥 GUIDED 時送【當下高度】, 不是 TECS_HOLD_ALT_M.
        #   SEARCH 已改成 pitch/高度全交 TECS, 這裡若送 70m, 飛機一撥 GUIDED
        #   就會從巡航 120m 往下掉, 到 PRE_DIVE 點高度全錯.
        #   70m 要等進 PULLOUT 才送 (sm.tecs_resend 觸發).
        _cruise_alt = s.get('rel_alt')
        if _cruise_alt is None or _cruise_alt < 5.0:
            _cruise_alt = cfg.CRUISE_AGL_M          # 掛機/沒高度時用設定值
        _search_spd = getattr(cfg, 'SEARCH_SPEED_MPS', cfg.TECS_SPEED_MPS)
        ok_a, ok_s = bridge.send_tecs_targets(alt_m=_cruise_alt,
                                              speed_mps=_search_spd,
                                              tag='/巡航')
        event(f"TECS 目標預送 ALT={_cruise_alt:.0f}m "
              f"SPD={_search_spd:.0f}m/s ok=({ok_a},{ok_s})")
        if not ok_s:
            _w = (f"★ SEARCH 空速目標 {_search_spd:.0f}m/s 被飛控拒絕 — "
                  f"檢查 AIRSPEED_MIN (舊韌體 ARSPD_FBW_MIN) "
                  f"是否 ≤ {_search_spd:.0f}")
            print(_w)
            event(_w)
        sm.arm()

    try:
        while not rospy.is_shutdown() and not flags['quit']:
            try:
                t = time.time() - t0
                s = io.snapshot()

                if s['guided'] and not guided_prev:
                    if sm.mode == sm.SEARCH and not sm.mission_armed:
                        event("★ GUIDED 上升沿 — 種點 + ARM (任務開始)")
                        if s['lat'] is not None:
                            do_arm(s, t)
                    elif sm.mode == sm.DONE:
                        print("[flight] 任務已結束 (DONE) — 重啟程式再飛")
                guided_prev = s['guided']

                if flags['arm']:
                    flags['arm'] = False
                    if s['guided'] and s['lat'] is not None:
                        do_arm(s, t)
                if flags['abort']:
                    flags['abort'] = False
                    sm.operator_abort(t)

                if s['lat'] is None or s['home_lat'] is None:
                    rate.sleep()
                    continue

                ac_n, ac_e = gps_to_ned(s['lat'], s['lon'],
                                        s['home_lat'], s['home_lon'])
                agl = s['rel_alt']
                hdg = s['compass_hdg'] if s['compass_hdg'] is not None else 0.0
                pitch = s['pitch_deg'] if s['pitch_deg'] is not None else 0.0
                V = s['airspeed'] if s['airspeed'] is not None else 0.0

                dive_mode = (sm.mode == sm.DIVE)

                # ── 虛擬目標: 這一版控制路徑的影像誤差來源 ──
                vac = {
                    'n': ac_n, 'e': ac_e, 'agl': agl,
                    'theta_deg': pitch, 'chi_deg': hdg,
                    'gimbal_locked': dive_mode,   # DIVE 用 body 投影, 否則置中
                    't': t,                       # 移動目標算當下位置用
                }
                vfields = virt.get_fields(vac)

                # ── 合成 bbox 注入 (只在 --bench + --inject-bbox) ──
                if _inject is not None:
                    if _inject_t0 is None:
                        _inject_t0 = t
                        print(f"[flight] ★ 合成 bbox 注入啟動: {_inject} "
                              f"幅度 ±{args.inject_amp:.0f}px "
                              f"週期 {args.inject_period:.0f}s "
                              f"(瞄準點 u={cx:.0f} v={_inject_sp_v:.1f})")
                    _iu, _iv = inject_bbox(_inject, t - _inject_t0, cx,
                                           _inject_sp_v, args.inject_amp,
                                           args.inject_period)
                    if _iu is not None:
                        vfields['bbox_u'] = _iu
                        vfields['bbox_v'] = _iv
                        vfields['bbox_detected'] = 1
                        vfields['bbox_center_offset_norm'] = (
                            math.hypot(_iu - cx, _iv - cy)
                            / math.hypot(cfg.IMG_W / 2.0, cfg.IMG_H / 2.0))

                frame_idx, vst, gatt, gmode = shared.get()

                # ── 組狀態機 inp: 視覺欄位全來自虛擬目標 ──
                inp = {
                    't': t, 'guided': s['guided'],
                    'agl_m': agl, 'airspeed_mps': s['airspeed'],
                    'gs_mps': (math.hypot(s['vn'], s['ve'])
                               if s['vn'] is not None else None),
                    'vn': s['vn'], 've': s['ve'], 'vd': s['vd'],
                    'yaw_true_deg': hdg, 'pitch_deg': pitch, 'roll_deg': s['roll_deg'],
                    'ac_n': ac_n, 'ac_e': ac_e,
                    **vfields,   # tgt_n/tgt_e/tgt_valid/slant_m/ground_m/
                                 # predive_dist_m/bbox_*/bbox_center_offset_norm
                }
                out = sm.update(inp)

                # ── 視覺線程要記的虛擬 setpoint ───────────────────
                # ★★ 2026/09/12 修正 (影響論文核心比較):
                #   舊版把 vfields['bbox_u/v'] 寫進來 —— 那是虛擬目標投影在
                #   畫面上的【量測值】, 不是 PID 的 setpoint.
                #   真正的 setpoint 在 dive_guidance 裡另外算:
                #     sp_u/v = lead_setpoint_px(...)      移動目標的前置偏移
                #     sp_v  += DIVE_AIM_V_OFFSET_PX       攻角補償 (48.1 px)
                #     再過 τ=0.6s 低通
                #   後處理是拿 virt_u_sp/virt_v_sp 當參考去重放 real PID 的,
                #   記錯就等於【把誤差當成目標】, real-vs-virtual 的逐幀比較
                #   全部失效 —— 而那正是這一版流程要驗的核心結論.
                #   靜止版也躲不掉: 那 48.1 px 的攻角補償是常數偏移.
                #   必須放在 sm.update() 之後才拿得到.
                virt_sp_ref['u'] = out.get('pid_u_sp', cx)
                virt_sp_ref['v'] = out.get('pid_v_sp', cy)

                if out['mode'] == 'DIVE':
                    dive_flag.set()
                else:
                    dive_flag.clear()

                if out['mode'] != mode_prev:
                    banner = (f"★★ MODE: {mode_prev} → {out['mode']} "
                              f"(t={t:.1f}s AGL={agl if agl is not None else '?'} "
                              f"V={V:.1f} pitch_cmd={out['pitch_cmd_deg']:+.1f} "
                              f"hdg_cmd={out['heading_cmd_deg']:.1f}"
                              + (f" abort={out.get('abort_reason')}"
                                 if out.get('abort_reason') else "") + ")")
                    print("\n" + "=" * 64 + "\n" + banner + "\n" + "=" * 64)
                    event(banner)
                    mode_prev = out['mode']
                    if getattr(sm, 'tecs_resend', False):
                        # PULLOUT 觸發 → 現在才把 TECS 高度目標切到拉平高度
                        sm.tecs_resend = False
                        bridge.send_tecs_targets(
                            alt_m=cfg.TECS_HOLD_ALT_M, tag='/拉平')
                        event(f"TECS 高度目標 → {cfg.TECS_HOLD_ALT_M:.0f}m")

                bridge.submit(out, hdg, s['yaw_rate_dps'], s['airspeed'],
                              roll_deg=s['roll_deg'])

                # ── 主 log ──
                dt_loop = (t - pid_last_t) if pid_last_t is not None else 0.0
                pid_last_t = t
                gam = (math.degrees(math.asin(max(-1, min(1,
                       -s['vd'] / max(V, 1.0))))) if s['vd'] is not None else 0.0)
                vu = vfields['bbox_u']
                vv = vfields['bbox_v']
                _brg = bridge.get_debug()

                # ── 逐幀 miss distance (沿當下速度向量外插到地面) ──
                _mp = _mlon = _mlat = _tf = ''
                try:
                    # ★ 2026/09/11 修正: 原本寫 s['ac_n'] / s['ac_e'], 但
                    #   MavrosIO 沒有這兩個屬性 (NED 是下面 gps_to_ned 算的
                    #   local variable) → KeyError 被 except 吃掉 →
                    #   miss_pred_* 永遠是空值而且不報錯.
                    if (agl is not None and agl > 0.5
                            and (s['vd'] or 0.0) > 0.1
                            and vfields.get('tgt_valid')):
                        _in, _ie, _tfall = extrapolate_impact(
                            ac_n, ac_e, agl,
                            s['vn'] or 0.0, s['ve'] or 0.0, s['vd'] or 0.0)
                        if _in is not None:
                            _md = miss_distance(
                                _in, _ie, _tfall,
                                vfields['tgt_n'], vfields['tgt_e'],
                                vfields.get('tgt_vn') or 0.0,
                                vfields.get('tgt_ve') or 0.0,
                                s['vn'] or 0.0, s['ve'] or 0.0)
                            _mp = round(_md['miss_total'], 2)
                            _mlon = round(_md['miss_lon'], 2)
                            _mlat = round(_md['miss_lat'], 2)
                            _tf = round(_tfall, 2)
                except Exception as _e:
                    # ★ 不再靜默吞掉: 只印第一次, 避免洗版
                    if not _miss_warned:
                        _miss_warned = True
                        _w = f"★ 逐幀 miss 計算失敗: {type(_e).__name__}: {_e}"
                        print("\n" + _w)
                        event(_w)
                row = {
                    't': round(t, 3),
                    'gps_unix': (f"{s['gps_unix']:.3f}" if s['gps_unix'] is not None else ''),
                    'frame_idx': frame_idx,
                    'mode': out['mode'], 'guided': int(s['guided']),
                    'ac_n': round(ac_n, 2), 'ac_e': round(ac_e, 2),
                    'agl': round(agl, 2) if agl is not None else '',
                    'V': round(V, 2),
                    'chi_deg': round(hdg, 2), 'gamma_deg': round(gam, 2),
                    'theta_deg': round(pitch, 2),
                    'roll_deg': round(s['roll_deg'], 2) if s['roll_deg'] is not None else '',
                    'yaw_deg': round(hdg, 1),
                    'vn': round(s['vn'], 2) if s['vn'] is not None else '',
                    've': round(s['ve'], 2) if s['ve'] is not None else '',
                    'vd': round(s['vd'], 2) if s['vd'] is not None else '',
                    'dt': round(dt_loop, 4),
                    'tgt_n': round(vfields['tgt_n'], 2),
                    'tgt_e': round(vfields['tgt_e'], 2),
                    'slant_m': round(vfields['slant_m'], 1) if vfields['slant_m'] is not None else '',
                    'ground_m': round(vfields['ground_m'], 1) if vfields['ground_m'] is not None else '',
                    'predive_dist_m': round(vfields['predive_dist_m'], 1),
                    'virt_bbox_u': round(vu, 1) if vu is not None else '',
                    'virt_bbox_v': round(vv, 1) if vv is not None else '',
                    'virt_detected': int(vfields['bbox_detected']),
                    'virt_u_sp': out.get('pid_u_sp', ''),
                    'virt_v_sp': out.get('pid_v_sp', ''),
                    'pitch_cmd': out['pitch_cmd_deg'],
                    'heading_cmd': out['heading_cmd_deg'],
                    'throttle': _brg.get('brg_thr', ''),
                    **{k: _brg.get(k, '') for k in (
                        'brg_mask', 'rudder_raw_deg', 'rudder_cmd_deg',
                        'rudder_sent_deg', 'rudder_limit_deg',
                        'lim_by_airspeed', 'lim_by_roll',
                        'hdg_cmd_ok_cnt', 'lateral_fallback',
                        'brg_stale', 'brg_stale_events')},
                    **{k: out.get(k, '') for k in (
                        'pid_u_err_deg', 'pid_v_err_deg',
                        'pid_lat_p', 'pid_lat_i', 'pid_lat_d',
                        'pid_lon_p', 'pid_lon_i', 'pid_lon_d',
                        'pid_dpsi_deg', 'pid_dtheta_deg', 'pid_compute_ms')},
                    'miss_pred_m': _mp,
                    'miss_pred_lon_m': _mlon,
                    'miss_pred_lat_m': _mlat,
                    'impact_t_fall_s': _tf,
                    'img_err_u_pct': (round((vu - cx) / half_w * 100, 2)
                                      if vu is not None else ''),
                    'img_err_v_pct': (round((vv - cy) / half_h * 100, 2)
                                      if vv is not None else ''),
                    'glide_slope_err_deg': (round(gam + cfg.DIVE_ANGLE_DEG, 2)
                                            if out['mode'] == 'DIVE' else ''),
                    'p_dps': round(s['p_dps'], 1) if s['p_dps'] is not None else '',
                    'q_dps': round(s['q_dps'], 1) if s['q_dps'] is not None else '',
                    'r_dps': round(s['r_dps'], 1) if s['r_dps'] is not None else '',
                    'nz_g': round(s['nz_g'], 2) if s['nz_g'] is not None else '',
                    'fc_pitch_tgt_deg': round(s['fc_pitch_tgt_deg'], 2) if s['fc_pitch_tgt_deg'] is not None else '',
                    'fc_nav_pitch_deg': round(s['fc_nav_pitch_deg'], 2) if s['fc_nav_pitch_deg'] is not None else '',
                    'fc_nav_roll_deg': round(s['fc_nav_roll_deg'], 2) if s['fc_nav_roll_deg'] is not None else '',
                    'fc_nav_bearing_deg': round(s['fc_nav_bearing_deg'], 1) if s['fc_nav_bearing_deg'] is not None else '',
                    'fc_mode': s['fc_mode'],
                    'batt_v': round(s['batt_v'], 2) if s['batt_v'] is not None else '',
                    **{f'rcout{i+1}': (s['rcout'][i] if s['rcout'][i] is not None else '')
                       for i in range(4)},
                    'lat': s['lat'], 'lon': s['lon'],
                }
                writer.writerow(row)

                # ══════════════════════════════════════════════════
                #  健康看門狗 (每 2s) — 2026/09/08 實飛的教訓:
                #  視覺執行緒在 754.7s 靜默死亡, 主 log 卻跑到 1075.6s,
                #  飛完才發現影片和視覺資料缺了最後 5 分 21 秒.
                #  現在任何一條資料鏈斷掉都會【當場】在 console 跳出來.
                # ══════════════════════════════════════════════════
                if t - _vid_check_t >= 2.0:
                    _vid_check_t = t
                    # (a) 視覺執行緒: thread 死了, 或活著但卡住 (心跳過期)
                    if 'vision' not in _vid_warned:
                        _hb_age = time.time() - vision_hb.get('t', 0)
                        _dead = not vision_thread.is_alive()
                        _fatal = vision_hb.get('fatal')
                        if _dead or _fatal or _hb_age > 2.0:
                            _vid_warned.add('vision')
                            _why = (_fatal or ("執行緒已結束" if _dead
                                    else f"心跳停了 {_hb_age:.1f}s (可能卡在 YOLO/IO)"))
                            _w = (f"★★★ 視覺資料鏈中斷 @ t={t:.1f}s "
                                  f"(frame_idx={vision_hb.get('frame')}): {_why}"
                                  + (f" | 最後錯誤: {vision_hb.get('last_err')}"
                                     if vision_hb.get('last_err') else "")
                                  + "  ← 控制與 raw 錄影不受影響")
                            print("\n" + "!" * 64 + "\n" + _w + "\n" + "!" * 64)
                            event(_w)
                    # (b) 相機取像執行緒 (raw 錄影的命脈)
                    if 'camera' not in _vid_warned:
                        _gs = grabber.stats()
                        if _gs['age'] is not None and _gs['age'] > 3.0:
                            _vid_warned.add('camera')
                            _w = (f"★★★ 相機停止供幀 @ t={t:.1f}s "
                                  f"(已 {_gs['age']:.1f}s, 連續失敗 "
                                  f"{_gs['read_fail']}, 重開 {_gs['reopens']} 次)")
                            print("\n" + _w)
                            event(_w)
                    # (c) 兩支影片的寫入器
                    for _nm, _vw in (('annot', video_annot), ('raw', video_raw)):
                        if _vw is None or _nm in _vid_warned:
                            continue
                        _hok, _hmsg = _vw.health()
                        if not _hok:
                            _vid_warned.add(_nm)
                            _w = f"★★★ 錄影 [{_nm}] 已停止: {_hmsg}"
                            print("\n" + _w)
                            event(_w)

                if t - last_print >= 1.0:
                    last_print = t
                    _np = s['fc_nav_pitch_deg']
                    _nr = s['fc_nav_roll_deg']
                    _fc = (f" | FC nav_pitch={_np:+6.1f} nav_roll={_nr:+6.1f}"
                           if _np is not None else " | FC nav_* 無資料")
                    print(f"[{out['mode']:8s}] t={t:6.1f} AGL={agl or 0:6.1f} "
                          f"V={V:5.1f} pd_dist={vfields['predive_dist_m']:6.0f} "
                          f"det={int(vst.detected) if vst else 0} "
                          f"pitch={out['pitch_cmd_deg']:+6.1f} "
                          f"hdg={out['heading_cmd_deg']:6.1f} g={int(s['guided'])}"
                          + _fc)

                if sm.pullout_snapshot is not None and not scored['done']:
                    scored['done'] = True
                    sp = sm.pullout_snapshot
                    i_n, i_e, t_f = extrapolate_impact(
                        sp['ac_n'], sp['ac_e'], sp['agl'],
                        sp['vn'] or 0, sp['ve'] or 0, sp['vd'] or 0.1)
                    if i_n is not None:
                        md = miss_distance(i_n, i_e, t_f, sp['tgt_n'], sp['tgt_e'],
                                           sp['tgt_vn'], sp['tgt_ve'],
                                           sp['vn'] or 0, sp['ve'] or 0)
                        event(f"★★★ 成績: miss = {md['miss_total']:.2f} m "
                              f"(lon {md['miss_lon']:+.2f}, lat {md['miss_lat']:+.2f})")

            except Exception as e:
                err_count += 1
                if err_count <= 20:
                    event(f"★ 主迴圈例外 #{err_count}: {type(e).__name__}: {e}")
            rate.sleep()
    finally:
        try:
            bridge.stop()
        except Exception:
            pass
        event(f"程式結束 mode={sm.mode} errors={err_count}")
        _sync_run['on'] = False
        flags['quit'] = True
        time.sleep(0.3)
        try:
            grabber.stop()
        except Exception:
            pass
        if video_annot is not None:
            video_annot.release()
        if video_raw is not None:
            video_raw.release()
        gimbal.close()
        fcsv.flush(); os.fsync(fcsv.fileno()); fcsv.close()
        fvis.flush(); os.fsync(fvis.fileno()); fvis.close()

        # ══════════════════════════════════════════════════════
        #  ★ 錄影 / 資料鏈總結 — 飛完【立刻】知道資料完不完整,
        #    不用等回家對 CSV 才發現少了 5 分鐘.
        # ══════════════════════════════════════════════════════
        try:
            _dur = time.time() - t0
            _gs = grabber.stats()
            _vok = (vision_hb.get('fatal') is None
                    and vision_hb.get('errors', 0) == 0
                    and 'vision' not in _vid_warned)
            L = ["", "=" * 58, "            錄影 / 資料鏈 總結", "=" * 58,
                 f"  任務時長          : {_dur:.1f} s "
                 f"({int(_dur // 60)}:{_dur % 60:04.1f})",
                 f"  相機擷取幀數      : {_gs['captured']}"
                 + (f"  (重開 {_gs['reopens']} 次)" if _gs['reopens'] else ""),
                 f"  視覺處理幀數      : {vision_hb.get('frame', -1) + 1}"
                 f"   平均 {(vision_hb.get('frame', 0) + 1) / max(_dur, 1e-6):.1f} fps",
                 f"  跳過的重複影像    : {vision_hb.get('dup', 0)}"
                 f"   (相機比視覺迴圈慢時會出現, 正常)"]
            if video_raw is not None and video_raw.written > 1 and \
                    video_raw.t_first and video_raw.t_last:
                _cs = video_raw.t_last - video_raw.t_first
                if _cs > 1.0:
                    _cfps = video_raw.written / _cs
                    L.append(f"  相機實際 fps      : {_cfps:.1f}"
                             f"   (config CAM_FPS={cfg.CAM_FPS})")
                    if _cfps < 0.8 * cfg.CAM_FPS:
                        L.append(f"                      ★ 遠低於設定值 — "
                                 f"raw 影片時基會差 "
                                 f"{(cfg.CAM_FPS / _cfps - 1) * 100:.0f}%")
            for _nm, _vw in (('raw  ', video_raw), ('annot', video_annot)):
                if _vw is None:
                    L.append(f"  {_nm} 影片          : 未啟動")
                else:
                    _bk = _vw.enqueued - _vw.written - _vw.dropped
                    L.append(f"  {_nm} 影片          : 入列 {_vw.enqueued}, "
                             f"寫入 {_vw.written}, 丟棄 {_vw.dropped}, "
                             f"殘留 {_bk}"
                             + (f", 第 {_vw.died_at_t:.1f}s 停止"
                                if _vw.died_at_t is not None else "")
                             + ("  ★ 有錯誤 (見上)" if _vw.error else "  OK"))
                    if _nm == 'raw  ' and _vw.first_frame_id is not None:
                        L.append(f"                      MP4 第 0 幀 = 相機第 "
                                 f"{_vw.first_frame_id} 幀 "
                                 f"(後處理 raw_frame_idx = MP4幀號 + "
                                 f"{_vw.first_frame_id})")
                    if _vw.enqueued and _vw.written < 0.95 * _vw.enqueued:
                        L.append(f"                      ★ 只寫出 "
                                 f"{100.0 * _vw.written / _vw.enqueued:.0f}% "
                                 f"— 編碼器跟不上或中途死亡")
            if _vok:
                L.append("  視覺執行緒        : OK")
            else:
                L.append(f"  視覺執行緒        : ★★★ 異常 — "
                         f"最後 frame_idx={vision_hb.get('frame')}, "
                         f"例外 {vision_hb.get('errors', 0)} 次")
                if vision_hb.get('fatal'):
                    L.append(f"                      {vision_hb['fatal']}")
                if vision_hb.get('last_err'):
                    L.append(f"                      最後錯誤: "
                             f"{vision_hb['last_err']}")
                L.append(f"                      traceback → {err_log_path}")
            L += ["=" * 58, ""]
            _sum = "\n".join(L)
            print(_sum)
            for _ln in L:
                if _ln.strip():
                    event(_ln.strip())
        except Exception as _e:
            print(f"[flight] 總結產生失敗: {_e}")

        fev.close()
        print(f"[flight] log 已存 {csv_path}")
        print(f"[flight] 後處理: python3 postprocess_compare.py {log_dir}/{stem}")


if __name__ == '__main__':
    main()
