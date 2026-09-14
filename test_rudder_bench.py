#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_rudder_bench.py — 掛機驗證 SET_ATTITUDE_TARGET 的 yaw 欄位到底是什麼
═══════════════════════════════════════════════════════════════════
╔═══════════════════════════════════════════════════════════════╗
║ ⚠⚠⚠ 安全鐵則: 一定要先【拆槳】(remove propellers)!            ║
║     本程式送 thrust=0, 但飛控/失效保護行為不可 100% 預期.      ║
║     飛機請固定在架上或壓住, 舵面附近淨空.                      ║
╚═══════════════════════════════════════════════════════════════╝

要驗證的假設 (2026/09 分析):
  ArduPlane GUIDED 的 SET_ATTITUDE_TARGET, yaw 欄位「不是航向目標」,
  而是被 calc_nav_yaw_coordinated() 直接當成方向舵命令:
        forced_rpy_cd.z = degrees(q.get_euler_yaw()) * 100
        steering_control.rudder = constrain(forced_rpy_cd.z, -4500, +4500)
  也就是 45° = 滿舵. 送 heading 0~360 幾乎必定飽和 (打死).

  另外 attitude_bridge 的編碼鏈 (Q_CONV_INV + MAVROS ENU→NED) 對 yaw 有
  鏡像: 飛控抽出的 yaw ≈ 180 − 送出的 yaw. 本程式會逐點驗這件事.

本程式完全「不」使用 attitude_bridge 的送出迴圈 — 它自己直接發布
AttitudeTarget, 這樣測到的是飛控的真實行為, 不會被 bridge 的語意假設污染.
(四元數的組法則直接 import attitude_bridge 的函式, 確保跟實飛同一條鏈.)

需要:
  - 飛控通電 + MAVROS 連線
  - 飛控切到 GUIDED (否則 ArduPlane 會忽略 SET_ATTITUDE_TARGET)
  - 【拆槳】

用法:
  python3 test_rudder_bench.py --list-channels     # 先確認舵面在第幾通道
  python3 test_rudder_bench.py --test A            # 驗證 heading→舵量對照表
  python3 test_rudder_bench.py --test B            # 舵量線性/符號掃描
  python3 test_rudder_bench.py --test C            # 互動 (鍵盤微調)
  python3 test_rudder_bench.py --test D            # mask 驗證 (roll/yaw 交還飛控)
  python3 test_rudder_bench.py --test E            # ★V尾: 俯衝時還剩多少方向舵
  python3 test_rudder_bench.py --test A --rudder-ch 4 --aileron-ch 1

判讀:
  Test A 若「預測 rudder」與「實測 PWM 偏移」形狀一致 → 假設成立,
         yaw 欄位就是舵量, 全部 lateral 設計必須改.
  Test A 若 rudder 完全不動 → 假設不成立 (此韌體沒有這段), 回報給我.
"""

import argparse
import csv
import math
import os
import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ★ 直接沿用實飛那條編碼鏈, 不另外寫一份 (避免測的跟飛的不一樣)
from attitude_bridge import (
    euler_to_quaternion, quaternion_multiply, Q_CONV_INV,
)

try:
    import rospy
    from mavros_msgs.msg import State, RCOut, AttitudeTarget
    from sensor_msgs.msg import Imu
    ROS_OK = True
except ImportError:
    ROS_OK = False


SEND_HZ = 20.0
HOLD_PITCH_DEG = 0.0          # 掛機不需要壓機頭, 設 0 讓升降舵也好觀察
RUDDER_FULL_CD = 4500.0       # ArduPlane 滿舵 = 4500 centideg


# ══════════════════════════════════════════════════════════════════
#  預測: 模擬 MAVROS 的 ENU→NED 轉換, 算出飛控會抽到什麼 euler
# ══════════════════════════════════════════════════════════════════
# MAVROS setpoint_raw attitude plugin:
#   q_fcu = NED_ENU_Q * (q_ros * AIRCRAFT_BASELINK_Q)
NED_ENU_Q = euler_to_quaternion(math.pi, 0.0, math.pi / 2)
AIRCRAFT_BASELINK_Q = euler_to_quaternion(math.pi, 0.0, 0.0)


def _quat_to_euler_ned(q):
    """(x,y,z,w) → (roll, pitch, yaw) 度, 標準航空 ZYX (飛控端的抽法)."""
    x, y, z, w = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def build_quat(roll_deg, pitch_deg, yaw_deg, frame_fix=True):
    """完全照 attitude_bridge._publish_att 的組法 (含 pitch 取負)."""
    q_ned = euler_to_quaternion(math.radians(roll_deg),
                                -math.radians(pitch_deg),
                                math.radians(yaw_deg))
    if frame_fix:
        return quaternion_multiply(Q_CONV_INV, q_ned)
    return q_ned


def predict_fcu(roll_deg, pitch_deg, yaw_deg, frame_fix=True):
    """回傳 (飛控抽到的 roll, pitch, yaw 度, 預測 rudder centideg 夾限後)."""
    q_ros = build_quat(roll_deg, pitch_deg, yaw_deg, frame_fix)
    q_fcu = quaternion_multiply(NED_ENU_Q,
                               quaternion_multiply(q_ros, AIRCRAFT_BASELINK_Q))
    r, p, y = _quat_to_euler_ned(q_fcu)
    rud = max(-RUDDER_FULL_CD, min(RUDDER_FULL_CD, y * 100.0))
    return r, p, y, rud


def yaw_for_rudder(delta_deg, frame_fix=True):
    """反解: 想要飛控抽到 yaw = delta_deg (即舵量 delta*100 cd), 該送多少 yaw.

    對現行編碼鏈是 180 − delta, 但這裡用數值反解, 編碼一改自動跟著對.
    """
    best, best_err = 0.0, 1e9
    # 粗掃 + 細掃 (夠用, 不需要解析解)
    for coarse in range(0, 3600):
        cand = coarse / 10.0
        _, _, y, _ = predict_fcu(0.0, 0.0, cand, frame_fix)
        err = abs((y - delta_deg + 180.0) % 360.0 - 180.0)
        if err < best_err:
            best_err, best = err, cand
    for fine in range(-100, 101):
        cand = (best + fine / 100.0) % 360.0
        _, _, y, _ = predict_fcu(0.0, 0.0, cand, frame_fix)
        err = abs((y - delta_deg + 180.0) % 360.0 - 180.0)
        if err < best_err:
            best_err, best = err, cand
    return best % 360.0


# ══════════════════════════════════════════════════════════════════
#  MAVROS I/O
# ══════════════════════════════════════════════════════════════════
class BenchIO:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = ''
        self.guided = False
        self.armed = False
        self.rcout = []
        self.roll_deg = None
        self.pitch_deg = None
        self.p_dps = None
        self.r_dps = None
        rospy.Subscriber("/mavros/state", State, self._cb_state)
        rospy.Subscriber("/mavros/rc/out", RCOut, self._cb_rcout)
        rospy.Subscriber("/mavros/imu/data", Imu, self._cb_imu)

    def _cb_state(self, m):
        with self.lock:
            self.mode = m.mode
            self.guided = (m.mode == 'GUIDED')
            self.armed = m.armed

    def _cb_rcout(self, m):
        with self.lock:
            self.rcout = list(m.channels)

    def _cb_imu(self, m):
        with self.lock:
            q = m.orientation
            sinr = 2.0 * (q.w * q.x + q.y * q.z)
            cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            self.roll_deg = math.degrees(math.atan2(sinr, cosr))
            sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
            self.pitch_deg = -math.degrees(math.asin(sinp))
            self.p_dps = math.degrees(m.angular_velocity.x)
            self.r_dps = -math.degrees(m.angular_velocity.z)

    def snap(self):
        with self.lock:
            return {
                'mode': self.mode, 'guided': self.guided, 'armed': self.armed,
                'rcout': list(self.rcout),
                'roll_deg': self.roll_deg, 'pitch_deg': self.pitch_deg,
                'p_dps': self.p_dps, 'r_dps': self.r_dps,
            }


class AttitudeStreamer:
    """20Hz 持續發布 AttitudeTarget. 送什麼由 set() 決定."""

    # ArduPlane XOR 語意下的幾組 mask (見分析):
    #   7   = IGNORE_ROLL/PITCH/YAW_RATE → roll+pitch+yaw 全部強制 (實飛現況)
    #   129 = IGNORE_ATTITUDE|IGNORE_ROLL_RATE → roll 交還飛控, pitch+yaw 強制
    #   133 = IGNORE_ATTITUDE|ROLL_RATE|YAW_RATE → 只強制 pitch
    MASK_ALL = 7            # roll+pitch+yaw 全鎖
    MASK_ROLL_PITCH = 132   # roll+pitch 鎖, yaw 交還飛控 (PULLOUT/ABORT)
    MASK_NO_ROLL = 129      # roll 交還, pitch+yaw 鎖
    MASK_PITCH_ONLY = 133   # 只鎖 pitch, roll+yaw 都交還

    def __init__(self, frame_fix=True):
        self.pub = rospy.Publisher("/mavros/setpoint_raw/attitude",
                                   AttitudeTarget, queue_size=1)
        self.lock = threading.Lock()
        self.roll = 0.0
        self.pitch = HOLD_PITCH_DEG
        self.yaw = 180.0            # 180 → 飛控抽到 0 → 舵量 0 (安全起點)
        self.mask = self.MASK_ALL
        self.frame_fix = frame_fix
        self.sent = 0
        self._run = False
        self._t = None

    def set(self, roll=None, pitch=None, yaw=None, mask=None):
        with self.lock:
            if roll is not None:
                self.roll = float(roll)
            if pitch is not None:
                self.pitch = float(pitch)
            if yaw is not None:
                self.yaw = float(yaw) % 360.0
            if mask is not None:
                self.mask = int(mask)

    def start(self):
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._run = False
        if self._t:
            self._t.join(timeout=1.0)

    def _loop(self):
        rate = rospy.Rate(SEND_HZ)
        while self._run and not rospy.is_shutdown():
            with self.lock:
                r, p, y, m = self.roll, self.pitch, self.yaw, self.mask
            q = build_quat(r, p, y, self.frame_fix)
            msg = AttitudeTarget()
            msg.header.stamp = rospy.Time.now()
            msg.type_mask = m
            msg.orientation.x, msg.orientation.y = q[0], q[1]
            msg.orientation.z, msg.orientation.w = q[2], q[3]
            msg.thrust = 0.0        # ★ 掛機一律 0 油門
            self.pub.publish(msg)
            self.sent += 1
            rate.sleep()


# ══════════════════════════════════════════════════════════════════
#  共用工具
# ══════════════════════════════════════════════════════════════════
def _ch(rcout, idx1):
    """取 1-based 通道 PWM; 沒有就回 None."""
    if not rcout or idx1 < 1 or idx1 > len(rcout):
        return None
    return rcout[idx1 - 1]


def settle_and_read(io, streamer, args, settle_s=1.5, samples=10):
    """等舵面穩定, 取多筆 rcout 平均 (降低抖動)."""
    time.sleep(settle_s)
    acc = {}
    n = 0
    roll_acc = 0.0
    for _ in range(samples):
        s = io.snap()
        for name, idx in (('rud', args.rudder_ch), ('ail', args.aileron_ch),
                          ('elev', args.elevator_ch)):
            v = _ch(s['rcout'], idx)
            if v is not None:
                acc[name] = acc.get(name, 0) + v
        if s['roll_deg'] is not None:
            roll_acc += s['roll_deg']
        n += 1
        time.sleep(0.05)
    out = {k: v / n for k, v in acc.items()}
    out['roll_deg'] = roll_acc / n if n else None
    return out


def require_guided(io, warn_only=False):
    s = io.snap()
    if s['guided']:
        return True
    print(f"\n  ⚠ 飛控目前模式 = {s['mode'] or '(未知)'}, 不是 GUIDED.")
    print("    ArduPlane 只在 GUIDED 才理會 SET_ATTITUDE_TARGET.")
    if warn_only:
        return False
    input("    請用遙控器/地面站切到 GUIDED, 然後按 Enter 繼續 ...")
    return io.snap()['guided']


def open_log(tag):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_logs')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"rudder_bench_{datetime.now():%Y%m%d_%H%M%S}_{tag}.csv")
    f = open(p, 'w', newline='', buffering=1)
    return f, p


# ══════════════════════════════════════════════════════════════════
#  Test A — heading 對照表驗證
# ══════════════════════════════════════════════════════════════════
def test_A(io, st, args):
    print("\n" + "=" * 68)
    print("Test A — 驗證 heading_cmd → 方向舵 對照表")
    print("=" * 68)
    print("送出實飛用的 heading_cmd 值 (0~360), 量測方向舵 PWM.")
    print("預期 (若假設成立): 除了 135~225 之間, 其餘全部滿舵不變.")
    print("若假設不成立: 方向舵幾乎不動, 或跟 heading 呈連續關係.\n")

    if not require_guided(io):
        print("  仍不是 GUIDED, 中止 Test A.")
        return

    headings = [180, 176, 170, 160, 135, 90, 45, 0, 315, 270, 225, 200, 180]
    f, path = open_log('A')
    w = csv.writer(f)
    w.writerow(['heading_cmd', 'pred_fcu_yaw_deg', 'pred_rudder_cd',
                'pred_saturated', 'rud_pwm', 'rud_delta_pwm',
                'ail_pwm', 'elev_pwm', 'roll_deg'])

    # 先取中立基準 (yaw=180 → 預測舵量 0)
    st.set(roll=0.0, pitch=HOLD_PITCH_DEG, yaw=180.0, mask=st.MASK_ALL)
    base = settle_and_read(io, st, args, settle_s=2.5)
    rud0 = base.get('rud')
    if rud0 is None:
        print(f"  ✗ 讀不到通道 {args.rudder_ch} 的 RCOut — 用 --list-channels 確認")
        f.close()
        return
    print(f"  中立基準 (yaw=180): 方向舵 PWM = {rud0:.0f}\n")

    print(f"  {'heading':>8} {'預測yaw':>9} {'預測舵量':>9} {'實測PWM':>9} "
          f"{'ΔPWM':>8} {'副翼':>8} {'roll':>7}")
    print("  " + "-" * 64)
    rows = []
    for h in headings:
        st.set(yaw=h)
        m = settle_and_read(io, st, args)
        _, _, py, prud = predict_fcu(0.0, HOLD_PITCH_DEG, h)
        d = m['rud'] - rud0
        sat = abs(py * 100.0) > RUDDER_FULL_CD
        print(f"  {h:8.0f} {py:9.1f} {prud:9.0f}{'*' if sat else ' '} "
              f"{m['rud']:9.0f} {d:+8.0f} {m.get('ail', 0):8.0f} "
              f"{m['roll_deg'] if m['roll_deg'] is not None else 0:7.1f}")
        w.writerow([h, round(py, 2), round(prud, 0), int(sat),
                    round(m['rud'], 1), round(d, 1),
                    round(m.get('ail', 0), 1), round(m.get('elev', 0), 1),
                    round(m['roll_deg'], 2) if m['roll_deg'] is not None else ''])
        rows.append((h, prud, d))
    print("  (* = 預測已飽和)")
    f.close()

    # ── 判讀 ──
    # ★ 2026/09/11 修正: 舊版把正負飽和區混在一起算離散度, 必定誤判.
    #   正飽和 (+305) 與負飽和 (-472) 本來就差很多, 要【分開】看.
    span = max(abs(d) for _, _, d in rows)
    print(f"\n  方向舵 PWM 最大偏移 = {span:.0f}")
    if span < 20:
        print("  → 方向舵幾乎沒動. 假設【不成立】(或此韌體沒這段 / 通道給錯).")
        print("    請確認 --rudder-ch 對不對, 再把這份 log 給我.")
        return
    pos = [d for h, prud, d in rows if prud >= RUDDER_FULL_CD]
    neg = [d for h, prud, d in rows if prud <= -RUDDER_FULL_CD]
    ok_pos = (len(pos) < 2) or (max(pos) - min(pos)) <= max(10.0, 0.05 * span)
    ok_neg = (len(neg) < 2) or (max(neg) - min(neg)) <= max(10.0, 0.05 * span)
    lin = [(prud, d) for h, prud, d in rows if abs(prud) < RUDDER_FULL_CD]
    mono = all((a[0] - b[0]) * (a[1] - b[1]) >= 0
               for a in lin for b in lin)
    if ok_pos and ok_neg and mono:
        print("  → 假設【成立】:")
        print(f"     · 正飽和區 {len(pos)} 點全部一致 (離散 "
              f"{(max(pos) - min(pos)) if len(pos) > 1 else 0:.0f} PWM)")
        print(f"     · 負飽和區 {len(neg)} 點全部一致 (離散 "
              f"{(max(neg) - min(neg)) if len(neg) > 1 else 0:.0f} PWM)")
        print("     · 未飽和區與預測舵量單調同向")
        print("     yaw 欄位 = 方向舵命令 (centideg), 45° 滿舵.")
        print("     ※ 正負飽和值不同是正常的 — SERVO TRIM 不在幾何中點.")
    else:
        why = []
        if not ok_pos:
            why.append("正飽和區不一致")
        if not ok_neg:
            why.append("負飽和區不一致")
        if not mono:
            why.append("未飽和區非單調")
        print(f"  → 有反應但形狀跟預測不一致 ({', '.join(why)}), 需看 log 細判.")
    print(f"  log → {path}")


# ══════════════════════════════════════════════════════════════════
#  Test B — 舵量線性 / 符號掃描
# ══════════════════════════════════════════════════════════════════
def test_B(io, st, args):
    print("\n" + "=" * 68)
    print("Test B — 舵量線性與符號掃描 (δr = −40° … +40°)")
    print("=" * 68)
    print("用反解的 yaw 值下達指定舵量, 確認:")
    print("  1. 舵面方向對不對 (正 δr 往哪邊打?)")
    print("  2. |δr| ≤ 45 區間是否線性")
    print("  3. |δr| > 45 是否飽和\n")
    print("★ 請站在機尾正後方, 記錄正 δr 時方向舵往【左還右】—— 這決定符號.\n")

    if not require_guided(io):
        print("  仍不是 GUIDED, 中止 Test B.")
        return

    deltas = [0, 3, 5, 10, 20, 30, 45, 60, 0, -3, -5, -10, -20, -30, -45, -60, 0]
    f, path = open_log('B')
    w = csv.writer(f)
    w.writerow(['delta_r_deg', 'yaw_cmd_deg', 'pred_rudder_cd',
                'rud_pwm', 'rud_delta_pwm', 'ail_pwm', 'roll_deg', 'r_dps'])

    st.set(roll=0.0, pitch=HOLD_PITCH_DEG, yaw=yaw_for_rudder(0.0),
           mask=st.MASK_ALL)
    base = settle_and_read(io, st, args, settle_s=2.5)
    rud0 = base.get('rud')
    if rud0 is None:
        print(f"  ✗ 讀不到通道 {args.rudder_ch} 的 RCOut")
        f.close()
        return
    print(f"  中立基準: 方向舵 PWM = {rud0:.0f}\n")
    print(f"  {'δr(°)':>7} {'yaw_cmd':>9} {'預測cd':>8} {'實測PWM':>9} "
          f"{'ΔPWM':>8} {'副翼':>8} {'roll':>7}")
    print("  " + "-" * 60)

    for d in deltas:
        yc = yaw_for_rudder(float(d))
        st.set(yaw=yc)
        m = settle_and_read(io, st, args, settle_s=1.2)
        pred = max(-RUDDER_FULL_CD, min(RUDDER_FULL_CD, d * 100.0))
        dp = m['rud'] - rud0
        s = io.snap()
        print(f"  {d:7.0f} {yc:9.1f} {pred:8.0f} {m['rud']:9.0f} {dp:+8.0f} "
              f"{m.get('ail', 0):8.0f} "
              f"{m['roll_deg'] if m['roll_deg'] is not None else 0:7.1f}")
        w.writerow([d, round(yc, 2), round(pred, 0), round(m['rud'], 1),
                    round(dp, 1), round(m.get('ail', 0), 1),
                    round(m['roll_deg'], 2) if m['roll_deg'] is not None else '',
                    round(s['r_dps'], 1) if s['r_dps'] is not None else ''])

    st.set(yaw=yaw_for_rudder(0.0))
    f.close()
    print(f"\n  舵面已回中. log → {path}")
    print("  判讀重點:")
    print("    - ΔPWM 對 δr 在 ±45 內應接近線性")
    print("    - δr=±60 的 ΔPWM 應與 ±45 相同 (飽和)")
    print("    - 副翼欄若跟著大幅變動 → 表示 roll 也被污染, 要回報")


# ══════════════════════════════════════════════════════════════════
#  Test C — 互動微調
# ══════════════════════════════════════════════════════════════════
class Keyboard:
    def __init__(self):
        self.key = None
        self.run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self.run:
                self.key = sys.stdin.read(1)
                if self.key == 'q':
                    self.run = False
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    def get(self):
        k = self.key
        self.key = None
        return k


def test_C(io, st, args):
    print("\n" + "=" * 68)
    print("Test C — 互動微調 (即時看舵面)")
    print("=" * 68)
    print("  a/d = δr −1° / +1°     A/D = δr −5° / +5°")
    print("  w/s = pitch +1° / −1°  z   = 全部歸零")
    print("  m   = 切換 type_mask (7 → 129 → 133 → 7)")
    print("  q   = 結束\n")

    require_guided(io, warn_only=True)
    delta = 0.0
    pitch = HOLD_PITCH_DEG
    masks = [st.MASK_ALL, st.MASK_NO_ROLL, st.MASK_PITCH_ONLY]
    mask_names = {7: '7 (roll+pitch+yaw 全鎖)',
                  129: '129 (roll 交還飛控)',
                  133: '133 (只鎖 pitch)'}
    mi = 0
    st.set(roll=0.0, pitch=pitch, yaw=yaw_for_rudder(delta), mask=masks[mi])

    kb = Keyboard()
    last = 0.0
    try:
        while kb.run and not rospy.is_shutdown():
            k = kb.get()
            if k in ('a', 'd', 'A', 'D', 'w', 's', 'z', 'm'):
                if k == 'a':
                    delta -= 1.0
                elif k == 'd':
                    delta += 1.0
                elif k == 'A':
                    delta -= 5.0
                elif k == 'D':
                    delta += 5.0
                elif k == 'w':
                    pitch += 1.0
                elif k == 's':
                    pitch -= 1.0
                elif k == 'z':
                    delta, pitch = 0.0, 0.0
                elif k == 'm':
                    mi = (mi + 1) % len(masks)
                    print(f"\n  → type_mask = {mask_names[masks[mi]]}")
                delta = max(-60.0, min(60.0, delta))
                pitch = max(-20.0, min(20.0, pitch))
                st.set(pitch=pitch, yaw=yaw_for_rudder(delta), mask=masks[mi])
            now = time.time()
            if now - last >= 0.4:
                last = now
                s = io.snap()
                rud = _ch(s['rcout'], args.rudder_ch)
                ail = _ch(s['rcout'], args.aileron_ch)
                ele = _ch(s['rcout'], args.elevator_ch)
                print(f"\r  δr={delta:+6.1f}° pitch={pitch:+5.1f}° "
                      f"mask={masks[mi]:3d} | mode={s['mode']:>8s} "
                      f"rud={rud if rud else 0:4d} ail={ail if ail else 0:4d} "
                      f"ele={ele if ele else 0:4d} "
                      f"roll={s['roll_deg'] if s['roll_deg'] is not None else 0:+6.1f}°",
                      end='', flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        kb.run = False
        st.set(roll=0.0, pitch=0.0, yaw=yaw_for_rudder(0.0), mask=st.MASK_ALL)
        print("\n  已歸零.")


# ══════════════════════════════════════════════════════════════════
#  Test D — type_mask 驗證
# ══════════════════════════════════════════════════════════════════
def test_D(io, st, args):
    print("\n" + "=" * 68)
    print("Test D — type_mask 行為驗證")
    print("=" * 68)
    print("驗證三組 mask 下, 飛控是否真的把對應的軸交還給自己:")
    print("  mask=7   : roll+pitch+yaw 全鎖 (實飛現況)")
    print("  mask=129 : roll 交還飛控 (SEARCH 想用的)")
    print("  mask=133 : 只鎖 pitch, roll+yaw 都交還")
    print("\n★ 掛機時飛機不動, 飛控的自主 roll/yaw 輸出不見得會有明顯反應,")
    print("  這個測試主要是確認【送出的 mask 不會讓飛控整個拒收】.")
    print("  真正的差別要在空中才看得出來.\n")

    if not require_guided(io):
        print("  仍不是 GUIDED, 中止 Test D.")
        return

    f, path = open_log('D')
    w = csv.writer(f)
    w.writerow(['mask', 'roll_cmd', 'pitch_cmd', 'delta_r_deg',
                'rud_pwm', 'ail_pwm', 'elev_pwm', 'roll_deg'])

    # ★ 2026/09/11: settle 從 2.0 拉到 4.0 — 飛控的 forced-roll 逾時是 3 秒,
    #   等不夠久會量到上一組的殘留 (舊版 mask=129 那幾列就是被污染的).
    #   同時補上 mask=132 (PULLOUT/ABORT 用: roll+pitch 鎖, yaw 交還).
    for mask in (st.MASK_ALL, st.MASK_ROLL_PITCH,
                 st.MASK_NO_ROLL, st.MASK_PITCH_ONLY):
        for (roll_cmd, pitch_cmd, dr) in ((0.0, 5.0, 0.0), (0.0, 5.0, 10.0),
                                          (10.0, 5.0, 0.0)):
            st.set(roll=roll_cmd, pitch=pitch_cmd,
                   yaw=yaw_for_rudder(dr), mask=mask)
            m = settle_and_read(io, st, args, settle_s=4.0)
            print(f"  mask={mask:3d} roll_cmd={roll_cmd:+5.1f} "
                  f"pitch_cmd={pitch_cmd:+5.1f} δr={dr:+5.1f} → "
                  f"rud={m['rud']:.0f} ail={m.get('ail', 0):.0f} "
                  f"ele={m.get('elev', 0):.0f}")
            w.writerow([mask, roll_cmd, pitch_cmd, dr,
                        round(m['rud'], 1), round(m.get('ail', 0), 1),
                        round(m.get('elev', 0), 1),
                        round(m['roll_deg'], 2) if m['roll_deg'] is not None else ''])
    st.set(roll=0.0, pitch=0.0, yaw=yaw_for_rudder(0.0), mask=st.MASK_ALL)
    f.close()
    print(f"\n  判讀: mask=7 時 roll_cmd=10 應讓副翼明顯偏; mask=133 時應不受")
    print(f"        roll_cmd 影響 (副翼由飛控自己決定).")
    print(f"  log → {path}")


# ══════════════════════════════════════════════════════════════════
#  Test E — V 尾: 俯衝 pitch 下還剩多少方向舵權限
# ══════════════════════════════════════════════════════════════════
def test_E(io, st, args):
    """V 尾 (ruddervator) 兩片舵面共用行程:
         舵面A = elevator + rudder
         舵面B = elevator − rudder     (各自夾限在 ±4500)
       真正產生【偏航力矩】的是兩片的【差動量】= A − B = 2×rudder.
       但只要其中一片因為 pitch 太大而撞到端點, 差動量就會被砍掉,
       方向舵權限跟著消失, 而且殘餘的舵量會變成不想要的俯仰擾動.

       本測試: 在不同 pitch 命令下量同一個 δr 造成的差動量,
               以 pitch=0 當 100%, 算出俯衝時還剩幾成.
    """
    print("\n" + "=" * 68)
    print("Test E — V 尾: 俯衝 pitch 下的方向舵剩餘權限")
    print("=" * 68)
    print("★ 這個測試會下到 pitch = −28.7° (實飛 DIVE 的命令值),")
    print("  舵面會大幅偏轉並可能撞到端點. 確認手指遠離舵面與連桿.\n")
    if not require_guided(io):
        print("  仍不是 GUIDED, 中止 Test E.")
        return
    if input("  準備好了嗎? 輸入 yes 繼續: ").strip().lower() != 'yes':
        print("  已取消.")
        return

    pitches = [0.0, -10.0, -20.0, -28.7]
    probe = 5.0            # 探測用舵量 (度)
    f, path = open_log('E')
    w = csv.writer(f)
    w.writerow(['pitch_cmd', 'delta_r_deg', 'chA_pwm', 'chB_pwm',
                'dA', 'dB', 'differential', 'authority_pct'])

    print(f"  探測舵量 δr = ±{probe:.0f}°;  A=通道{args.rudder_ch} "
          f"B=通道{args.elevator_ch}\n")
    print(f"  {'pitch':>7} {'δr':>5} {'A(PWM)':>8} {'B(PWM)':>8} "
          f"{'ΔA':>6} {'ΔB':>6} {'差動':>7} {'權限':>7}")
    print("  " + "-" * 62)

    ref_diff = None
    results = []
    for p in pitches:
        st.set(roll=0.0, pitch=p, yaw=yaw_for_rudder(0.0), mask=st.MASK_ALL)
        base = settle_and_read(io, st, args, settle_s=3.0)
        bA, bB = base.get('rud', 0.0), base.get('elev', 0.0)
        print(f"  {p:7.1f} {0.0:5.1f} {bA:8.0f} {bB:8.0f} "
              f"{0:6.0f} {0:6.0f} {'--':>7} {'基準':>7}")
        w.writerow([p, 0.0, round(bA, 1), round(bB, 1), 0, 0, '', ''])
        for d in (probe, -probe):
            st.set(yaw=yaw_for_rudder(d))
            m = settle_and_read(io, st, args, settle_s=2.0)
            dA = m.get('rud', 0.0) - bA
            dB = m.get('elev', 0.0) - bB
            diff = dA - dB          # ★ 差動量 = 真正的偏航輸入
            if p == 0.0 and d == probe:
                ref_diff = abs(diff) if abs(diff) > 1e-6 else None
            pct = (100.0 * abs(diff) / ref_diff) if ref_diff else float('nan')
            print(f"  {p:7.1f} {d:5.1f} {m.get('rud', 0):8.0f} "
                  f"{m.get('elev', 0):8.0f} {dA:+6.0f} {dB:+6.0f} "
                  f"{diff:+7.0f} {pct:6.0f}%")
            w.writerow([p, d, round(m.get('rud', 0), 1),
                        round(m.get('elev', 0), 1), round(dA, 1),
                        round(dB, 1), round(diff, 1),
                        ('' if ref_diff is None else round(pct, 1))])
            results.append((p, d, dA, dB, diff, pct))
    st.set(roll=0.0, pitch=0.0, yaw=yaw_for_rudder(0.0), mask=st.MASK_ALL)
    f.close()

    # ── 判讀 ──
    print("\n  判讀:")
    worst = {}
    for p, d, dA, dB, diff, pct in results:
        if pct == pct:      # not NaN
            worst[p] = min(worst.get(p, 1e9), pct)
    for p in pitches:
        if p in worst:
            v = worst[p]
            tag = ("OK" if v >= 85 else
                   "★ 明顯衰退" if v >= 50 else
                   "★★★ 幾乎失效")
            print(f"    pitch={p:6.1f}° → 方向舵權限剩 {v:5.0f}%   {tag}")
    dive = worst.get(-28.7)
    print()
    if dive is None:
        print("    (資料不足, 看 CSV)")
    elif dive >= 85:
        print("    → V 尾在俯衝姿態下沒有明顯排擠, DIVE 的方向舵迴路可用.")
    elif dive >= 50:
        print("    → 俯衝時方向舵權限衰退. DIVE 的增益需要按 pitch 補償,")
        print("       或把橫向修正的責任往 SEARCH/PRE_DIVE 移.")
    else:
        print("    → ★★★ 俯衝時方向舵幾乎沒有作用. DIVE 段用方向舵做橫向")
        print("       修正【不可行】, 不是調參能解決的. 另外要注意: 舵面單邊")
        print("       飽和時, 方向舵命令會變成不對稱的俯仰擾動 — 在 30° 俯衝")
        print("       低空階段這是安全問題, DIVE 的舵量限幅要壓到最小或歸零.")
    print(f"\n  log → {path}")
    print("  另外請順手查參數: MIXING_GAIN (V 尾混控增益).")
    print("    0.5 = 升降舵與方向舵各分一半行程 (不會互相排擠)")
    print("    1.0 = 升降舵單獨就能吃滿行程 (俯衝時方向舵必然被擠掉)")
    print("    由 Test B 反推你的值接近 1.0.")


# ══════════════════════════════════════════════════════════════════
def list_channels(io):
    print("\n觀察 RCOut 各通道 10 秒 — 請同時用遙控器動舵面, 看哪一格跟著變.")
    print("(ArduPlane 預設: 1=副翼 2=升降舵 3=油門 4=方向舵)\n")
    t0 = time.time()
    while time.time() - t0 < 10.0 and not rospy.is_shutdown():
        s = io.snap()
        if s['rcout']:
            cols = ' '.join(f"ch{i+1}={v:4d}" for i, v in
                            enumerate(s['rcout'][:8]))
            print(f"\r  {cols}  mode={s['mode']}", end='', flush=True)
        time.sleep(0.2)
    print("\n")


def print_prediction_table():
    print("\n離線預測表 (不需要飛控, 純算目前編碼鏈會送出什麼):")
    print(f"  {'heading_cmd':>12} {'飛控抽到yaw':>12} {'rudder(cd)':>11}  飽和?")
    print("  " + "-" * 48)
    for h in (0, 45, 90, 135, 176, 180, 200, 225, 270, 315, 350):
        _, _, y, rud = predict_fcu(0.0, 0.0, h)
        sat = '★打死' if abs(y * 100.0) > RUDDER_FULL_CD else ''
        print(f"  {h:12.0f} {y:12.1f} {rud:11.0f}  {sat}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', choices=['A', 'B', 'C', 'D', 'E'],
                    help='要跑哪個測試')
    ap.add_argument('--list-channels', action='store_true',
                    help='只印 RCOut 各通道, 用來確認舵面在第幾通道')
    ap.add_argument('--predict-only', action='store_true',
                    help='不連飛控, 只印離線預測表')
    ap.add_argument('--rudder-ch', type=int, default=4, help='方向舵 RCOut 通道')
    ap.add_argument('--aileron-ch', type=int, default=1, help='副翼 RCOut 通道')
    ap.add_argument('--elevator-ch', type=int, default=2, help='升降舵 RCOut 通道')
    ap.add_argument('--no-frame-fix', action='store_true',
                    help='關閉 APPLY_MAVROS_FRAME_FIX (對照用)')
    ap.add_argument('--skip-safety', action='store_true',
                    help='跳過拆槳確認 (只在 --predict-only / SITL 用)')
    args = ap.parse_args()

    if args.predict_only:
        print_prediction_table()
        return

    if not ROS_OK:
        print("rospy / mavros_msgs 不可用 — 本程式要在機載電腦 (ROS 環境) 跑.")
        print("可以先用 --predict-only 看離線預測表.")
        sys.exit(1)

    print(__doc__)
    if not args.skip_safety:
        print("=" * 68)
        ans = input("★ 已確認【拆槳】、飛機已固定、舵面附近淨空了嗎? 輸入 yes 繼續: ")
        if ans.strip().lower() != 'yes':
            print("未確認, 中止.")
            sys.exit(0)

    rospy.init_node('test_rudder_bench', anonymous=True)
    io = BenchIO()
    print("\n等待 MAVROS ...")
    t0 = time.time()
    while time.time() - t0 < 10.0:
        if io.snap()['mode']:
            break
        time.sleep(0.2)
    s = io.snap()
    if not s['mode']:
        print("  ✗ 收不到 /mavros/state — 檢查 MAVROS 有沒有跑.")
        sys.exit(1)
    print(f"  ✓ 飛控模式 = {s['mode']}, armed = {s['armed']}")

    if args.list_channels:
        list_channels(io)
        return

    print_prediction_table()

    st = AttitudeStreamer(frame_fix=not args.no_frame_fix)
    st.start()
    print(f"[Bench] 姿態串流啟動 @ {SEND_HZ:.0f}Hz, thrust=0, "
          f"frame_fix={'ON' if not args.no_frame_fix else 'OFF'}")
    time.sleep(1.0)

    try:
        if args.test == 'A':
            test_A(io, st, args)
        elif args.test == 'B':
            test_B(io, st, args)
        elif args.test == 'C':
            test_C(io, st, args)
        elif args.test == 'D':
            test_D(io, st, args)
        elif args.test == 'E':
            test_E(io, st, args)
        else:
            print("\n沒指定 --test, 只做了連線檢查. 建議順序: A → B → E → D → C")
    except KeyboardInterrupt:
        print("\n中止.")
    finally:
        st.set(roll=0.0, pitch=0.0, yaw=yaw_for_rudder(0.0), mask=st.MASK_ALL)
        time.sleep(0.5)
        st.stop()
        print(f"[Bench] 結束, 共送出 {st.sent} 筆姿態命令.")


if __name__ == '__main__':
    main()
