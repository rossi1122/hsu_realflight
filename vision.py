# -*- coding: utf-8 -*-
"""
vision.py — 視覺前端 (vision_dive)
═══════════════════════════════════════════════════════════════════
資料流 (每幀):
  HDMI 擷取 (FrameGrabber, 永遠拿最新幀)
    → YOLO 偵測 (可能多框)
    → 選框 (select_detection): 首次截獲取「離畫面中心最近」,
                               鎖定後取「離 DKF 預測最近」+ 像素閘門
    → 地理閘門 (geo_gate): 偵測反投影的 NED 位置須落在預期目標半徑內
    → DKF (DKFTrackerGimbal): 延遲補償 + ego comp + coast 補位
    → 射線投影測距 (RayProjectionDistanceEstimator): slant/ground/tgt NED
  輸出一個 VisionState (給控制迴路 + CSV + overlay)

★ 一個目標: 選框永遠只挑一個, 別的目標 conf 再高也搶不走 (見 select_detection).
★ DKF 延遲補償: use_yolo_when_available=False → PID 吃的是「延遲補償後的
  當前時刻估測」而非落後 80ms 的 raw 量測 (見 README「影像延遲補償」).
★ DIVE 段 ego comp: 相機 FPV body-fixed, 餵「機身姿態」進 gimbal_*_deg 參數,
  DKF 粗近似路徑的 du=fx·Δyaw / dv=fy·Δpitch 即變成機身姿態差補償 (一行不改).
"""

import math
import threading
import time

import numpy as np

from dkf_tracker_gimbal import DKFTrackerGimbal
from ray_projection_distance import (
    RayProjectionDistanceEstimator,
    convert_mavros_enu_to_ned_attitude,
    convert_siyi_gimbal_to_estimator,
)


# ══════════════════════════════════════════════════════════════════
#  取像: 背景 thread 永遠拿最新幀 (吃掉 V4L2 排隊延遲)
# ══════════════════════════════════════════════════════════════════
class FrameGrabber:
    """背景取像. ★2026/09/09: raw 錄影搬到這條執行緒 (與 YOLO 解耦).

    原因: 舊架構 raw 影片是在 vision_loop 裡寫的, 只要 YOLO/DKF/雲台/CSV
    任一步丟例外, vision_loop 執行緒就死, raw 影片跟著停 —— 2026/09/08
    實飛就是這樣: 主 log 跑到 1075.6s, 視覺只到 754.7s, 影片缺了最後 5 分鐘.
    現在 raw 錄影只依賴「相機讀得到幀」這一件事, YOLO 掛掉不影響它.

    對齊: 每幀給一個 frame_id, vision log 記 raw_frame_idx, 後處理靠它對齊
    (不再假設 raw 影片第 N 幀 = 視覺 log 第 N 列).
    """

    def __init__(self, cfg):
        import cv2
        self.cfg = cfg
        self.cv2 = cv2
        self.cap = None
        self.frame = None
        self.frame_id = -1              # ★ 相機累計幀序 (raw 影片的幀號)
        self.lock = threading.Lock()
        self.running = False
        self.t = None
        self.recorder = None            # ★ 掛上 VideoWriter 就會逐幀錄
        self.captured = 0               # 相機成功讀到的幀數
        self.read_fail = 0              # 連續讀取失敗次數
        self.reopens = 0
        self.last_capture_t = 0.0
        self.error = None

    def _open(self):
        cv2 = self.cv2
        for idx in (self.cfg.CAM_INDEX, self.cfg.CAM_INDEX + 1,
                    self.cfg.CAM_INDEX + 2):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            except Exception:
                cap = cv2.VideoCapture(idx)
            if cap is None or not cap.isOpened():
                continue
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            # ★ 2026/09/11 test_camera_bench 實測:
            #   這張擷取卡【只有 YUYV】, 舊的 MJPG 要求被無聲忽略 (實測
            #   MJPG 與 YUYV 明示結果完全相同), 改成明示避免誤導.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            # ★ 向 V4L2 要求的 fps 要【高於】實際需求.
            #   BUFFERSIZE=1 時, 消費端只要慢一點就會固定錯過下一幀:
            #     要求 30 (間隔 33ms), 執行緒被卡到 40ms → 掉成 15 fps
            #     要求 60 (間隔 17ms), 同樣卡 40ms       → 掉成 20 fps
            #   單獨跑時本路徑實測可達 60 fps; 整個程式跑起來會被 YOLO /
            #   NVENC / 控制迴路搶 CPU, 要求高一點才有排擠的餘裕.
            cap.set(cv2.CAP_PROP_FPS,
                    float(getattr(self.cfg, 'CAM_REQUEST_FPS',
                                  self.cfg.CAM_FPS)))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.IMG_W))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.IMG_H))
            for _ in range(8):
                ok, f = cap.read()
                if ok and f is not None:
                    print(f"[Camera] V4L2 index={idx} OK")
                    return cap
            cap.release()
        return None

    def start(self):
        self.cap = self._open()
        if self.cap is None:
            return False
        self.running = True
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        for _ in range(50):
            with self.lock:
                if self.frame is not None:
                    return True
            time.sleep(0.1)
        return False

    def _run(self):
        """★ 這條執行緒只做兩件事: 讀相機 + 寫 raw 影片. 絕不能死."""
        while self.running:
            try:
                ok, f = self.cap.read()
            except Exception as e:
                ok, f = False, None
                self.error = f"cap.read() 例外: {e}"
            if ok and f is not None:
                self.read_fail = 0
                now = time.time()
                with self.lock:
                    self.frame = f
                    self.frame_id += 1
                    self.captured += 1
                    self.last_capture_t = now
                # raw 錄影 (與 YOLO 完全解耦)
                if self.recorder is not None:
                    try:
                        self.recorder.write(f, frame_id=self.frame_id)
                    except Exception as e:
                        self.error = f"raw 錄影 write 例外: {e}"
            else:
                self.read_fail += 1
                # 連續讀不到 → 嘗試重開相機 (別讓 raw 影片就此結束)
                if self.read_fail in (150, 600) or \
                        (self.read_fail > 600 and self.read_fail % 600 == 0):
                    print(f"[Camera] ★ 連續 {self.read_fail} 次讀不到幀, 嘗試重開")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    cap = self._open()
                    if cap is not None:
                        self.cap = cap
                        self.reopens += 1
                        self.read_fail = 0
                        print(f"[Camera] ✓ 重開成功 (第 {self.reopens} 次)")
                time.sleep(0.001)

    def read(self):
        """相容舊介面: 只回傳最新幀."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def read_with_id(self):
        """回傳 (最新幀, 該幀的相機幀序). 給 vision log 對齊 raw 影片用."""
        with self.lock:
            if self.frame is None:
                return None, -1
            return self.frame.copy(), self.frame_id

    def stats(self):
        with self.lock:
            return {
                'captured': self.captured,
                'frame_id': self.frame_id,
                'read_fail': self.read_fail,
                'reopens': self.reopens,
                'age': (time.time() - self.last_capture_t
                        if self.last_capture_t else None),
                'error': self.error,
            }

    def stop(self):
        self.running = False
        if self.t is not None:
            self.t.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
#  選框: 一個目標假設下, 從 YOLO 多框中挑「該追的那一個」
# ══════════════════════════════════════════════════════════════════
def select_detection(boxes, confs, locked, dkf_pred, img_w, img_h,
                     center_norm_max, track_gate_px):
    """
    boxes: list of (cx, cy, w, h);  confs: list;  兩者同長度 (可能為空)
    locked: 是否已鎖定 (已建立 DKF 追蹤)
    dkf_pred: (u, v) DKF 上一拍預測, 鎖定後選框基準; None 用畫面中心

    回傳選中框的 index, 或 None (沒有可接受的框).
      - 未鎖定 (首次截獲): 取「離畫面中心最近」且在 center_norm_max 內
      - 已鎖定: 取「離 dkf_pred 最近」且距離 ≤ track_gate_px
    → 別的車/樹只要不是最近的那個, 就選不上 (不看 conf 排序).
    """
    if not boxes:
        return None
    if locked and dkf_pred is not None:
        ref_u, ref_v = dkf_pred
        gate = track_gate_px
    else:
        ref_u, ref_v = img_w / 2.0, img_h / 2.0
        gate = center_norm_max * math.hypot(img_w / 2.0, img_h / 2.0)
    best_i, best_d = None, float('inf')
    for i, (cx, cy, w, h) in enumerate(boxes):
        d = math.hypot(cx - ref_u, cy - ref_v)
        if d < best_d:
            best_d, best_i = d, i
    if best_d > gate:
        return None
    return best_i


# ══════════════════════════════════════════════════════════════════
#  視覺前端狀態 (給控制迴路 / CSV / overlay 讀)
# ══════════════════════════════════════════════════════════════════
class VisionState:
    __slots__ = (
        'detected', 'conf', 'bbox_u', 'bbox_v', 'bbox_w', 'bbox_h',
        'dkf_used', 'dkf_status', 'u_used', 'v_used',
        'offset_norm', 'gate_pass', 'gate_dist_m',
        'slant_m', 'ground_m', 'tgt_n', 'tgt_e', 'range_conf',
        'inf_ms', 'delay_ms',
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)
        self.detected = False
        self.dkf_used = False
        self.gate_pass = False


# ══════════════════════════════════════════════════════════════════
#  VisionFrontend — 對主程式的單一介面
# ══════════════════════════════════════════════════════════════════
class VisionFrontend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fx, self.fy, self.cx, self.cy = cfg.effective_intrinsics()

        # YOLO (延遲 import, 避免無 GPU 環境掛掉整包)
        from ultralytics import YOLO
        print(f"[YOLO] 載入 {cfg.YOLO_MODEL_PATH} ...")
        self.model = YOLO(cfg.YOLO_MODEL_PATH)
        dummy = np.zeros((cfg.IMG_H, cfg.IMG_W, 3), dtype=np.uint8)
        for _ in range(3):
            self.model.predict(dummy, conf=cfg.YOLO_CONF, verbose=False)
        print("[YOLO] warmup 完成")

        # DKF: 延遲補償後估測 → PID (use_yolo_when_available=False)
        self.dkf = DKFTrackerGimbal(
            fps=cfg.CAM_FPS,
            screen_width=cfg.IMG_W, screen_height=cfg.IMG_H,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            use_yolo_when_available=False,   # ★ 吃延遲補償後估測
            verbose=True)

        # 射線投影測距
        self.ranger = RayProjectionDistanceEstimator(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            gimbal_pitch_offset_deg=cfg.GIMBAL_PITCH_OFFSET_DEG,
            verbose=True)

        # 鎖定狀態
        self.locked = False
        self._acquire_streak = 0
        self._prev_dkf_pred = None
        self._frames_since_det = 0
        # 目標 NED 先驗 (閘門中心): 初始 = 種點前方; 鎖定後 EMA 更新
        self.tgt_prior = None        # (n, e) 或 None
        self._last_frame_annot = None

    # ------------------------------------------------------------------
    def set_prior(self, n, e):
        """主程式在撥 GUIDED 種點時給閘門一個初始中心 (航跡前方 300m)."""
        self.tgt_prior = (float(n), float(e))

    # ------------------------------------------------------------------
    def _yolo(self, frame):
        """回傳 (boxes, confs, annotated_frame, inference_ms).
        boxes: list of (cx, cy, w, h)."""
        import cv2
        t0 = time.perf_counter()
        res = self.model.predict(frame, conf=self.cfg.YOLO_CONF,
                                 classes=self.cfg.YOLO_TARGET_CLASS_IDS,
                                 verbose=False)[0]
        inf_ms = (time.perf_counter() - t0) * 1000.0
        boxes, confs = [], []
        annot = frame
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cf = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, cf):
                boxes.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                              x2 - x1, y2 - y1))
                confs.append(float(c))
        return boxes, confs, annot, inf_ms

    # ------------------------------------------------------------------
    def _draw(self, frame, sel_box, other_boxes, u_used, v_used, dkf_used,
              gate_pass):
        """畫框: 選中綠框, 其他灰框, DKF 補位藍十字."""
        import cv2
        if sel_box is not None:
            cx, cy, w, h = sel_box
            col = (0, 255, 0) if gate_pass else (0, 165, 255)  # 綠=接受 橙=閘門拒
            cv2.rectangle(frame, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)), col, 2)
        for (cx, cy, w, h) in other_boxes:
            cv2.rectangle(frame, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)), (128, 128, 128), 1)
        if u_used is not None:
            c = (255, 128, 0) if dkf_used else (0, 0, 255)
            cv2.drawMarker(frame, (int(u_used), int(v_used)), c,
                           cv2.MARKER_CROSS, 20, 2)
        return frame

    # ------------------------------------------------------------------
    def step(self, frame, ac, gimbal, dive_mode, dt):
        """
        每幀處理一次. 回傳 (VisionState, annotated_frame).

        ac: dict — roll_enu/pitch_enu/yaw_enu (rad, MAVROS) 或已轉好的 NED,
            body_pitch_deg, body_yaw_true_deg, x_ned, y_ned, agl
        gimbal: dict — yaw/pitch (SIYI 原始度)
        dive_mode: True = DIVE 段 (FPV, ego comp 餵機身姿態; 不測距)
        """
        cfg = self.cfg
        st = VisionState()

        boxes, confs, annot, inf_ms = self._yolo(frame)
        st.inf_ms = inf_ms

        # ── 選框 (一個目標) ──
        sel = select_detection(
            boxes, confs, self.locked, self._prev_dkf_pred,
            cfg.IMG_W, cfg.IMG_H,
            cfg.ACQUIRE_CENTER_NORM_MAX, self._track_gate_px())
        sel_box = boxes[sel] if sel is not None else None
        other = [b for i, b in enumerate(boxes) if i != sel]

        # ── 地理閘門 (只在有先驗時檢查; 用當幀姿態反投影偵測到 NED) ──
        gate_pass = True
        gate_dist = None
        if sel_box is not None and self.tgt_prior is not None and not dive_mode:
            gate_pass, gate_dist = self._geo_gate(sel_box, ac, gimbal)
        st.gate_pass = gate_pass
        st.gate_dist_m = gate_dist

        det_ok = sel_box is not None and gate_pass
        if det_ok:
            self._frames_since_det = 0
        else:
            self._frames_since_det += 1

        # ── 首次截獲判定 (未鎖定): 連續穩定居中才鎖 ──
        if not self.locked:
            if det_ok:
                cx, cy, w, h = sel_box
                off = math.hypot(cx - cfg.IMG_W / 2.0, cy - cfg.IMG_H / 2.0) \
                    / math.hypot(cfg.IMG_W / 2.0, cfg.IMG_H / 2.0)
                if off <= cfg.ACQUIRE_CENTER_NORM_MAX:
                    self._acquire_streak += 1
                else:
                    self._acquire_streak = 0
                if self._acquire_streak >= cfg.ACQUIRE_STREAK:
                    self.locked = True
                    print(f"[Vision] ★ 鎖定目標 (連續 {cfg.ACQUIRE_STREAK} 幀居中)")
            else:
                self._acquire_streak = 0

        # ── 延遲補償量 (YOLO 推論 + 擷取鏈) ──
        delay_ms = inf_ms + cfg.CAPTURE_LATENCY_MS
        st.delay_ms = delay_ms

        # ── DKF: DIVE 餵機身姿態差 (FPV body-fixed), 其他餵雲台角 ──
        if dive_mode:
            ego_yaw = ac.get('body_yaw_true_deg')
            ego_pitch = ac.get('body_pitch_deg')
        else:
            ego_yaw, ego_pitch = gimbal['yaw'], gimbal['pitch']

        u_yolo = v_yolo = bw = bh = conf = None
        if det_ok:
            u_yolo, v_yolo, bw, bh = sel_box[0], sel_box[1], sel_box[2], sel_box[3]
            conf = confs[sel]

        u_used, v_used, dkf_used, status = self.dkf.step(
            detected=det_ok,
            u_yolo=u_yolo, v_yolo=v_yolo, bbox_w=bw, bbox_h=bh,
            dt=dt, confidence=conf,
            gimbal_yaw_deg=ego_yaw, gimbal_pitch_deg=ego_pitch,
            measurement_delay_ms=delay_ms)

        st.detected = det_ok
        st.conf = conf
        st.bbox_u, st.bbox_v = (sel_box[0], sel_box[1]) if sel_box else (None, None)
        st.bbox_w, st.bbox_h = (sel_box[2], sel_box[3]) if sel_box else (None, None)
        st.u_used, st.v_used = u_used, v_used
        st.dkf_used, st.dkf_status = dkf_used, status
        if u_used is not None:
            st.offset_norm = math.hypot(u_used - self.cx, v_used - self.cy) \
                / math.hypot(cfg.IMG_W / 2.0, cfg.IMG_H / 2.0)
            self._prev_dkf_pred = (u_used, v_used)

        # ── 測距 (只在非 DIVE; DIVE 純 bbox 迴路不需要) ──
        if not dive_mode and u_used is not None and ac.get('agl'):
            self._range(st, u_used, v_used, ac, gimbal)
            # 鎖定後用估測 NED 更新閘門先驗 (EMA)
            if self.locked and st.tgt_n is not None:
                if self.tgt_prior is None:
                    self.tgt_prior = (st.tgt_n, st.tgt_e)
                else:
                    a = 0.3
                    self.tgt_prior = (
                        (1 - a) * self.tgt_prior[0] + a * st.tgt_n,
                        (1 - a) * self.tgt_prior[1] + a * st.tgt_e)

        annot = self._draw(annot, sel_box, other, u_used, v_used, dkf_used,
                           gate_pass)
        return st, annot

    # ------------------------------------------------------------------
    def _track_gate_px(self):
        """鎖定後選框像素閘門, 隨掉幀放寬."""
        t_lost = self._frames_since_det / max(self.cfg.CAM_FPS, 1.0)
        return self.cfg.TRACK_GATE_PX + self.cfg.TRACK_GATE_PX_PER_S * t_lost

    def _gate_radius_m(self):
        """地理閘門半徑, 動態目標隨丟失時間膨脹.

        ★★ 2026/09/12 修正: 原本寫 cfg.TARGET_MOVING, 但 config 定義的是
        MOVING_TARGET —— AttributeError.
        這不是死碼: 呼叫路徑是 _geo_gate(), 條件 tgt_prior is not None,
        也就是【目標鎖定建立之後每一幀都會踩到】.
        2026/09/08 那趟視覺執行緒在 t=754.7s 猝死, 就是這行:
          t=470 有 3 幀偵測但 ACQUIRE_STREAK=5 沒滿 → 未鎖定 → 走不到這裡
          t=754 連續 8 幀偵測 → streak 滿 → 建立 tgt_prior
                → 下一幀進 _geo_gate → 這行炸 → 舊版無 try/except → 執行緒死
          視覺 log 最後一筆 frame 18254 = 最後一筆偵測的同一幀, 完全吻合.
        """
        r = self.cfg.GATE_RADIUS_M
        if getattr(self.cfg, 'MOVING_TARGET', False):
            t_lost = self._frames_since_det / max(self.cfg.CAM_FPS, 1.0)
            r += self.cfg.GATE_MOVING_VMAX_MPS * t_lost
        return r

    def _ned_attitude(self, ac):
        """取 NED 姿態 (優先用已轉好的, 否則從 ENU rad 轉)."""
        if ac.get('roll_ned_deg') is not None:
            return (ac['roll_ned_deg'], ac['pitch_ned_deg'],
                    ac['yaw_true_deg'])
        return convert_mavros_enu_to_ned_attitude(
            ac['roll_enu'], ac['pitch_enu'], ac['yaw_enu'])

    def _geo_gate(self, sel_box, ac, gimbal):
        """偵測反投影到 NED, 與先驗比距離. 回傳 (pass, dist_m)."""
        cx, cy = sel_box[0], sel_box[1]
        roll, pitch, yaw_t = self._ned_attitude(ac)
        gp, gy = convert_siyi_gimbal_to_estimator(
            gimbal['yaw'], gimbal['pitch'],
            yaw_sign=self.cfg.SIYI_YAW_SIGN, pitch_sign=self.cfg.SIYI_PITCH_SIGN)
        slant, ground, conf = self.ranger.estimate_distance(
            cx, cy, roll, pitch, yaw_t, gp, gy,
            ac['agl'], ac.get('x_ned', 0.0), ac.get('y_ned', 0.0))
        if self.ranger.last_target_ned is None:
            return True, None   # 無法估 (射線不朝下) → 不擋, 交 streak 判斷
        tn, te = self.ranger.last_target_ned[0], self.ranger.last_target_ned[1]
        d = math.hypot(tn - self.tgt_prior[0], te - self.tgt_prior[1])
        return (d <= self._gate_radius_m()), d

    def _range(self, st, u, v, ac, gimbal):
        """射線投影測距, 填 st.slant_m/ground_m/tgt_n/tgt_e/range_conf."""
        roll, pitch, yaw_t = self._ned_attitude(ac)
        gp, gy = convert_siyi_gimbal_to_estimator(
            gimbal['yaw'], gimbal['pitch'],
            yaw_sign=self.cfg.SIYI_YAW_SIGN, pitch_sign=self.cfg.SIYI_PITCH_SIGN)
        slant, ground, conf = self.ranger.estimate_distance(
            u, v, roll, pitch, yaw_t, gp, gy,
            ac['agl'], ac.get('x_ned', 0.0), ac.get('y_ned', 0.0))
        st.slant_m, st.ground_m, st.range_conf = slant, ground, conf
        if self.ranger.last_target_ned is not None:
            st.tgt_n = float(self.ranger.last_target_ned[0])
            st.tgt_e = float(self.ranger.last_target_ned[1])
