#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess_compare.py — 離線比較 (vision_dive)
═══════════════════════════════════════════════════════════════════
飛行時控制吃「虛擬目標」的影像誤差; 這支腳本把飛行錄的 raw 影片帶回來:

  raw 影片逐幀重跑 YOLO → 真實影像誤差
    → 用飛行當下的飛機姿態 / setpoint / dt (從視覺 log 逐幀對齊取)
       完整重放一個獨立 DivePID 實例 (積分/微分狀態跑起來)
    → 得到「若當初餵真實視覺, PID 會下的姿態命令」
  跟飛行當下虛擬目標實際下的命令 (主 log) 逐幀比較 → 輸出比較 CSV + 圖

★ 只比 DIVE 段 (影像 PID 真正在動的地方).
★ 兩條命令流跑在同一條飛行姿態序列上, 只差影像誤差來源 (虛擬 vs 真實).
  這是「虛擬目標是不是真實視覺的好替身」的驗證 (反事實比較).

對齊: raw 影片第 N 幀 ⟷ 視覺 log 第 N 列 (frame_idx). 主 log 用 frame_idx
關聯回控制拍 (取該 frame_idx 最後一筆).

移動目標 (--car 車BIN): 額外用 GPS 絕對時間 (gps_unix) 把「真車實際位置」對到
每一 DIVE 幀, 跟虛擬目標當幀位置比 → 輸出沿馬路 / 垂直馬路殘差, 量化「虛擬
目標軌跡 vs 真車軌跡」差多少 (虛擬替身準不準). 車 BIN 與飛機 log 靠 GPS 絕對
時間對齊 (兩邊都記 GPS 週時間, 精度到毫秒).

用法:
  # 靜止版
  python3 postprocess_compare.py flight_logs/flight_20260828_153000
  # 移動版 (帶車 BIN 做軌跡對齊)
  python3 postprocess_compare.py flight_logs/flight_<時間> --car car.BIN
  python3 postprocess_compare.py <prefix> --no-plot   # 只出 CSV
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dive_guidance import GuidanceConfig, DivePID


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_vision_log(path):
    """視覺 log → list of dict (每處理幀一列, 已含飛機狀態快照)."""
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_flight_log(path):
    """主 log → dict: frame_idx → 該 frame_idx 最後一筆控制列 (取虛擬命令)."""
    by_frame = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            fi = r.get('frame_idx')
            if fi not in (None, ''):
                by_frame[int(float(fi))] = r
    return by_frame


def build_gcfg(cfg):
    """重建與飛行一致的 GuidanceConfig (PID 增益/濾波/內參要相同)."""
    fx, fy, cx, cy = cfg.effective_intrinsics()
    return GuidanceConfig(
        DIVE_ANGLE_DEG=cfg.DIVE_ANGLE_DEG,
        ALPHA_NOM_DEG=cfg.ALPHA_NOM_DEG, ALPHA_DIVE_DEG=cfg.ALPHA_DIVE_DEG,
        FX=fx, FY=fy, CX=cx, CY=cy, IMG_W=cfg.IMG_W, IMG_H=cfg.IMG_H,
        PID_LAT_KP=cfg.PID_LAT_KP, PID_LAT_KI=cfg.PID_LAT_KI, PID_LAT_KD=cfg.PID_LAT_KD,
        PID_LON_KP=cfg.PID_LON_KP, PID_LON_KI=cfg.PID_LON_KI, PID_LON_KD=cfg.PID_LON_KD,
        PID_SP_LEAD_S=cfg.PID_SP_LEAD_S,
        PID_MEAS_TAU_S=cfg.PID_MEAS_TAU_S, PID_D_TAU_S=cfg.PID_D_TAU_S,
        PID_OUT_TAU_S=cfg.PID_OUT_TAU_S,
        DIVE_AIM_V_OFFSET_PX=cfg.dive_aim_v_offset_px,
    )


def yolo_detect_center(model, frame, conf, img_w, img_h):
    """對一幀跑 YOLO, 回傳離畫面中心最近的框中心 (u,v,w,h,conf) 或 None."""
    res = model.predict(frame, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    cf = res.boxes.conf.cpu().numpy()
    best, bestd = None, float('inf')
    for (x1, y1, x2, y2), c in zip(xyxy, cf):
        cu, cv = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        d = math.hypot(cu - img_w / 2.0, cv - img_h / 2.0)
        if d < bestd:
            bestd = d
            best = (cu, cv, x2 - x1, y2 - y1, float(c))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix', help='log 前綴 (不含副檔名), 例 flight_logs/flight_20260828_153000')
    ap.add_argument('--conf', type=float, default=None, help='YOLO conf (預設用 config)')
    ap.add_argument('--car', type=str, default=None,
                    help='車上小無人機 BIN (移動版: 對齊真車軌跡, 量虛擬替身誤差)')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    raw_path = args.prefix + '_raw.mp4'
    vis_path = args.prefix + '_vision.csv'
    flt_path = args.prefix + '.csv'
    out_csv = args.prefix + '_compare.csv'
    out_png = args.prefix + '_compare.png'

    for p in (raw_path, vis_path, flt_path):
        if not os.path.exists(p):
            print(f"[FATAL] 找不到 {p}")
            sys.exit(1)

    cfg = Config()
    gcfg = build_gcfg(cfg)
    conf = args.conf if args.conf is not None else cfg.YOLO_CONF
    fx, fy, cx, cy = cfg.effective_intrinsics()

    print("[post] 載入 log ...")
    vis_rows = load_vision_log(vis_path)
    flt_by_frame = load_flight_log(flt_path)
    print(f"[post] 視覺 log {len(vis_rows)} 幀, 主 log {len(flt_by_frame)} frame_idx")

    # ── 車 BIN (移動版): 建真車軌跡 + 從主 log 反推 home 對齊 NED ──
    car = None
    home_ll = None
    if args.car:
        if not os.path.exists(args.car):
            print(f"[FATAL] 找不到車 BIN {args.car}")
            sys.exit(1)
        from car_track import CarTrack
        from virtual_target_geom import gps_to_ned
        car = CarTrack(args.car)
        if not car.has_data():
            print("[post] ★ 車 BIN 無有效軌跡, 略過軌跡對齊")
            car = None
        else:
            # 反推 home: 主 log 某列有 lat/lon 與 ac_n/ac_e (NED 相對 home)
            #   home ≈ 由 (lat,lon) 反推使 gps_to_ned(lat,lon,home)=(ac_n,ac_e)
            #   等距近似下 home_lat = lat − ac_n/R(度), home_lon 同理.
            R = 6378137.0
            for r in flt_by_frame.values():
                la, lo = _f(r.get('lat')), _f(r.get('lon'))
                an, ae = _f(r.get('ac_n')), _f(r.get('ac_e'))
                if None not in (la, lo, an, ae):
                    hlat = la - math.degrees(an / R)
                    hlon = lo - math.degrees(ae / (R * math.cos(math.radians(la))))
                    home_ll = (hlat, hlon)
                    break
            if home_ll is None:
                print("[post] ★ 主 log 無法反推 home, 略過車軌跡對齊")
                car = None
            else:
                print(f"[post] 反推 home=({home_ll[0]:.7f},{home_ll[1]:.7f})")

    # 馬路方位單位向量 (沿/垂直馬路分解用; 由 config 兩座標算)
    road_dir = None
    if car is not None:
        from virtual_target_geom import gps_to_ned
        if cfg.MOVING_HEADING_DEG is not None:
            brg = math.radians(cfg.MOVING_HEADING_DEG)
        else:
            hn, he = gps_to_ned(cfg.MOVING_HEADING_GPS[0], cfg.MOVING_HEADING_GPS[1],
                                cfg.MOVING_START_GPS[0], cfg.MOVING_START_GPS[1])
            brg = math.atan2(he, hn)
        road_dir = (math.cos(brg), math.sin(brg))   # (北分量, 東分量)

    # YOLO
    import cv2
    from ultralytics import YOLO
    print(f"[post] 載入 YOLO {cfg.YOLO_MODEL_PATH} ...")
    model = YOLO(cfg.YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        print(f"[FATAL] 無法開 raw 影片 {raw_path}")
        sys.exit(1)

    # 離線重放的獨立 PID (積分/微分狀態跑起來)
    real_pid = DivePID(gcfg)

    results = []
    n_dive = 0

    # ★ 2026/09/09: raw 錄影已搬到取像執行緒, 與 YOLO 解耦 →
    #   raw 影片幀數 > 視覺 log 列數, 不再是「第 N 幀 = 第 N 列」.
    #   改用視覺 log 的 raw_frame_idx 對齊 (舊 log 沒這欄 → 自動退回舊行為).
    _has_raw_idx = bool(vis_rows) and (vis_rows[0].get('raw_frame_idx')
                                       not in (None, ''))
    # ★ 2026/09/12: raw 錄影在 FrameGrabber 啟動後才掛上 → MP4 第 0 幀
    #   不是相機第 0 幀. VideoWriter 把偏移寫進 <name>_raw.fps.txt.
    _raw_offset = 0
    try:
        _fp = os.path.splitext(raw_path)[0] + '.fps.txt'
        if os.path.exists(_fp):
            for _ln in open(_fp):
                if _ln.startswith('first_frame_id='):
                    _v = _ln.split('=', 1)[1].strip()
                    if _v:
                        _raw_offset = int(_v)
                elif _ln.startswith('dropped=') and _ln.split('=')[1].strip() not in ('0', ''):
                    print(f"[post] ⚠ raw 錄影有掉幀 ({_ln.strip()}), "
                          f"固定偏移不再成立, 對齊可能錯位")
        if _raw_offset:
            print(f"[post] raw 幀號偏移 = {_raw_offset} "
                  f"(MP4 第 0 幀 = 相機第 {_raw_offset} 幀)")
    except Exception as _e:
        print(f"[post] 讀 .fps.txt 失敗 ({_e}), 偏移當 0")

    if _has_raw_idx:
        by_raw = {}
        for _r in vis_rows:
            try:
                by_raw[int(_r['raw_frame_idx'])] = _r
            except (TypeError, ValueError):
                pass
        print(f"[post] 對齊模式: raw_frame_idx ({len(by_raw)} 幀有視覺資料)")
    else:
        by_raw = None
        print("[post] 對齊模式: 順序 (舊格式 log, 無 raw_frame_idx)")

    raw_idx = -1
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_idx += 1
        if by_raw is not None:
            vr = by_raw.get(raw_idx + _raw_offset)
            if vr is None:
                continue          # 這一幀相機有錄, 但視覺沒處理到
            frame_idx = int(_f(vr.get('frame_idx'), raw_idx))
        else:
            frame_idx = raw_idx
            if frame_idx >= len(vis_rows):
                break
            vr = vis_rows[frame_idx]

        # 只比 DIVE 段
        if int(_f(vr.get('dive_mode'), 0)) != 1:
            continue
        n_dive += 1

        # 這一幀飛機狀態 (視覺 log 已釘好)
        pitch_now = _f(vr.get('theta_deg'), 0.0)
        hdg_now = _f(vr.get('chi_deg'), 0.0)
        dt = _f(vr.get('frame_dt'), 0.033) or 0.033
        u_sp = _f(vr.get('virt_u_sp'), cx)
        v_sp = _f(vr.get('virt_v_sp'), cy)

        # raw 影片重跑 YOLO → 真實影像位置
        det = yolo_detect_center(model, frame, conf, cfg.IMG_W, cfg.IMG_H)
        if det is not None:
            ru, rv, rw, rh, rc = det
            detected = True
        else:
            ru = rv = None
            detected = False

        # 重放 PID: 真實影像誤差 + 同一飛行姿態/ setpoint
        r_pitch, r_hdg = real_pid.update(
            ru, rv, u_sp, v_sp, detected, pitch_now, hdg_now, dt)

        # 飛行當下虛擬命令 (主 log, 同 frame_idx)
        fr = flt_by_frame.get(frame_idx, {})
        v_pitch = _f(fr.get('pitch_cmd'))
        v_hdg = _f(fr.get('heading_cmd'))

        # ── 真車軌跡對齊 (移動版): 用該幀 GPS 絕對時間查真車位置 ──
        car_dn = car_de = car_along = car_cross = ''
        if car is not None:
            gu = _f(vr.get('gps_unix'))
            vtn, vte = _f(fr.get('tgt_n')), _f(fr.get('tgt_e'))
            if gu is not None and None not in (vtn, vte):
                cne = car.position_at(gu, home_ll[0], home_ll[1])
                if cne is not None:
                    car_dn = cne[0] - vtn      # 真車 − 虛擬 (北)
                    car_de = cne[1] - vte      # 真車 − 虛擬 (東)
                    # 沿馬路 / 垂直馬路分解 (用馬路方位)
                    if road_dir is not None:
                        rn, re = road_dir
                        car_along = car_dn * rn + car_de * re      # 沿馬路
                        car_cross = -car_dn * re + car_de * rn     # 垂直馬路(右正)
                        car_along, car_cross = round(car_along, 2), round(car_cross, 2)
                    car_dn, car_de = round(car_dn, 2), round(car_de, 2)

        row = {
            'frame_idx': frame_idx,
            't': vr.get('t', ''),
            # 真實 (raw 影片離線)
            'real_detected': int(detected),
            'real_u': round(ru, 1) if ru is not None else '',
            'real_v': round(rv, 1) if rv is not None else '',
            'real_pitch_cmd': round(r_pitch, 3),
            'real_heading_cmd': round(r_hdg, 3),
            # 虛擬 (飛行實際)
            'virt_u_sp': round(u_sp, 1), 'virt_v_sp': round(v_sp, 1),
            'virt_pitch_cmd': v_pitch if v_pitch is not None else '',
            'virt_heading_cmd': v_hdg if v_hdg is not None else '',
            # 差值 (real − virt)
            'd_pitch_deg': (round(r_pitch - v_pitch, 3)
                            if v_pitch is not None else ''),
            'd_heading_deg': (round(((r_hdg - v_hdg + 180) % 360) - 180, 3)
                              if v_hdg is not None else ''),
            # 機上即時 YOLO (交叉驗證)
            'onboard_u': vr.get('real_bbox_u', ''),
            'onboard_v': vr.get('real_bbox_v', ''),
            # 真車 vs 虛擬目標軌跡殘差 (移動版; 虛擬替身準不準)
            'car_minus_virt_n': car_dn, 'car_minus_virt_e': car_de,
            'car_along_road': car_along, 'car_cross_road': car_cross,
        }
        results.append(row)

    cap.release()
    print(f"[post] DIVE 段 {n_dive} 幀, 比較 {len(results)} 幀")

    # ── 輸出比較 CSV ──
    fields = ['frame_idx', 't', 'real_detected', 'real_u', 'real_v',
              'real_pitch_cmd', 'real_heading_cmd', 'virt_u_sp', 'virt_v_sp',
              'virt_pitch_cmd', 'virt_heading_cmd', 'd_pitch_deg', 'd_heading_deg',
              'onboard_u', 'onboard_v',
              'car_minus_virt_n', 'car_minus_virt_e', 'car_along_road', 'car_cross_road']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f"[post] 比較 CSV → {out_csv}")

    # ── 統計 ──
    dp = [_f(r['d_pitch_deg']) for r in results if _f(r['d_pitch_deg']) is not None]
    dh = [_f(r['d_heading_deg']) for r in results if _f(r['d_heading_deg']) is not None]

    def _rms(a):
        return math.sqrt(sum(x * x for x in a) / len(a)) if a else float('nan')

    def _mean(a):
        return sum(a) / len(a) if a else float('nan')
    print(f"[post] pitch 命令差:  RMS={_rms(dp):.3f}°  mean={_mean(dp):+.3f}°  "
          f"max|·|={max((abs(x) for x in dp), default=float('nan')):.3f}°")
    print(f"[post] heading 命令差: RMS={_rms(dh):.3f}°  mean={_mean(dh):+.3f}°  "
          f"max|·|={max((abs(x) for x in dh), default=float('nan')):.3f}°")

    # ── 真車 vs 虛擬目標軌跡殘差 (移動版; 虛擬替身準不準) ──
    if car is not None:
        al = [_f(r['car_along_road']) for r in results
              if _f(r['car_along_road']) is not None]
        cr = [_f(r['car_cross_road']) for r in results
              if _f(r['car_cross_road']) is not None]
        if al:
            print(f"[post] 真車−虛擬 沿馬路: RMS={_rms(al):.2f}m mean={_mean(al):+.2f}m "
                  f"max|·|={max((abs(x) for x in al)):.2f}m")
            print(f"[post] 真車−虛擬 垂直馬路: RMS={_rms(cr):.2f}m mean={_mean(cr):+.2f}m "
                  f"max|·|={max((abs(x) for x in cr)):.2f}m")
            print(f"[post]   → 沿馬路殘差 = 切 GUIDED 相位差 + 車速非 10 的累積; "
                  f"垂直馬路殘差 = 車偏離直線程度")

    # ── 圖 (論文用) ──
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            ts = [_f(r['t']) for r in results]
            rp = [_f(r['real_pitch_cmd']) for r in results]
            vp = [_f(r['virt_pitch_cmd']) for r in results]
            rh = [_f(r['real_heading_cmd']) for r in results]
            vh = [_f(r['virt_heading_cmd']) for r in results]
            fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            ax[0].plot(ts, vp, label='virtual (flown)', lw=1.5)
            ax[0].plot(ts, rp, label='real vision (offline)', lw=1.2, ls='--')
            ax[0].set_ylabel('pitch cmd (deg)'); ax[0].legend(); ax[0].grid(alpha=0.3)
            ax[0].set_title('Pitch command: virtual target vs real vision (DIVE)')
            ax[1].plot(ts, vh, label='virtual (flown)', lw=1.5)
            ax[1].plot(ts, rh, label='real vision (offline)', lw=1.2, ls='--')
            ax[1].set_ylabel('heading cmd (deg)'); ax[1].set_xlabel('t (s)')
            ax[1].legend(); ax[1].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_png, dpi=130)
            print(f"[post] 比較圖 → {out_png}")
        except ImportError:
            print("[post] 無 matplotlib, 跳過畫圖 (CSV 已出)")


if __name__ == '__main__':
    main()
