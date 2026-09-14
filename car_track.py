# -*- coding: utf-8 -*-
"""
car_track.py — 讀車上小無人機的 ArduPilot BIN, 建真車軌跡 (vision_dive 後處理)
═══════════════════════════════════════════════════════════════════
車上放一台小無人機記 GPS (dataflash .BIN). 這支把它讀成「GPS 絕對時間 → 真車
NED 位置」的查詢表, 供 postprocess_compare 沿馬路對齊虛擬目標.

用 POS 訊息 (EKF 融合位置, 平滑) + GPS 訊息的週時間 (GWk/GMS) 換算絕對時間.
飛機 CSV 記的 gps_unix 也是 GPS 絕對時間 → 兩邊同一時間軸, 對齊到毫秒.

需 pymavlink: pip install pymavlink
"""

import bisect

_GPS_EPOCH_UNIX = 315964800   # 1980-01-06 00:00:00 UTC


def gps_week_to_unix(gwk, gms):
    """GPS 週數 + 週內毫秒 → unix 秒 (不扣閏秒; 與 MAVROS time_reference 同慣例)."""
    return _GPS_EPOCH_UNIX + gwk * 604800 + gms / 1000.0


class CarTrack:
    """
    讀車 BIN → 依 GPS 絕對時間排序的 (unix, lat, lon) 序列.
    position_at(unix, home_lat, home_lon) → (n, e) 線性內插的真車 NED.
    """

    def __init__(self, bin_path, verbose=True):
        from pymavlink import mavutil
        m = mavutil.mavlink_connection(bin_path)
        samples = []          # (unix, lat, lon)
        last_unix = None
        n = 0
        while True:
            msg = m.recv_match(blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == 'GPS':
                d = msg.to_dict()
                if d.get('GWk', 0) > 0 and d.get('Status', 0) >= 3:
                    last_unix = gps_week_to_unix(d['GWk'], d['GMS'])
            elif t == 'POS':
                d = msg.to_dict()
                if last_unix is not None and d.get('Lat', 0) != 0:
                    samples.append((last_unix, d['Lat'], d['Lng']))
            n += 1
            if n > 3_000_000:
                break
        samples.sort(key=lambda x: x[0])
        # 去重複時間 (POS 比 GPS 密, 同一 GPS 時間會有多筆 → 保留最後)
        self.times = [s[0] for s in samples]
        self.lats = [s[1] for s in samples]
        self.lons = [s[2] for s in samples]
        if verbose:
            if self.times:
                print(f"[CarTrack] {len(self.times)} 筆真車軌跡, "
                      f"時間 {self.times[0]:.1f}~{self.times[-1]:.1f} "
                      f"({self.times[-1]-self.times[0]:.1f}s)")
            else:
                print("[CarTrack] ★ 沒讀到有效車軌跡 (檢查 BIN 有無 POS + GPS fix)")

    def has_data(self):
        return len(self.times) > 0

    def span(self):
        return (self.times[0], self.times[-1]) if self.times else (None, None)

    def position_at(self, unix, home_lat, home_lon):
        """指定 GPS 絕對時間的真車 NED (相對 home). 超出範圍夾到端點. 線性內插."""
        from virtual_target_geom import gps_to_ned
        if not self.times:
            return None
        i = bisect.bisect_left(self.times, unix)
        if i <= 0:
            lat, lon = self.lats[0], self.lons[0]
        elif i >= len(self.times):
            lat, lon = self.lats[-1], self.lons[-1]
        else:
            t0, t1 = self.times[i - 1], self.times[i]
            a = 0.0 if t1 == t0 else (unix - t0) / (t1 - t0)
            lat = self.lats[i - 1] + a * (self.lats[i] - self.lats[i - 1])
            lon = self.lons[i - 1] + a * (self.lons[i] - self.lons[i - 1])
        return gps_to_ned(lat, lon, home_lat, home_lon)
