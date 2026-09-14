# -*- coding: utf-8 -*-
"""
virtual_target_geom.py — 純幾何工具 (vision_dive)
═══════════════════════════════════════════════════════════════════
從 virtual_target.py 抽出三個純函數 (無模擬狀態, 實飛也要用):
  gps_to_ned          — 經緯度 → 相對 home 的 (北, 東) 公尺
  extrapolate_impact  — 拉平瞬間沿速度向量外插到地面 → 預測彈著點
  miss_distance       — 彈著點 vs 目標 (含目標外插) → miss 分解 (lon/lat)

★ 逐字沿用 virtual_target.py 的實作, 只是搬到獨立檔避免 import 整個模擬器.
"""

import math

_R_EARTH = 6378137.0   # WGS-84 赤道半徑 (m)


def gps_to_ned(lat_deg, lon_deg, ref_lat_deg, ref_lon_deg):
    """經緯度 → 相對參考點的 (北, 東) 公尺. 小範圍等距近似 (誤差 << 1m @ 數 km)."""
    dlat = math.radians(lat_deg - ref_lat_deg)
    dlon = math.radians(lon_deg - ref_lon_deg)
    n = dlat * _R_EARTH
    e = dlon * _R_EARTH * math.cos(math.radians(ref_lat_deg))
    return n, e


def extrapolate_impact(n, e, agl, vn, ve, vd):
    """
    拉平觸發瞬間: 沿當下速度向量直線延伸到地面.

    Args:
        n, e: 飛機 NED 水平位置 (m)
        agl:  高度 (m, 向上為正)
        vn, ve: 水平速度 NED (m/s)
        vd: 下沉率 (m/s, 向下為正; 必須 > 0 才有交點)
    Returns:
        (impact_n, impact_e, t_fall) 或 (None, None, None) 若沒在下降
    """
    if vd <= 0.1:
        return None, None, None
    t_f = agl / vd
    return n + vn * t_f, e + ve * t_f, t_f


def miss_distance(impact_n, impact_e, t_fall,
                  tgt_n, tgt_e, tgt_vn, tgt_ve,
                  ac_vn, ac_ve):
    """
    Miss distance (論文慣例: 目標也外插到落地時刻).

    Returns dict:
        miss_total, miss_lon (沿飛機航跡向), miss_lat (垂直航跡向, 右正),
        tgt_impact_n, tgt_impact_e, t_fall
    """
    ti_n = tgt_n + tgt_vn * t_fall
    ti_e = tgt_e + tgt_ve * t_fall
    mn = impact_n - ti_n
    me = impact_e - ti_e
    total = math.hypot(mn, me)
    gs = math.hypot(ac_vn, ac_ve)
    if gs > 0.5:
        un, ue = ac_vn / gs, ac_ve / gs
        lon = mn * un + me * ue
        lat = -mn * ue + me * un
    else:
        lon, lat = total, 0.0
    return {
        'miss_total': total, 'miss_lon': lon, 'miss_lat': lat,
        'tgt_impact_n': ti_n, 'tgt_impact_e': ti_e,
        't_fall': t_fall,
    }
