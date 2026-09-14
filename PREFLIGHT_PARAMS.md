# 實飛前飛控參數對齊清單 (vision_dive)

實飛前必讀。這份講**飛控 (ArduPlane) 裡要調的參數**，以及它們要跟程式
`config.py` 對齊到什麼值。

> ⚠ **兩套參數，別搞混**
> - **飛控參數**：在 Mission Planner 改，決定飛機本身的能力與限制。
> - **程式 config.py**：決定任務邏輯，但**受飛控參數限制**。
>
> 關鍵：程式送的命令，飛控會用它自己的參數去卡。程式叫 -30° 俯衝，但飛控
> `PTCH_LIM_MIN_DEG` 只准 -15°，飛機就只壓 15° → 任務失效還找不到原因。
> **所以兩邊一定要對齊。**

> ⚠ **參數名與預設值會隨 ArduPlane 版本變**。以下是方向與該找的參數；
> 確切名稱請在 Mission Planner 參數頁搜尋確認。數值要依你這台機子的特性定，
> **不要照抄**——尤其姿態控制增益是你之前實飛整定過的，別動。

---

## 最優先：這兩個沒對，任務直接失敗

| 飛控參數 | 為什麼 | 建議 |
|---|---|---|
| **`PTCH_LIM_MIN_DEG`** | 最大俯衝角(負值)。不放寬，飛機壓不到 30° | ≤ **-30**，保險設 **-35** |
| **`AIRSPEED_MAX`** | 太低的話俯衝時飛控會抬頭壓速度，跟俯衝命令打架 | ≥ **30**(對齊 PULLOUT_VMAX) |

---

## 完整對齊表 (飛控 ↔ config.py)

程式 config 目前的值列在右邊，飛控那邊要對齊或放寬到能容納它。

### 俯仰角
| 飛控參數 | 說明 | 對齊 config | config 現值 |
|---|---|---|---|
| `PTCH_LIM_MIN_DEG` | 最大俯衝角 | 要能容納 `-DIVE_ANGLE_DEG` | DIVE_ANGLE_DEG = 30 → 需 ≤ -30 |
| `PTCH_LIM_MAX_DEG` | 最大抬頭角 | 拉平/爬升夠用即可 | — |

### 空速
| 飛控參數 | 說明 | 對齊 config | config 現值 |
|---|---|---|---|
| `AIRSPEED_CRUISE` (舊 `TRIM_ARSPD_CM`) | 巡航空速 | = `V_CRUISE_MPS` | 20 m/s |
| `AIRSPEED_MAX` / `ARSPD_FBW_MAX` | 空速上限 | ≥ `PULLOUT_VMAX_MPS` | 30 m/s |
| `AIRSPEED_MIN` / `ARSPD_FBW_MIN` | 空速下限 | > `V_STALL_MPS` 留裕度 | stall 12 m/s |

> 注意 `TRIM_ARSPD_CM` 單位是 cm/s(2000 = 20 m/s)；新版 `AIRSPEED_CRUISE`
> 是 m/s。看你版本用哪個。

### 油門
| 飛控參數 | 說明 | 對齊 config | config 現值 |
|---|---|---|---|
| `TRIM_THROTTLE` | 配平油門 (%) | ≈ `THR_TRIM`×100 | 0.45 → 45% |
| `THR_MIN` | 最小油門 (%) | ≈ `THR_MIN`×100 | 0.05 → 5%(俯衝怠速) |
| `THR_MAX` | 最大油門 (%) | ≈ `THR_MAX`×100 | 0.85 → 85% |

### 高度 / TECS (拉平後平飛)
| 飛控參數 | 說明 | 對齊 config | config 現值 |
|---|---|---|---|
| `TECS_PITCH_MAX` | TECS 最大抬頭 | 拉平爬升夠用 | — |
| `TECS_PITCH_MIN` | TECS 最大低頭 | — | — |
| `TECS_SPDWEIGHT` 等 | 速度/高度權衡 | 沿用你調好的 | — |
| (TECS 目標由程式送) | 保持高度 / 空速 | `TECS_HOLD_ALT_M` / `TECS_SPEED_MPS` | 70 m / 18 m/s |

### 姿態控制器增益 (★ 別亂動)
| 飛控參數 | 說明 |
|---|---|
| `PTCH2SRV_*` | 俯仰角→升降舵。**你之前實飛整定過，是你機子的個性，別動** |
| `RLL2SRV_*` | 滾轉角→副翼。同上 |
| `YAW2SRV_*` / `KFF_RDDRMIX` | 偏航/方向舵混控。DIVE 橫向若靠 rudder，這個相關 |

> 程式的影像 PID 是**疊在這些飛控姿態增益之上**的：程式送姿態目標，飛控
> 用這些增益去追。所以飛控增益要先是「飛機能穩定追姿態」的狀態，程式 PID
> 才有意義。

---

## 安全 / 故障保護 (別漏)

| 飛控參數 | 說明 | 建議 |
|---|---|---|
| `FS_*` (failsafe) | RC 丟失 / GPS 丟失 / 電池低的行為 | 俯衝任務務必確認會帶到安全狀態 |
| `FENCE_*` | 地理圍欄 | **建議開**，俯衝失控時有邊界 |
| `FENCE_ALT_MIN` | 最低高度圍欄 | 可設在 ABORT 高度附近當硬底 |
| Q_* (VTOL) | 四旋翼切換 | 你的 SOP 是 50m 切四旋翼，確認 Q 參數與切換條件 |

> config 這邊的安全值：`PULLOUT_AGL_M`=70(拉平高度)、`ABORT_MIN_AGL_M`=55
> (逃生拉起)、SOP 50m 切四旋翼。飛控的 failsafe / fence 要跟這條安全鏈一致，
> 不要互相打架(例如 fence 觸發動作跟 ABORT 衝突)。

---

## 開飛前對齊步驟

1. **先確認 `PTCH_LIM_MIN_DEG` ≤ -30**（頭號，沒改俯衝壓不下去）。
2. **確認 `AIRSPEED_MAX` ≥ 30**（否則俯衝抬頭壓速度打架）。
3. 巡航空速、油門三個、TECS 目標，照上表對齊 config。
4. 姿態增益、failsafe、fence 用你之前調好的，別動增益，確認保護會動作。
5. 如果**你上次俯衝成功那趟**用的就是現在的飛控參數 → 大部分已 OK，
   只要確認 config 的值跟那趟一致即可。
6. 到場地後改 `config.py` 的 `MOVING_START_GPS` / `MOVING_HEADING_GPS`
   為當地座標（見 README 主檔）。

> **最省事的驗證**：你有一趟「20260818 俯衝成功」的紀錄。如果這次飛控參數
> 沒動過，那組參數已經能俯衝，你要做的只是確認 config.py 的數值跟那趟飛的
> 實際行為一致，不用重調飛控。

---

## config.py 目前值一覽 (對齊時參照)

```
巡航高度 CRUISE_AGL_M        = 120 m
俯衝角   DIVE_ANGLE_DEG      = 30°
拉平     PULLOUT_AGL_M       = 70 m   /  PULLOUT_VMAX_MPS = 30 m/s
ABORT    ABORT_MIN_AGL_M     = 55 m
巡航速   V_CRUISE_MPS        = 20 m/s (失速 12)
油門     THR_TRIM/MIN/MAX    = 0.45 / 0.05 / 0.85
TECS     HOLD_ALT / SPEED    = 70 m / 18 m/s
攻角     NOM / DIVE          = 3.5° / 1.3°
```

> 這些是「目前 config 的值」，不是叫你照抄到飛控。是讓你對齊時知道程式會送
> 什麼、飛控要能容納什麼。改 config 就改這裡，飛控端跟著對齊。
