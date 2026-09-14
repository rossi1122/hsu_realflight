# vision_dive — 固定翼無人機終端俯衝影像導引 (虛擬目標驗證版)

固定翼終端俯衝任務程式，以碩論《Image-Based Visual Servoing for Terminal
Dive Guidance of a Fixed-Wing UAV》為基礎，終端導引用**影像 PID**（實飛版）。

**這一版是「虛擬目標驗證」流程**：控制迴路吃**已知 GPS 的虛擬目標**投影出來
的影像誤差；真實相機 + YOLO 照跑但**只驅動雲台 + 錄影**，不進飛機控制。飛完
把 raw 影片帶回來離線重跑 YOLO，用同一條飛行姿態序列重放 PID，跟虛擬目標的
命令逐幀比較 —— 驗證「虛擬目標是不是真實視覺的好替身」。

---

## 一句話說明任務

程式支援**靜止目標**與**移動目標**兩種模式（`config.MOVING_TARGET` 或
`--moving/--static` 切換）。

**靜止目標**：給目標 GPS + PRE_DIVE 點 GPS。飛機定高逼近，水平到達 PRE_DIVE
點就進俯衝。

**移動目標（預設，你的實際場景）**：直的馬路，車在馬路上等速 10 m/s 往前開。
只給**目標起點 GPS**（切 GUIDED 時車在此），方向由 config 兩座標算死（= 馬路
方向 355.5°，不用飛機航向），車速 10 m/s。進俯衝改用**飛機到目標當下位置的
水平距離** ≤ 門檻（不需固定 PRE_DIVE 點）。之後以 30° 俯衝、虛擬目標投影的
影像誤差做 PID 把目標壓在畫面中心，到 70 m 或超速 30 m/s 自動拉平。

**三條資料路徑（兩種模式都一樣）**
- **控制路徑**：GPS 虛擬目標 → 投影成影像誤差 → 飛機 PID → 姿態命令（不依賴視覺）
- **視覺路徑**（並行，只記錄）：相機 → YOLO → 雲台追蹤 + 錄影（標註 + raw）+ 視覺 log
- **後處理**（離線）：raw 影片 → 重跑 YOLO → 重放 PID → 跟虛擬命令比較；
  移動版另用車上 GPS（BIN）對齊真車軌跡，量虛擬替身的軌跡誤差

實飛輸入 = 目標 GPS（移動版只需起點）+ 遙控器模式開關。撥 GUIDED = 開始；撥離 = 中止。

---

## 任務流程（狀態機）

```
   AUTO 繞圈對準 (未 GUIDED)
        │  雲台 STANDBY：FOLLOW 模式，yaw 跟機頭看正前方
        │  相機 + YOLO 照跑（錄影 + 暖機）
        ▼
  ┌─ 撥 GUIDED ─────────────────────────────────┐
  │  = arm + 預送 TECS 目標（目標/PRE_DIVE 點 GPS 已在啟動時給定）
  ▼                                              │
 SEARCH ── 定高逼近，虛擬目標投影 → 影像誤差 → PID 對準
  │         ★ 水平到達 PRE_DIVE GPS 點半徑內 → 進 PRE_DIVE
  │         （雲台此時 FOLLOW 追真實 YOLO bbox，讓影片有目標）
  ▼
 PRE_DIVE ── 定高逼近，到「距錐界 + margin」直接壓 -30°，
  │           pitch 進容差 → 交棒，雲台切 FPV（body-fixed）
  ▼
 DIVE ── 30° 俯衝。虛擬目標 body 投影 → 影像 PID：偏右→右轉、偏下→續壓頭。
  │        AGL ≤ 70 m 或 V ≥ 30 m/s → 拉平
  ▼
 PULLOUT ── 凍結成績（外插彈著點算 miss distance）→ 拉平過渡 → TECS 平飛
  ▼
 DONE ── 飛手撥離 GUIDED / 接手 FBWB 收場
        （逃生門 ABORT：AGL < 55 m 或目標丟失 > 1.5 s → 直線拉起）
```

> **控制吃虛擬目標，不吃 YOLO。** 這一版飛機 PID 的影像誤差全部來自虛擬目標
> （GPS 投影）。真實 YOLO 只用來驅動雲台（讓影片裡目標在畫面內）+ 錄影 +
> 記 log，它算的影像誤差**不進飛機 PID**。

---

## 移動目標怎麼操作（重點）

**車的開法 — 關鍵是「切 GUIDED 的時機」，不是「開車的時機」。**
虛擬目標在切 GUIDED 那刻（t=0）從你給的起點 GPS 開始跑，所以要重合就是：
**切 GUIDED 那一刻，真車必須剛好在起點 GPS、而且已經等速穩定。** 因此：

1. 車**先起步**，在到起點之前就加速到 10 m/s 穩定巡航（不要在起點才從靜止起步，
   那樣起點附近有加速段，跟等速模型對不上）。
2. 車保持 10 m/s 沿馬路往前開。
3. 你在空中看飛機追上來、對準馬路，**當車開到起點 GPS 那一刻，切 GUIDED**。

切不準沒關係（總會差幾公尺、車速不剛好 10.0）——車上 GPS（BIN）會記真車軌跡，
後處理用 GPS 絕對時間對齊，把相位差補掉。所以大方向對、車穩定等速在跑就好。

**進俯衝門檻**：飛機到目標當下位置的水平距離 ≤ `config.PREDIVE_DIST_M`（預設
200 m）。30° 俯衝從 120 m 高，理論俯衝點是「水平 = 120/tan30° ≈ 208 m」，所以
200 m 起跳合理；但你是**追擊**（飛機從後追、車往前跑），實際值要用第一趟 log 調。

**追擊時序要注意**：飛機巡航 20、車 10，相對接近只有 10 m/s。切 GUIDED 時飛機
在車後方的初始間距，決定多久追到俯衝門檻、那時剩多少高度。馬路你給的約 358 m、
車 10 m/s 跑完約 36 秒，要確認任務在車開出馬路前打完。這些第一趟飛就會清楚。

**虛擬 vs 真車的對齊（後處理）**：兩邊都記 GPS 絕對時間（飛機 CSV 的 `gps_unix`
來自 MAVROS `time_reference`；車 BIN 的 GPS 週時間換算），後處理 `--car car.BIN`
把真車位置對到每個 DIVE 幀，算沿馬路 / 垂直馬路殘差。沿馬路殘差 = 切 GUIDED 相位
差 + 車速非 10 的累積；垂直馬路殘差 = 車偏離直線的程度。這量化了虛擬替身多準。

---

## 為什麼這樣設計（幾個關鍵決定）

**切 GUIDED 前不鎖定，消滅「繞圈丟失 + 錯誤重截獲」。**
繞圈期間若一直鎖，容易跟丟又跟錯。改成 GUIDED 前雲台只用 FOLLOW 看正前方
（yaw 跟機頭、不亂掃），對準後撥 GUIDED，目標就在畫面中央 → 首次截獲取
「離畫面中心最近」的偵測（不是最高 conf），畫面裡有別的車也搶不走。

**鎖定後只認一台車。**
每幀 YOLO 可能吐多框，選框規則 = 取「離 DKF 預測點最近」+ 像素閘門，
超出閘門就整幀當沒偵測讓 DKF 補位。再加一道地理閘門（偵測反投影的 NED
位置要落在預期半徑內）。別的目標 conf 再高也進不來。→ 見 `vision.py`
`select_detection()`。

**DIVE 用雲台 FPV、不測距。**
射線投影測距靠「雲台世界穩定角度」的旋轉鏈，只在 SEARCH/PRE_DIVE（雲台
FOLLOW）成立。DIVE 切 FPV 後相機貼機身，改用純 bbox 影像 PID，靠 AGL/
空速判斷拉平，不再需要距離。

**攻角補償。** FPV 雲台俯角≈0，把 bbox 壓在畫面中心 = 機軸指向目標，但
飛機速度向量在機軸下方約 1.3°（俯衝攻角），不補會系統性打短。做法：把
縱向 setpoint 下移 `fy·tan(1.3°) ≈ 48 px`（見 `config.dive_aim_v_offset_px`）。

---

## 影像延遲補償（你問的那題）

**有補，而且用對了。** DKF（`DKFTrackerGimbal` → `EgoAwareDKF`）內建
retrodiction（論文 Algorithm 1）：每幀把 YOLO 推論耗時當量測延遲傳進去，
DKF 回退到 N 步前的狀態做更新、再沿存好的 ego 補償重播回當前時刻，內部
狀態是「已吸收延遲的當前估測」。

關鍵開關：本整合包設 `use_yolo_when_available=False`，讓 PID 吃的是
**延遲補償後的當前時刻估測**，而不是落後 50–80 ms 的原始 YOLO 量測。
（原版預設 True 只在補位時才用補償，正常追蹤仍輸出延遲量測 — 對雲台
追蹤無妨，但俯衝末端視線角速率高，80 ms 可差 30–60 px，故這裡關掉。）

兩個誠實邊界：
1. 量到的延遲只含 **YOLO 推論**。HDMI 擷取卡本身的 glass-to-frame 延遲
   另計 → `config.CAPTURE_LATENCY_MS`（預設 0，校一次填）。
2. 這只補**影像**延遲。機體 0.48 s 命令→響應死區是另一回事，PID 增益
   本來就繞著它調，不歸 DKF 管。

---

## 資料輸出（斷電保護）

每趟飛行四支檔 + 事件檔，論文畫圖與後處理都靠這些，且防 SD 卡沒存到：

- **主 log**（`flight_<時間>.csv`）：50 Hz 控制迴路逐拍。含 `frame_idx`（對齊鑰匙）、
  飛機狀態、虛擬目標 NED、`virt_bbox_u/v`（實際餵 PID 的影像位置）、
  `virt_u_sp/v_sp`（setpoint）、`pitch_cmd/heading_cmd`（虛擬命令＝比較基準）、
  PID 各項、影像誤差百分比、glide-slope 誤差、飛控取證（角速率/法向加速度/
  RCOUT/電壓）。
- **視覺 log**（`flight_<時間>_vision.csv`）：**每處理幀一列，跟 raw 影片 1:1**。
  含 `frame_idx`、時間、真實 YOLO bbox（機上即時，交叉驗證用）、DKF 輸出、
  **釘在該幀的飛機狀態快照**（pitch/heading/vn/ve/vd/dt）、該幀虛擬 setpoint、
  雲台角。後處理重放 PID 直接逐列取，不必跟主 log 內插。
- **標註影片**（`flight_<時間>.mp4`）：YOLO 框，給人看，節流 25 fps。
- **raw 影片**（`flight_<時間>_raw.mp4`）：**乾淨無框，每幀寫不節流**，跟視覺 log
  逐幀對齊。後處理重跑 YOLO 就吃這支（標註框會污染影像，故要 raw）。
- **事件檔**（`..._events.txt`）：模式切換、成績等。

兩支影片都優先用 Jetson NVENC + **fragmented MP4**（moov 先寫、每秒封片，
斷電只丟最後幾秒），無 NVENC 時退 OpenCV 分段 MP4。兩支 CSV 背景每 0.25 s
`fsync`，斷電最多丟最後幾筆。

## 後處理比較（飛完離線跑）

```bash
python3 postprocess_compare.py flight_logs/flight_<時間>
```

讀 raw 影片逐幀重跑 YOLO → 真實影像誤差 → 用視覺 log 裡**那一幀的飛機姿態 /
setpoint / dt**，完整重放一個獨立 `DivePID` 實例（積分/微分狀態跑起來，＝討論
時的方案 ii）→ 得到「若當初餵真實視覺，PID 會下的命令」，跟主 log 的虛擬命令
逐幀比較。只比 **DIVE 段**。輸出：

- `flight_<時間>_compare.csv`：逐幀 real vs virt 命令 + 差值。
- `flight_<時間>_compare.png`：pitch / heading 命令兩條曲線疊圖（論文用）。
- 終端印 RMS / mean / max 命令差。

> **這是反事實比較**：飛機實際軌跡是被虛擬目標命令帶出來的；後處理算的真實
> 視覺命令是「在這條真實軌跡上，若改餵真實視覺誤差 PID 會怎麼下命令」。兩條
> 命令流跑在同一條飛行姿態序列上，只差誤差來源 —— 這正是「虛擬目標是不是
> 真實視覺好替身」的驗證。

---

## 檔案結構

**新寫的**

| 檔案 | 作用 |
|---|---|
| `config.py` | 所有參數集中；2x 有效內參；靜止/移動目標 GPS、車速、進俯衝門檻 |
| `virtual_source.py` | **虛擬目標**：共同外殼 + `StaticTarget`/`MovingTarget`，每拍投影成餵 PID 的影像誤差 |
| `vision.py` | 視覺前端：取像→YOLO→選框→DKF→測距（這版只驅動雲台+錄影+記 log） |
| `gimbal.py` | SIYI 雲台：連線/2x/STANDBY 指向/FOLLOW 追蹤/切 FPV |
| `run_vision_flight.py` | 主程式：MAVROS IO + 視覺線程（錄影+視覺 log）+ 控制線程（虛擬目標→PID）|
| `video_writer.py` | 斷電保護影片寫入器（NVENC fragmented / cv2 分段；raw 版每幀寫） |
| `postprocess_compare.py` | **離線比較**：raw 影片重跑 YOLO → 重放 PID → 跟虛擬命令逐幀比 |
| `car_track.py` | **讀車上小無人機 BIN**：建真車軌跡，用 GPS 絕對時間對齊（移動版後處理）|
| `virtual_target_geom.py` | 純幾何函數（GPS→NED、彈著外插、miss distance） |

**驗證 / 診斷工具（2026/09 新增）**

| 檔案 | 作用 |
|---|---|
| `check_version.py` | **換檔案後先跑這個**：用關鍵字指紋確認每個檔案都是最新版 + 跨檔一致性 |
| `check_bench_log.py` | 飛完/掛機完自動判讀每個模式：mask、方向舵限幅、飛控有沒有吃 pitch |
| `test_rudder_bench.py` | 掛機驗方向舵映射與符號（Test A/B/C/D/E） |
| `test_camera_bench.py` | 相機取像效能對照（六種取像方式比 fps） |
| `make_annot_video.py` | 離線從 raw 影片 + 視覺 CSV 重建標註影片（`--no-annot` 飛的都要跑） |

**沿用專案既有（微改或原樣）**

| 檔案 | 說明 |
|---|---|
| `dive_guidance.py` | 任務狀態機 + 影像 PID（小改：攻角偏移參數 + **PRE_DIVE 觸發加 GPS 點/到目標距離模式**）|
| `dkf_tracker_gimbal.py` | DKF 追蹤器（小改：ego yaw 差分加 wrap，跨 0/360 不再產生假位移）|
| `ray_projection_distance.py` | 射線投影測距（未改） |
| `attitude_bridge.py` | MAVROS 姿態橋（**2026/09 大改**：yaw 欄位語意、type_mask、43002、方向舵迴路、stale watchdog）|
| `siyi_sdk.py` / `siyi_message.py` / `crc16_python.py` / `cameras.py` / `utils.py` | SIYI SDK（**一字未動**） |
| `vision_driver_car_dive_with_dkf_v3.py` | DKF 底層（`EgoAwareDKF`，未改） |

> **狀態機的 PRE_DIVE 觸發有三種模式**：`slant`（舊估距）、`gps_point`（靜止版：
> 到固定 PRE_DIVE 點）、`target_dist`（移動版：到目標當下位置水平距 ≤ 門檻）。
> PRE_DIVE → DIVE 維持原樣。靜止/移動由 `config.MOVING_TARGET` 或命令列決定。

---

## 執行

### 換檔案後第一件事

```bash
python3 check_version.py
```

用關鍵字指紋確認每個檔案都是最新版。**漏換一兩個的後果通常不是報錯，是靜默失效**
（例如 `run_vision_flight.py` 舊 + `vision.py` 新 → raw 影片完全不會被錄到）。
順便檢查 `PITCH_CMD_MIN_DEG` 與 `REQUIRED_PITCH_LIM_MIN_DEG` 有沒有對上。

---

### 實飛

```bash
# 機載電腦（已 source ROS + MAVROS 連上 Pixhawk）

# 移動目標（預設；起點/方向/車速可寫進 config，也可命令列覆寫）
python3 run_vision_flight.py --no-annot
python3 run_vision_flight.py --no-annot --start 22.9104301,120.3125242 --speed 10
python3 run_vision_flight.py --no-annot --dive-hold-heading     # 版本 B

# 靜止目標
python3 run_vision_flight.py --no-annot --static \
        --target 23.001234,120.005678 --predive 23.002000,120.004000
```

`--no-annot` = 飛行中不編標註影片，省一路 1080p NVENC（Jetson 曾出現
over-current 降頻）。落地用 `make_annot_video.py` 重建，內容完全一樣。
4 個 case 的完整流程見 `RUN_CASES.md`。

飛完立刻做：

```bash
python3 check_bench_log.py                    # 自動抓最新 log, 逐模式判讀
```

---

### 掛機測試（★ 拆槳）

不用把飛機搬到任何地方 —— **目標是虛擬的**，程式只拿「飛機 GPS」和「目標 GPS」
算距離。人和飛機都不動，軟體就以為目標在 150 m 外。

先算兩組座標（把飛機現在的位置和朝向填進去）：

```python
import math
lat0, lon0, hdg = 22.9956162, 120.2235067, 280.0    # ← 改成你的
R = 6378137.0
def at(brg, d):
    b = math.radians(brg)
    return (lat0 + math.degrees(d*math.cos(b)/R),
            lon0 + math.degrees(d*math.sin(b)/(R*math.cos(math.radians(lat0)))))
print('目標 (機頭偏右5度, 150m):', at(hdg+5, 150))
print('遠 PRE_DIVE 點 (300m)   :', at(hdg+5, 300))
print('近 PRE_DIVE 點          :', (lat0, lon0))   # = 飛機現在位置
```

目標放在機頭偏右 5°，是為了讓航向誤差落在 PID 未飽和的區間（`K_PSI=0.3`，
5° → 舵量 1.5°，不會打到 ±6 的限幅）。正前方誤差 0 什麼都看不到。

四輪（`<T>` = 目標座標，`<P近>` / `<P遠>` = PRE_DIVE 點）：

```bash
# Run 1 — SEARCH + 43002 (PRE_DIVE 點放遠, 停在 SEARCH)
python3 run_vision_flight.py --no-annot --bench --static \
        --target <T> --predive <P遠>
#   撥 GUIDED → 放 60 秒 → q
#   看: SPEED 15m/s、nav_roll 非 0 (飛控自己壓坡度)、hdg_cmd_ok_cnt 增加

# Run 2 — PRE_DIVE + ABORT + DONE  ★主要那輪
python3 run_vision_flight.py --no-annot --bench --static \
        --target <T> --predive <P近>
#   撥 GUIDED → 立刻進 PRE_DIVE → ★升降舵大幅偏轉並保持
#   → 看 15 秒 → 按 x (ABORT) → 等自動進 DONE → q

# Run 3 — DIVE 版本 A
python3 run_vision_flight.py --no-annot --bench --bench-dive --static \
        --target <T> --predive <P近>

# Run 4 — DIVE 版本 B (A/B 對照)
python3 run_vision_flight.py --no-annot --bench --bench-dive --static \
        --dive-hold-heading --target <T> --predive <P近>
```

每輪跑完：

```bash
python3 check_bench_log.py
```

掛機備用鍵盤：`d`=arm `x`=abort `q`=quit。**收工用 `q`，不要 Ctrl+C。**

#### 旗標的意思

| 旗標 | 作用 |
|---|---|
| `--bench` | 忽略 AGL 進入條件、幾何用假高度 120 m。**會停在 PRE_DIVE 持續命令俯衝姿態**，方便看舵面 |
| `--bench-dive` | 允許跳過 pitch 容差直接交棒 DIVE。⚠ DIVE 的 pitch 是相對實際姿態的增量式，桌上實際 pitch≈0 → 進 DIVE 後升降舵就不動了。**看舵面請用 `--bench`** |
| `--ground-test` | `heading_cmd` 改成當下機頭（不叫飛控轉向）。實飛四個 case 不要加 |
| `--no-video` | 兩支影片都不錄 |

#### 掛機驗不到的（原理上）

| 項目 | 為什麼 |
|---|---|
| PULLOUT 的 `mask=192` | AGL 門檻用假高度 120 > 70，走不到 PULLOUT（按 `x` 觸發的是 ABORT） |
| `miss_pred_*` | 計算條件是下沉率 > 0.1 m/s，桌上不成立 |
| DIVE 的影像 PID | pitch/heading 都是相對實際姿態的增量式，桌上姿態不對誤差就不對 |
| 方向舵的真實效果 | 沒有氣動負載 |

---

### 後處理（每趟一次）

```bash
# 重建標註影片 (--no-annot 飛的都要做)
python3 make_annot_video.py flight_logs/flight_<時間>

python3 postprocess_compare.py flight_logs/flight_<時間>            # 靜止
python3 postprocess_compare.py flight_logs/flight_<時間> --car car.BIN  # 移動(對齊真車)
```

**首飛前務必**：
1. **移動版**：確認 `config.MOVING_START_GPS`（車起點）、`MOVING_HEADING_GPS`
   （朝向點，算馬路方向）、`TARGET_SPEED_MPS`（車速）、`PREDIVE_DIST_M`（進俯衝
   門檻，預設 200 m，實飛調）。**靜止版**：`--target`/`--predive`。
2. `config.GIMBAL_PITCH_OFFSET_DEG` — 地面校正雲台報告角 vs 光軸偏差。
3. 確認 `config.YOLO_MODEL_PATH` 指到正確權重（機上跑 + 後處理都用它）。
4. `config.SIYI_YAW_SIGN / SIYI_PITCH_SIGN` — 若雲台方向相反改這個。
5. **移動版**：車上放記 GPS 的小無人機（ArduPilot BIN），飛後拿檔給
   `postprocess_compare --car` 對齊。

---

## 座標約定（全程一致）

NED（北-東-下）；航向 0=北、90=東；pitch 抬頭為正；
影像 u 向右為正、v 向下為正；目標偏右→右轉，目標偏下→壓頭。

## 機體參數（HERO2180 VTOL 固定翼）

巡航 17–22 m/s、失速 12 m/s、進場 AGL 120 m、目標前方 300 m、俯衝 30°、
攻角 巡航 3.5° / 俯衝 1.3°、拉平 70 m / 30 m/s。
