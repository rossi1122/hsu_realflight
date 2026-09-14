# 實驗執行小抄 — A/B × 靜止/移動 (vision_dive)

到場地照這份跑。4 個 case = DIVE 橫向 A/B 版本 × 靜止/移動目標。

> **版本 A** = DIVE 用影像 PID 修橫向 (方向舵)
> **版本 B** = DIVE 不修橫向, 凍結進 DIVE 那刻的航向, 方向舵交還飛控
>              (縱向 pitch 兩版都照修) — `--dive-hold-heading`

> 📌 **2026/09/11 掛機驗證後更新。**改了什麼看最下面「本次更新摘要」。

---

## 開飛前 (每次到場地做一次)

### 1. 程式座標

改 `config.py` 移動目標座標為當地:
```
MOVING_START_GPS   = (車起點 lat, lon)
MOVING_HEADING_GPS = (朝向點 lat, lon)   # 算馬路方向
```
(可選) 填靜止版座標，這樣 case 3/4 不用打長指令:
```
TARGET_GPS  = (目標 lat, lon)
PREDIVE_GPS = (俯衝點 lat, lon)
```

### 2. 飛控參數 (Mission Planner)

| 參數 | 值 | 為什麼 |
|---|---|---|
| `PTCH_LIM_MIN_DEG` | **-40** | ★ 必須與 `PITCH_CMD_MIN_DEG` **完全一致**。軟體要得比飛控多 → 被靜默夾掉 → 看不見的積分飽和 |
| `AIRSPEED_MAX` | ≥ **30** | 對齊 `PULLOUT_VMAX_MPS`，否則俯衝時飛控抬頭壓速度跟俯衝命令打架 |
| `AIRSPEED_MIN` | ≤ **15** | 對齊 `SEARCH_SPEED_MPS`。超出範圍的 43000 CHANGE_SPEED 會被拒 (`result=2`) |
| `AIRSPEED_CRUISE` | 記下來 | AUTO 實際巡航空速。`SEARCH_SPEED_MPS` 建議設成一樣，撥 GUIDED 才不會加減速 |

> ArduPlane 4.4 之後 `ARSPD_FBW_MIN/MAX` 改名為 `AIRSPEED_MIN/MAX`，
> `TRIM_ARSPD_CM` 改名為 `AIRSPEED_CRUISE`。舊韌體用舊名。
| `GUIDED_TIMEOUT` | **0.5** | forced attitude 逾時。太長會讓模式切換時殘留上一段的命令 |
| `GUIDED_P` | 存在即可 | 沒有這個參數代表韌體沒編進 43002 航向控制，SEARCH 會退回 legacy bank |

程式端對應 (`dive_guidance.py` / `attitude_bridge.py`):
```
PITCH_CMD_MIN_DEG          = -40.0
REQUIRED_PITCH_LIM_MIN_DEG = -40.0
```
三個數字要一模一樣。開機會印 `[Bridge] PTCH_LIM_MIN_DEG=-40° OK (需求 -40°)`，
沒看到這行就是還有一個沒對上。

### 3. 遙控器

**方向舵微調 (trim) 歸中。**

掛機實測發現不同時段的方向舵中立值差了 93 PWM，懷疑是發射機的舵量被疊加到
guided 命令上。我們的方向舵命令只有 ±20 PWM，會被這種偏壓完全淹沒。

### 4. 其他

- 車上 GPS 記錄器 (小無人機) 開好，飛完拿 BIN。
- **實飛不要帶 `--ground-test` / `--bench` / `--bench-dive`**。

---

## 4 個 CASE 指令

★ 一律加 **`--no-annot`**：不在飛行中編標註影片，省一路 1080p NVENC。
Jetson 壓力測試期間出現過 `system throttled due to over current`，這是最划算
的減負。標註影片飛完用 `make_annot_video.py` 重建，內容完全一樣。

### 移動目標 (座標從 config 讀)

```bash
# Case 1: 移動 + 版本A (影像 PID 修橫向)
python3 run_vision_flight.py --moving

# Case 2: 移動 + 版本B (橫向凍結)
python3 run_vision_flight.py --moving --dive-hold-heading
```

### 靜止目標

若已填 config 的 TARGET_GPS / PREDIVE_GPS:
```bash
# Case 3: 靜止 + 版本A
python3 run_vision_flight.py --static

# Case 4: 靜止 + 版本B
python3 run_vision_flight.py --static --dive-hold-heading
```

若沒填 config, 用命令列帶座標:
```bash
python3 run_vision_flight.py --static --no-annot \
        --target 目標lat,目標lon --predive 俯衝點lat,俯衝點lon
# 版本 B 再加 --dive-hold-heading
```

---

## 開機訊息檢查 (每趟都看)

程式起來之後，這四行都要正確才起飛:

```
[flight] --no-annot: 不錄標註影片 (事後 make_annot_video.py 重建)
[flight] raw 錄影已掛在取像執行緒 (與 YOLO 解耦)
[Bridge] PTCH_LIM_MIN_DEG=-40° OK (需求 -40°)
[Bridge] GUIDED_P=... → 43002 航向控制可用 (SEARCH 走 bank turn)
[Bridge] 方向舵迴路 ON — 限幅 ±3.0°, Kψ=0.3 Kr=0.15 K_lat=0.5, sign=+1
```

**看到任何一條下面的就停下來**:
- `★★★ 找不到 GUIDED_P` → SEARCH 會退回 legacy bank，不是預期的控制律
- `★★★ RUDDER_VERIFIED=False` → 方向舵命令會全部被歸零
- `★★★ 警告: PTCH_LIM_MIN_DEG=... 淺` → 俯衝角會被飛控夾掉

---

## 每趟操作流程 (共通)

1. 程式啟動 → 等 "等待 home_position 與 GPS fix" 通過 → 檢查上面四行。
2. AUTO 起飛、繞圈、飛到對準馬路 (移動: 車已在起點等速跑)。
3. 撥 **GUIDED** = 任務開始 (移動: 撥 GUIDED 瞬間車要在起點 GPS)。
4. 飛機自動 SEARCH → PRE_DIVE → DIVE → PULLOUT → 拉平交 TECS。
5. 撥離 GUIDED / 接手 FBWB 收場。
6. 掛機備用鍵盤: `d`=arm `x`=abort `q`=quit。**收工用 `q` 不要用 Ctrl+C。**

> 每趟自動產生獨立 log (檔名帶時間戳), 4 趟不會互相覆蓋。

### 飛行中各段在做什麼 (掛機已逐項驗證)

| 段 | type_mask | roll | pitch | 方向舵 |
|---|---|---|---|---|
| SEARCH | **0**(不送姿態) | 飛控 (43002) | 飛控 (TECS) | 飛控 |
| PRE_DIVE | 132 | 0 | -28.7 | 飛控 |
| DIVE 版本A | 128 | 0 | 影像 PID | 影像 PID |
| DIVE 版本B | 132 | 0 | 影像 PID | 飛控 |
| PULLOUT / ABORT | 132 | 0 | 拉平 / +12 | 飛控 |
| DONE | 198 | 0 | TECS | 飛控 |

---

## 每趟產出的檔 (在 flight_logs/)

```
Case 1 (移動+A) → flight_<時間>_moving_A_yawpid.*
Case 2 (移動+B) → flight_<時間>_moving_B_yawhold.*
Case 3 (靜止+A) → flight_<時間>_static_A_yawpid.*
Case 4 (靜止+B) → flight_<時間>_static_B_yawhold.*
```

```
<stem>.csv          主 log (50Hz, 論文畫圖 + 命令 + 飛控回饋 + 逐幀 miss)
<stem>_vision.csv   視覺 log (每幀處理的影像)
<stem>_raw.mp4      乾淨影片 (後處理重跑 YOLO 用)
<stem>_raw.fps.txt  真實 fps (時基修正用)
<stem>_events.txt   事件 (模式切換 / 成績 / 看門狗警告)
<stem>_error.log    視覺執行緒的例外 traceback (正常飛行不會產生)
```

> 想自己命名可加 `--case 標籤` (例 `--case trial3`)。

---

## 飛完立刻做 (還在場地)

### 1. 看收尾的總結

```
==========================================================
            錄影 / 資料鏈 總結
==========================================================
  任務時長          : ...
  相機擷取幀數      : ...
  視覺處理幀數      : ...   平均 23 fps 左右
  相機實際 fps      : 30.0
  raw   影片        : 入列 N, 寫入 N, 丟棄 0, 殘留 0  OK
  視覺執行緒        : OK
==========================================================
```

`入列 = 寫入`、`丟棄 0`、`視覺執行緒 OK` — 三個都對這趟資料才完整。

### 2. 自動判讀

```bash
python3 check_bench_log.py flight_logs/flight_<時間>.csv
```

逐模式印 PASS/FAIL: mask 對不對、方向舵有沒有超限、飛控的 `nav_pitch` 有沒有
跟上我們的 pitch 命令、43002 有沒有被拒。**在場地上就能發現哪趟是壞資料**，
不用回家才知道要重飛。

### 3. 確認 miss 成績

`events.txt` 裡找:
```
★★★ 成績: miss = xx.xx m (lon +x.xx, lat -x.xx)
```

---

## 回家後處理 (每趟一次)

```bash
# 重建標註影片 (--no-annot 飛的都要做)
python3 make_annot_video.py flight_logs/flight_<時間>

# 移動目標 (Case 1/2): 帶車 BIN 對齊真車軌跡
python3 postprocess_compare.py flight_logs/flight_<時間> --car 車的BIN檔

# 靜止目標 (Case 3/4)
python3 postprocess_compare.py flight_logs/flight_<時間>
```

產出 `*_compare.csv` (逐幀 real vs virt 命令 + 差值) 和 `*_compare.png`，
終端印 RMS 統計。

> ⚠ raw 錄影已與 YOLO 解耦，raw 影片幀數 > 視覺 log 列數 (相機 30fps，
> 視覺約 23fps)。後處理改用視覺 log 的 `raw_frame_idx` 對齊，舊格式 log
> 會自動退回順序對齊。跑的時候會印出用哪種模式。

---

## 比較這 4 個 case 時看什麼

### A vs B (同目標) — 這是論文的主對照

- **`miss_pred_lat_m` 對 `agl`**: 橫向誤差的收斂曲線。
  版本 B 不修橫向，lat 只會隨幾何線性收斂；版本 A 若有效，應該收得更快。
  **這條線就是「影像伺服橫向控制有沒有用」的直接證據。**
- **`miss_pred_lon_m`**: 打長/打短，歸縱向 PID 管，兩版應該差不多。
- `heading_cmd`: 版本 A 會擺頭追目標，版本 B 是一條水平線。
- `rudder_cmd_deg`: 版本 A 非零，版本 B 全 0。

### 靜止 vs 移動 (同版本)

移動版多了前置瞄準 + 車在動的挑戰。

### 虛擬 vs 真實視覺 (每趟後處理)

驗證虛擬目標是不是真實視覺的好替身 — **這是這一版流程要回答的核心問題**。

### 論文可以畫的圖 (資料都在主 CSV)

| 圖 | 欄位 |
|---|---|
| 目標在 FOV 的縱橫向保持 | `img_err_u_pct` / `img_err_v_pct`；視覺 log 的 `real_bbox_u/v` 疊上去比對 |
| 縱向命令 vs 實際 | `pitch_cmd` / `fc_nav_pitch_deg` / `theta_deg` 三線 |
| 橫向命令 vs 實際 | `heading_cmd` / `chi_deg` + `rudder_cmd_deg` |
| miss 收斂 | `miss_pred_m` / `miss_pred_lat_m` 對 `agl` |
| 俯衝穩定度 | `q_dps` / `nz_g` / `roll_deg` |

> `fc_nav_pitch_deg` 是飛控真的收到的姿態目標，夾在「命令」和「實際」之間。
> 三條疊起來可以分辨誤差出在通訊層還是飛行動力學。

---

## 本次更新摘要 (2026/09/11)

掛機逐項驗證後的改動。**指令本身只多了 `--no-annot`，其他都是背後的修正。**

### 控制路徑 (之前是斷的)

1. **`SET_ATTITUDE_TARGET` 的 yaw 欄位在 ArduPlane 固定翼是方向舵命令**
   (centidegree，45° = 滿舵)，**不是航向目標**。舊版把 0~360 的航向塞進去
   → 除了 135~225 之間，其餘全程滿舵打死。掛機 Test A/B 逐點驗證。
2. **`type_mask` 必須設 `IGNORE_ATTITUDE` 位元，roll/pitch 才生效。**
   `mask=7` 完全無效、`128`/`132` 正常。跟 ArduPlane 原始碼的 XOR 邏輯推導
   不符，以實測為準。
3. **PULLOUT / ABORT / 撥 GUIDED 前的暖機串流不再打舵** — 舊版在這三處也送
   滿舵，低空高速滿舵是安全問題。
4. **SEARCH 改用 43002 (`GUIDED_CHANGE_HEADING`)**，roll/pitch/方向舵全交
   飛控。43002 失敗 5 次會自動退回 legacy bank 並警告。

### 資料鏈 (之前會靜默失敗)

5. **raw 錄影搬到取像執行緒**，與 YOLO 解耦。2026/09/08 視覺執行緒在 754s
   靜默死亡、主 log 卻跑到 1075s，影片缺了最後 5 分 21 秒。現在 YOLO 掛掉不
   影響 raw 影片。
6. **視覺執行緒加例外保護 + 心跳看門狗**，斷掉會當場在 console 跳警告。
7. **相機 15fps → 30fps** (`CAM_REQUEST_FPS=60` 留 CPU 排擠餘裕)，並且不再
   重複處理同一張影像 (以前 36% 的視覺 log 是重複幀，會讓 DKF 過度自信)。
8. **影片時基修正** — 真實 fps 寫進 `.fps.txt`，有 ffmpeg 就自動修。

### 新增

9. `check_bench_log.py` — 飛完自動判讀每個模式。
10. `make_annot_video.py` — 離線重建標註影片。
11. 主 CSV 新增 `fc_nav_pitch_deg` / `fc_nav_roll_deg` (飛控真的收到什麼) 與
    `miss_pred_m` / `miss_pred_lon_m` / `miss_pred_lat_m` (逐幀 miss)。
12. 視覺 CSV 新增 `raw_frame_idx` (對齊 raw 影片) 與 `inf_ms` / `delay_ms`。

### 還沒解決的 (知情即可)

- **Jetson over-current 降頻**。15 分鐘壓力測試期間出現過，沒造成掉幀，但降
  頻最可能發生在俯衝這種高負載時刻。`--no-annot` 已經減掉一半編碼負載。
  若又出現視覺執行緒中途死掉，這是第一個要查的嫌疑犯。
- **DIVE 段的影像 PID 掛機驗不了** (pitch/heading 都是相對實際姿態的增量式，
  桌上飛機姿態不對，誤差就不對)。只能實飛驗。
- **方向舵權限第一版刻意很小** (±3°，約 6.7% 權限，空速上去還會再打折)。
  第一趟的目的是確認符號與穩定性，不是打中。確認沒問題後再逐步加大
  (建議一次加一倍 3→6→12，每次看 log 的 `roll_deg` / `r_dps` 有無振盪)。
