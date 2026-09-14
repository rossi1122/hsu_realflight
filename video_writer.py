# -*- coding: utf-8 -*-
"""
video_writer.py — 斷電保護的 YOLO 標註影片寫入器 (vision_dive)
═══════════════════════════════════════════════════════════════════
兩層後端, 自動挑可用的:
  1. NVENC + fragmented MP4 (Jetson 首選): gst-launch 硬體編碼,
     mp4mux streamable=true fragment-duration=1000 → moov 先寫 + 每秒封片,
     斷電最多丟最後幾秒, 之前照樣可播.
  2. cv2.VideoWriter MP4V (通用後援): 無 GStreamer/NVENC 時用. 標準 MP4 的
     索引在 release 才寫, 斷電會報廢整檔 → 每隔幾秒切一段新檔 (segment).

共通:
  - 背景 thread 從 queue 取幀寫, 不擋主視覺迴路
  - .ok 屬性讓主程式判斷是否成功啟動

═══════════════════════════════════════════════════════════════════
★ 2026/09/09 修正 (實飛 flight_20260908_101654 診斷)
═══════════════════════════════════════════════════════════════════
症狀: 12:34 的飛行, 標註影片播起來只有 11:33, 看起來像「錄一半就停」.

真因: 影片其實是完整的, 是【檔頭 fps 標錯】造成的時間壓縮.
  - 視覺迴圈實測只跑 24.2 fps (YOLO 卡住), 不是 config 的 30
  - 25Hz 節流又丟掉約 5% 的幀 (迴圈本來就低於 25, 抖動害的)
  - 實際寫入 17,325 幀, 檔頭卻寫 25 fps → 17325/25 = 693s = 11:33
  - 18,255 幀 / 754.4s 真實時長 → 真實寫入率 23.0 fps

修正三件事:
  1. 全程記錄真實幀數與真實時長, 收尾算出【實際 fps】並印出來
  2. 收尾若有 ffmpeg, 自動用實際 fps 重新封裝 (無損, 只改時基),
     播放時間就會跟 log 的 t 對得上. 沒 ffmpeg 就印出手動指令.
  3. 節流預設關閉 — 迴圈速率本來就低於目標 fps, 節流只是白丟幀

另外補三個「靜默失敗」漏洞 (都是飛完才會發現的那種):
  a. gst-launch 的 stderr 是 PIPE 但從來沒人讀 → 管線緩衝 (~64KB) 塞滿
     會把整條編碼管線卡死. 改成背景 thread 持續吸掉並留最後幾行.
  b. 寫入 thread 遇到例外只 break, 不印任何東西, .ok 還是 True →
     影片停了主程式完全不知道. 改成記錄錯誤 + 標記 ok=False + health().
  c. 沒有磁碟空間檢查. 8Mbps × 兩支影片 ≈ 每分鐘 120MB.
     開檔前檢查, 執行中低於門檻就停止寫入 (保住已寫的部分).
"""

import os
import shutil
import subprocess
import threading
import time
from collections import deque


class VideoWriter:
    MIN_FREE_MB_START = 2000.0    # 開檔前至少要有這麼多空間
    MIN_FREE_MB_STOP = 300.0      # 執行中低於此值就停寫 (保住已寫的部分)

    def __init__(self, path, fps, width, height, bitrate_bps=8_000_000,
                 seg_seconds=30.0, throttle=False, remux_on_release=True,
                 queue_len=8, drop_log=True):
        self.path = path
        self.fps = int(fps)                  # 檔頭標稱 fps (開檔時必須先給一個)
        self.w = int(width)
        self.h = int(height)
        self.bitrate = int(bitrate_bps)
        self.seg_seconds = float(seg_seconds)
        # ★ 預設不節流: 視覺迴圈實測 24 fps < VIDEO_FPS 25, 節流只會白丟幀
        self.throttle = bool(throttle)
        self.remux_on_release = bool(remux_on_release)
        self.ok = False
        self.backend = None
        self.error = None                    # ★ 失敗原因 (None = 正常)

        # ★ queue_len: raw 影片建議 120 (≈5s 緩衝). 舊值 8 只有 0.3s,
        #   編碼器或磁碟稍微一頓就會掉幀, 播放時看到畫面跳掉.
        self._q = deque(maxlen=int(queue_len))   # 滿了掉幀優於卡住主迴圈
        self.drop_log = bool(drop_log)
        self.drop_events = []                    # (t, queue_len, written)
        self._qlock = threading.Lock()
        self._run = True
        self._last_write_t = 0.0
        self._min_dt = 1.0 / max(self.fps, 1)
        self.written = 0
        self.dropped = 0
        self.enqueued = 0        # write() 被呼叫且成功入列的次數
        # ★ 2026/09/12: raw 錄影是在 FrameGrabber 啟動之後才掛上的, 所以
        #   MP4 第 0 幀 ≠ 相機第 0 幀 (實測差 200 幀左右).
        #   記下第一幀的相機幀號, 後處理才能正確換算:
        #     MP4 幀號 N  →  raw_frame_idx = N + first_frame_id  (前提 dropped=0)
        self.first_frame_id = None
        self.stop_requested_backlog = 0   # release() 時 queue 裡還有幾幀
        self.died_at_t = None    # 寫入停止的時刻 (相對第一幀)

        # ★ 真實時基 (算實際 fps 用)
        self.t_first = None                  # 第一幀進 queue 的時刻
        self.t_last = None                   # 最後一幀寫出的時刻
        self.last_written_t = 0.0            # 健康檢查用

        self._proc = None
        self._cv = None
        self._cv2 = None
        self._seg_idx = 0
        self._seg_t0 = 0.0
        self._stderr_tail = deque(maxlen=20)
        self._err_thread = None

        # ── 磁碟空間檢查 ──
        free_mb = self._free_mb()
        if free_mb is not None and free_mb < self.MIN_FREE_MB_START:
            self.error = (f"磁碟只剩 {free_mb:.0f}MB "
                          f"(需要 {self.MIN_FREE_MB_START:.0f}MB)")
            print(f"[Video] ★★★ 不啟動錄影: {self.error}")
            return

        _nvenc_err = None
        if self._try_nvenc():
            self.backend = 'nvenc'
            self.ok = True
            self.error = None
        else:
            _nvenc_err = self.error
            self.error = None
            if self._try_cv2():
                self.backend = 'cv2'
                self.ok = True
                self.error = None
                print(f"[Video] (NVENC 不可用: {_nvenc_err} → 改用 cv2)")
            else:
                self.error = (f"NVENC: {_nvenc_err} | cv2: {self.error}")

        if self.ok:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
            print(f"[Video] 後端={self.backend} 標稱fps={self.fps} "
                  f"節流={'ON' if self.throttle else 'OFF'} "
                  f"剩餘空間={free_mb:.0f}MB → {self.path}")

    # ------------------------------------------------------------------
    def _free_mb(self):
        try:
            d = os.path.dirname(os.path.abspath(self.path)) or '.'
            os.makedirs(d, exist_ok=True)
            return shutil.disk_usage(d).free / 1e6
        except Exception:
            return None

    # ---- NVENC fragmented MP4 ----
    def _try_nvenc(self):
        cmd = [
            'gst-launch-1.0', '-e', '-q',
            'fdsrc', 'fd=0', '!',
            'rawvideoparse', 'format=bgr',
            f'width={self.w}', f'height={self.h}', f'framerate={self.fps}/1', '!',
            'videoconvert', '!', 'nvvidconv', '!',
            'video/x-raw(memory:NVMM),format=NV12', '!',
            'nvv4l2h264enc', f'bitrate={self.bitrate}', '!',
            'h264parse', '!',
            'mp4mux', 'streamable=true', 'fragment-duration=1000', '!',
            'filesink', f'location={self.path}',
        ]
        try:
            # (2026/09/11 實測) start_new_session=True 讓 gst-launch 脫離本
            # 行程的 process group. 否則 Ctrl+C 的 SIGINT 會【先】打死 gst,
            # release() 還沒來得及關 stdin 管線就斷了 -> BrokenPipe, 最後幾
            # 秒丟失, fragmented MP4 也沒正常收尾.
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0,
                start_new_session=True)
        except Exception as e:
            self._proc = None
            self.error = f"gst-launch 啟動失敗: {e}"
            return False
        # ★ 一定要有人讀 stderr, 否則管線緩衝塞滿會把 gst 整個卡死
        self._err_thread = threading.Thread(target=self._drain_stderr,
                                            daemon=True)
        self._err_thread.start()
        time.sleep(0.15)
        if self._proc.poll() is not None:
            self.error = ("gst-launch 立刻結束 ("
                          + " / ".join(self._stderr_tail) + ")")
            self._proc = None
            return False
        return True

    def _drain_stderr(self):
        """持續吸掉 gst 的 stderr, 只留最後幾行給診斷用."""
        try:
            for line in iter(self._proc.stderr.readline, b''):
                s = line.decode('utf-8', 'replace').strip()
                if s:
                    self._stderr_tail.append(s)
        except Exception:
            pass

    # ---- cv2 segmented MP4V ----
    def _try_cv2(self):
        try:
            import cv2
            self._cv2 = cv2
        except Exception as e:
            self.error = f"cv2 不可用: {e}"
            return False
        return self._open_cv_segment()

    def _open_cv_segment(self):
        cv2 = self._cv2
        base, ext = os.path.splitext(self.path)
        seg_path = (self.path if self._seg_idx == 0
                    else f"{base}_seg{self._seg_idx:02d}{ext}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._cv = cv2.VideoWriter(seg_path, fourcc, self.fps, (self.w, self.h))
        if not self._cv.isOpened():
            self._cv = None
            self.error = f"cv2.VideoWriter 開不起來: {seg_path}"
            return False
        self._seg_t0 = time.time()
        return True

    # ---- 對外: 入 queue ----
    def write(self, frame, frame_id=None):
        """frame_id: 來源的累計幀號 (FrameGrabber.frame_id).
        只記第一筆 → 後處理可換算 MP4 幀號 ↔ 相機幀號."""
        if not self.ok or frame is None:
            return
        if frame_id is not None and self.first_frame_id is None:
            self.first_frame_id = int(frame_id)
        now = time.time()
        if self.throttle and (now - self._last_write_t < self._min_dt):
            return
        self._last_write_t = now
        if self.t_first is None:
            self.t_first = now
        with self._qlock:
            if len(self._q) >= self._q.maxlen:
                self.dropped += 1
                # ★ 不再 silent drop: 記下時間點, 收尾才知道哪幾秒跳掉
                if self.drop_log and len(self.drop_events) < 5000:
                    self.drop_events.append((now, len(self._q), self.written))
            self._q.append(frame)
            self.enqueued += 1

    # ---- 健康檢查 (主迴圈可定期呼叫) ----
    def health(self):
        """回傳 (ok, 訊息). ok=False 代表寫入已經停了, 訊息說明原因."""
        if not self.ok:
            return False, (self.error or "未啟動")
        if self.last_written_t:
            idle = time.time() - self.last_written_t
            if idle > 5.0:
                return False, f"已經 {idle:.0f}s 沒有寫出任何一幀"
        return True, "OK"

    # ---- 背景寫 ----
    def _loop(self):
        last_disk_check = 0.0
        while True:
            frame = None
            with self._qlock:
                if self._q:
                    frame = self._q.popleft()
                _empty = not self._q
            if frame is None:
                # ★ 2026/09/12: 收尾時把 queue 清完再退出.
                #   舊版是 `while self._run:` —— release() 把 _run 設 False,
                #   迴圈立刻結束, queue 裡還沒寫出的幀直接丟掉.
                #   raw queue 有 120 幀, 最壞丟掉影片尾端 4 秒.
                #   (舊版的「收尾時仍在佇列 N」只是報告, 沒有真的處理)
                if not self._run:
                    break
                time.sleep(0.002)
                continue

            # 每 10 秒看一次磁碟
            now = time.time()
            if now - last_disk_check > 10.0:
                last_disk_check = now
                free = self._free_mb()
                if free is not None and free < self.MIN_FREE_MB_STOP:
                    self.error = (f"磁碟剩餘 {free:.0f}MB, "
                                  f"停止寫入以保住已錄部分")
                    print(f"[Video] ★★★ {self.error}")
                    self.ok = False
                    self._run = False
                    break

            try:
                if self.backend == 'nvenc':
                    self._proc.stdin.write(frame.tobytes())
                else:
                    if time.time() - self._seg_t0 >= self.seg_seconds:
                        try:
                            self._cv.release()
                        except Exception:
                            pass
                        self._seg_idx += 1
                        self._open_cv_segment()
                    if self._cv is not None:
                        self._cv.write(frame)
                self.written += 1
                self.t_last = time.time()
                self.last_written_t = self.t_last
            except Exception as e:
                # ★ 舊版這裡只 break, 什麼都不印, .ok 還是 True →
                #   影片停了主程式完全不知道. 現在一定要留下證據.
                tail = " / ".join(self._stderr_tail) if self._stderr_tail else ""
                self.error = (f"寫入失敗於第 {self.written} 幀: "
                              f"{type(e).__name__}: {e}"
                              + (f" | gst: {tail}" if tail else ""))
                print(f"[Video] ★★★ {self.error}")
                self.died_at_t = (now - self.t_first) if self.t_first else None
                self.ok = False
                self._run = False
                break

    # ---- 收尾 ----
    def release(self):
        # ★ 先讓寫入執行緒把 queue 清完 (最多等 drain_timeout), 再關 encoder.
        #   join 的 timeout 要夠久: 120 幀 @ 30fps 約 4s, 給 8s 餘裕.
        self.stop_requested_backlog = 0
        with self._qlock:
            self.stop_requested_backlog = len(self._q)
        if self.stop_requested_backlog:
            print(f"[Video] 收尾: 佇列還有 {self.stop_requested_backlog} 幀, "
                  f"等它寫完 ...")
        self._run = False
        try:
            if hasattr(self, '_t'):
                self._t.join(timeout=8.0)
                if self._t.is_alive():
                    print("[Video] ★ 佇列 8s 內沒清完, 放棄剩餘的幀")
        except Exception:
            pass
        if self.backend == 'nvenc' and self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10.0)
            except Exception:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        elif self._cv is not None:
            try:
                self._cv.release()
            except Exception:
                pass

        # ── 真實 fps 統計 ──
        real_fps = None
        span = None
        if self.t_first and self.t_last and self.t_last > self.t_first:
            span = self.t_last - self.t_first
            real_fps = self.written / span
        backlog = self.enqueued - self.written - self.dropped
        msg = (f"[Video] 收尾: 入列 {self.enqueued}, "
               f"寫入 {self.written}, 丟棄 {self.dropped}, "
               f"收尾時仍在佇列 {backlog}")
        if self.died_at_t is not None:
            msg += (f"\n[Video]   ★ 寫入在第 "
                    f"{self.died_at_t:.1f}s 停止")
        if self.dropped and self.drop_events:
            _t0 = self.t_first or self.drop_events[0][0]
            _win = {}
            for _te, _ql, _w in self.drop_events:
                _win[int(_te - _t0)] = _win.get(int(_te - _t0), 0) + 1
            _top = sorted(_win.items(), key=lambda kv: -kv[1])[:5]
            msg += ("\n[Video]   掉幀集中在 t=" +
                    ", ".join(f"{k}s({v}幀)" for k, v in _top))
        if real_fps:
            play_s = self.written / max(self.fps, 1)
            msg += (f"\n[Video]   真實時長 {span:.1f}s "
                    f"({int(span // 60)}:{span % 60:04.1f}), "
                    f"真實 fps={real_fps:.2f} (檔頭標 {self.fps})"
                    f"\n[Video]   不修正的話播放時間會是 {play_s:.1f}s "
                    f"({int(play_s // 60)}:{play_s % 60:04.1f}) — "
                    f"誤差 {(play_s / span - 1) * 100:+.1f}%")
        if self.error:
            msg += f"\n[Video]   ★ 錯誤: {self.error}"
        print(msg)

        # ── 用真實 fps 重新封裝 (無損, 只改時基) ──
        if real_fps:
            self._write_fps_sidecar(real_fps, span)
        if (self.remux_on_release and real_fps and self.written > 30
                and abs(real_fps - self.fps) / max(self.fps, 1) > 0.02):
            self._remux(real_fps)

        return real_fps

    def _write_fps_sidecar(self, real_fps, span):
        """把真實 fps 寫成旁檔; 機上沒有 ffmpeg 時資訊才不會遺失."""
        try:
            scale = float(self.fps) / real_fps
            base = os.path.splitext(self.path)[0]
            with open(base + '.fps.txt', 'w') as fp:
                fp.write('nominal_fps=%d\n' % self.fps)
                fp.write('real_fps=%.4f\n' % real_fps)
                fp.write('frames=%d\n' % self.written)
                fp.write('real_duration_s=%.3f\n' % span)
                fp.write('itsscale=%.6f\n' % scale)
                fp.write('first_frame_id=%s\n' %
                         ('' if self.first_frame_id is None
                          else self.first_frame_id))
                fp.write('dropped=%d\n' % self.dropped)
                fp.write("fix_cmd=ffmpeg -itsscale %.6f -i '%s' -c copy "
                         "'%s_fixed.mp4'\n" % (scale, self.path, base))
        except Exception:
            pass

    def _remux_gst(self, real_fps):
        """沒有 ffmpeg 時的退路: 用 gst-launch 重新封裝並改寫 framerate caps.

        Jetson 上本來就有 gst-launch (NVENC 就是它跑的), 不必裝任何東西.
        h264parse 可以接受下游指定的 framerate, mp4mux 會照這個寫時基.

        ★ 安全設計: 這條路我無法在出貨前驗證輸出正確性, 所以【不覆蓋原檔】,
          只另外產生 <name>_fixed.mp4. 原檔永遠保留.
        """
        gst = shutil.which('gst-launch-1.0')
        if gst is None:
            return False
        base = os.path.splitext(self.path)[0]
        out = base + '_fixed.mp4'
        num = max(1, int(round(real_fps * 1000)))
        cmd = [gst, '-q',
               'filesrc', f'location={self.path}', '!',
               'qtdemux', '!', 'h264parse', '!',
               f'video/x-h264,framerate={num}/1000', '!',
               'mp4mux', '!', 'filesink', f'location={out}']
        try:
            r = subprocess.run(cmd, timeout=300,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            if r.returncode == 0 and os.path.exists(out) and \
                    os.path.getsize(out) > 1000:
                print(f'[Video]   ✓ gst 已產生時基修正版 (原檔保留): {out}')
                print(f'[Video]     請開來確認長度是否 ≈ '
                      f'{self.written / real_fps:.1f}s; 不對就用原檔.')
                return True
            if os.path.exists(out):
                os.remove(out)
        except Exception as e:
            print(f'[Video]   gst 重新封裝失敗: {e}')
            try:
                if os.path.exists(out):
                    os.remove(out)
            except Exception:
                pass
        return False

    def _remux(self, real_fps):
        """改時基讓播放時間 = 實際飛行時間 (不重新編碼).

        優先 ffmpeg -itsscale (原地覆蓋, 已驗證);
        沒有就退到 gst-launch (另存 _fixed.mp4, 不覆蓋).
        """
        if shutil.which('ffmpeg') is None:
            print('[Video]   ※ 沒有 ffmpeg — 真實 fps 已寫入 .fps.txt 旁檔.')
            if self._remux_gst(real_fps):
                return
            print('[Video]     (gst 退路也沒成功; 時基請在其他機器上修)')
            print(f"[Video]   (沒有 ffmpeg, 無法自動修正) 手動指令:\n"
                  f"          ffmpeg -itsscale {float(self.fps)/real_fps:.6f} "
                  f"-i '{self.path}' -c copy "
                  f"'{os.path.splitext(self.path)[0]}_fixed.mp4'")
            return
        base, ext = os.path.splitext(self.path)
        tmp = f"{base}_fixfps{ext}"
        # ★ 用 -itsscale 縮放輸入時戳 (不是 -r; -r 對已有時戳的 mp4 無效).
        #   scale = 檔頭標稱 fps / 真實 fps
        #   例: 標 25 實際 12.7 → scale 1.97 → 影片長度變成 1.97 倍 = 真實時長
        scale = float(self.fps) / real_fps
        cmd = ['ffmpeg', '-y', '-loglevel', 'error',
               '-itsscale', f'{scale:.6f}', '-i', self.path,
               '-c', 'copy', tmp]
        try:
            r = subprocess.run(cmd, timeout=180)
            if (r.returncode == 0 and os.path.exists(tmp)
                    and os.path.getsize(tmp) > 0):
                os.replace(tmp, self.path)
                print(f"[Video]   ✓ 已用真實 fps {real_fps:.2f} 重新封裝, "
                      f"播放時間現在對得上 log 的 t")
            else:
                print(f"[Video]   ✗ 重新封裝失敗 (rc={r.returncode}), 原檔保留")
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception as e:
            print(f"[Video]   ✗ 重新封裝例外: {e}, 原檔保留")
