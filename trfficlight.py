import os
import cv2
import numpy as np
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

# 抑制輸出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from ultralytics import YOLO
from tensorflow.keras.models import load_model
import pyttsx3

# ================= 配置設定 (Config) =================
@dataclass
class Config:
    # --- 檔案路徑 ---
    VIDEO_PATH: str = 1
    YOLO_MODEL_PATH: str = "Model/traffic_1280.pt"
    CNN_MODEL_PATH: str = "cnn_digit_model_new.h5"
    
    # --- 偵測參數 ---
    YOLO_IMGSZ: int = 640
    CONF_THRESHOLD: float = 0.5
    YOLO_MIN_CONF: float = 0.7
    CNN_MIN_CONF: float = 0.7
    PROCESS_EVERY_N: int = 3
    QUEUE_MAX: int = 5
    
    # --- 邏輯與時間參數 ---
    TIMEOUT_LOCK: float = 2.0       # 訊號消失多久後視為斷線
    TTS_INTERVAL: float = 2.0       # 倒數計時語音間隔
    TTS_RESET_INTERVAL: float = 8.0 # 完全離開路口多久重置 (此參數現在為輔助用)
    
    # --- 通訊編碼 ---
    CODE_GREEN: int = int("01000001", 2)      
    CODE_RED: int = int("01000011", 2)        
    CODE_COUNTDOWN: int = int("01000010", 2) 

    # --- 系統設定 ---
    IMG_SIZE: Tuple[int, int] = (28, 28)

# ================= 語音模組 (TTS) =================
class TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.engine = None
        self.last_spoken = {}

    def run(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 130)
            self.engine.setProperty('volume', 2.0)
            self.engine.startLoop(False)
        except: pass

        while not self.stop_event.is_set():
            try:
                self.engine.iterate()
                try:
                    msg = self.queue.get_nowait()
                    if msg:
                        print(f"[Voice]: {msg}")
                        self.engine.say(msg)
                except queue.Empty:
                    pass
                time.sleep(0.05)
            except: time.sleep(0.1)
        try: self.engine.endLoop()
        except: pass

    def speak(self, msg: str, key: str = "general", interval: float = 0):
        now = time.time()
        last_time = self.last_spoken.get(key, 0)
        if now - last_time > interval:
            self.queue.put(msg)
            self.last_spoken[key] = now

    def stop(self):
        self.stop_event.set()

# ================= 狀態管理模組 (修正同步問題) =================
class TrafficStateManager:
    def __init__(self, config: Config, tts: TTSWorker):
        self.cfg = config
        self.tts = tts
        
        self.lights = {
            "green": {"state": False, "last_seen": 0},
            "red":   {"state": False, "last_seen": 0}
        }
        
        self.countdown = {
            "active": False, "value": 0, "last_tick": 0, "last_digit_seen": None
        }
        
        self.cnn_enabled = True
        self.prev_light_tts = None 
        self.last_active_time = time.time()

    def update_detection(self, detections: dict):
        now = time.time()
        any_light_active = False 

        # 1. 更新 Last Seen
        for key in ["green", "red"]:
            if detections.get(key):
                self.lights[key]["last_seen"] = now

        # 2. 處理狀態與鎖定
        for key in ["green", "red"]:
            is_detected = detections.get(key, False)
            current_state = self.lights[key]["state"]
            
            if is_detected:
                if not current_state:
                    self._set_light_state(key, True)
            else:
                if current_state:
                    # 超時檢查 (3秒)
                    if now - self.lights[key]["last_seen"] > self.cfg.TIMEOUT_LOCK:
                        self._set_light_state(key, False)
            
            if self.lights[key]["state"]:
                any_light_active = True

        # 離開路口輔助檢查 (當所有燈都滅了很久)
        if any_light_active:
            self.last_active_time = now
        else:
            if now - self.last_active_time > self.cfg.TTS_RESET_INTERVAL:
                if self.prev_light_tts is not None:
                    # print(f"--- [Reset] 完全離開路口 ---")
                    self.prev_light_tts = None 

        # 3. 倒數計時與語音
        self._handle_countdown(detections.get('digit'))
        self._trigger_tts()

    def _set_light_state(self, key, new_state):
        self.lights[key]["state"] = new_state
        code = self.cfg.CODE_GREEN if key == "green" else self.cfg.CODE_RED
        
        if new_state:
            # 狀態：開啟 (硬體發送訊號)
            print(f"SEND: {code} ({key})")
        else:
            # 狀態：關閉 (訊號丟失/超時)
            # *** 關鍵修正：當燈號斷線時，若它是上次播報的燈號，則清除記憶 ***
            # 這樣下次重新連上 (SEND) 時，語音就會再次觸發
            if self.prev_light_tts == key:
                self.prev_light_tts = None

    def _handle_countdown(self, digit):
        if self.countdown["active"]:
            if time.time() - self.countdown["last_tick"] >= 1.0:
                self.countdown["value"] -= 1
                self.countdown["last_tick"] = time.time()
                if self.countdown["value"] <= 0:
                    self._reset_countdown()

        if digit is not None:
            if self.countdown["last_digit_seen"] == 11 and digit == 10:
                print(f"SEND: {self.cfg.CODE_COUNTDOWN} (countdown_start)")
                self.countdown.update({"active": True, "value": 10, "last_tick": time.time()})
                self.cnn_enabled = False
                if "fast" in self.tts.last_spoken: 
                    self.tts.last_spoken["fast"] = 0 
            self.countdown["last_digit_seen"] = digit
        
        if not self.countdown["active"]:
            self.cnn_enabled = True

    def _reset_countdown(self):
        self.countdown.update({"active": False, "value": 0})
        self.cnn_enabled = True
        self.countdown["last_digit_seen"] = None

    def _trigger_tts(self):
        current_light = "red" if self.lights["red"]["state"] else \
                        "green" if self.lights["green"]["state"] else None

        # 播報邏輯：只要當前燈號與「記憶中」的不同，就播報
        # 由於我們在 _set_light_state 中加入了斷線清除記憶的邏輯，
        # 所以只要有新的 SEND (斷線重連)，這裡就會判定為「不同」，進而播報。
        if current_light and current_light != self.prev_light_tts:
            if not self.countdown["active"] or self.countdown["value"] > 5:
                msg = "紅燈請停下" if current_light == "red" else "綠燈可以走"
                self.tts.speak(msg)
                self.prev_light_tts = current_light

        # 倒數 10 秒
        if self.countdown["active"] and self.countdown["value"] == 10:
            prefix = "紅燈" if current_light == "red" else \
                     "綠燈" if current_light == "green" else ""
            self.tts.speak(f"{prefix}剩餘10秒", key="fast", interval=2.0)

    def get_display_info(self):
        return {"countdown_val": self.countdown["value"] if self.countdown["active"] else None}

# ================= 影像偵測與主程式 (不變) =================
class TrafficDetector:
    def __init__(self, config: Config):
        self.cfg = config
        self.yolo = YOLO(config.YOLO_MODEL_PATH)
        self.cnn = load_model(config.CNN_MODEL_PATH)
        for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]: os.environ[k] = "1"
        try: cv2.setNumThreads(1); import torch; torch.set_num_threads(1)
        except: pass

    def crop_digits(self, img, max_digits=2, min_w=10):
        if img is None or img.size == 0: return []
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(g, 50, 255, cv2.THRESH_BINARY)
        col_sum = np.sum(th, axis=0)
        if np.all(col_sum == 0): return []
        l, r = np.argmax(col_sum > 0), len(col_sum) - np.argmax(col_sum[::-1] > 0)
        if r <= l: return []
        w = r - l
        num = min(max_digits, max(1, w // min_w))
        return [img[:, i * w // num : (i + 1) * w // num] for i in range(num)]

    def process_frame(self, frame, cnn_enabled=True):
        detections = {"green": False, "red": False, "digit": None}
        boxes_to_draw = []
        try: results = self.yolo(frame, imgsz=self.cfg.YOLO_IMGSZ, conf=self.cfg.CONF_THRESHOLD, verbose=False)
        except: return detections, boxes_to_draw

        for r in results:
            if not r.boxes: continue
            for box, cls, score in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if score < self.cfg.YOLO_MIN_CONF: continue
                x1, y1, x2, y2 = map(int, box); c = int(cls)
                label = ""
                if c == 0: 
                    label = "cnt"
                    if cnn_enabled:
                        crop = frame[y1:y2, x1:x2]
                        digits = self.crop_digits(crop)
                        batch = [cv2.resize(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), (28,28))/255.0 for d in digits if np.mean(d)>50]
                        if batch:
                            p = self.cnn.predict(np.array(batch).reshape(-1,28,28,1), verbose=0)
                            res = [np.argmax(x) for x in p if np.max(x) >= self.cfg.CNN_MIN_CONF]
                            if res: detections["digit"] = int("".join(map(str, res))); label=f"Num:{detections['digit']}"
                    boxes_to_draw.append((label, score, (x1, y1, x2, y2), c))
                elif c == 1: detections["green"]=True; boxes_to_draw.append(("Green", score, (x1,y1,x2,y2), c))
                elif c == 2: detections["red"]=True; boxes_to_draw.append(("Red", score, (x1,y1,x2,y2), c))
        return detections, boxes_to_draw

def main():
    cfg = Config()
    tts = TTSWorker(); tts.start()
    detector = TrafficDetector(cfg)
    state_manager = TrafficStateManager(cfg, tts)
    
    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    frame_q = queue.Queue(maxsize=cfg.QUEUE_MAX)
    stop_event = threading.Event()

    def capture_loop():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret: frame_q.put(None); break
            try: frame_q.put(frame, timeout=0.1)
            except: frame_q.get_nowait(); frame_q.put(frame)
        cap.release()
    threading.Thread(target=capture_loop, daemon=True).start()

    print(f"System Ready. Reset Interval: {cfg.TTS_RESET_INTERVAL}s. 'q' to exit.")
    cv2.namedWindow("Traffic Eye", cv2.WINDOW_NORMAL)
    
    idx = 0
    try:
        while True:
            try: frame = frame_q.get(timeout=0.5)
            except: continue
            if frame is None: break
            idx += 1
            
            boxes = []
            if idx % cfg.PROCESS_EVERY_N == 0:
                dets, boxes = detector.process_frame(frame, state_manager.cnn_enabled)
                state_manager.update_detection(dets)

            for lab, sc, (x1,y1,x2,y2), cid in boxes:
                color = (0,255,0) if cid==1 else (0,0,255) if cid==2 else (0,255,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,lab,(x1,y1-5),0,0.6,color,2)
            
            inf = state_manager.get_display_info()
            if inf["countdown_val"]: cv2.putText(frame,f"CNT: {inf['countdown_val']}",(30,80),0,2,(0,200,255),3)
            
            cv2.imshow("Traffic Eye", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
    finally:
        stop_event.set(); tts.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()