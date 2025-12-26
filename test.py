import os
import cv2
import numpy as np
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

# 設定環境變數，抑制 TensorFlow 和 Ultralytics 的冗餘輸出，保持終端機乾淨
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from ultralytics import YOLO
from tensorflow.keras.models import load_model
import pyttsx3

# ==========================================
# 1. 系統參數配置 (Config)
# ==========================================
@dataclass
class Config:
    """
    集中管理所有參數，方便後續調整靈敏度與路徑
    """
    # --- 檔案路徑設定 ---
    VIDEO_PATH: str = "video/IMG_3396.MOV"             # 測試影片路徑 (若要用 WebCam 請改成 0)
    YOLO_MODEL_PATH: str = "Model/traffic_1280.pt"  # 您訓練的專用模型 (紅綠燈、倒數讀秒)
    COMMON_MODEL_PATH: str = "yolov8n.pt"           # 官方通用模型 (人、車) - 第一次執行會自動下載
    CNN_MODEL_PATH: str = "cnn_digit_model_new.h5"  # 用於辨識倒數數字的 CNN 模型
    
    # --- YOLO 偵測參數 ---
    YOLO_IMGSZ: int = 1280         # 紅綠燈模型解析度 (維持高解析，確保能抓到遠處小號誌)
    
    # [優化重點] 降低通用模型解析度
    # 因為人與車輛體積較大，用 320x320 辨識通常足夠，且速度能快 2~3 倍
    YOLO_COMMON_IMGSZ: int = 320  
    
    CONF_THRESHOLD: float = 0.5   # 紅綠燈的信心門檻 (低於此分數不採信)
    YOLO_MIN_CONF: float = 0.5    # 人車的信心門檻
    CNN_MIN_CONF: float = 0.7     # CNN 數字辨識的信心門檻 (需較高以防誤判)
    
    # --- [核心優化] 錯峰運算 (Load Balancing) 設定 ---
    # 將繁重的 AI 運算分散到不同幀數執行，避免單一幀卡頓
    PROCESS_CYCLE: int = 3        # 處理週期：每 3 幀為一個循環
    FRAME_TRAFFIC: int = 0        # 第 0 幀：執行紅綠燈偵測 (高負載)
    FRAME_COMMON: int = 1         # 第 1 幀：執行人車偵測 (中負載)
                                  # 第 2 幀：休息 (只做邏輯判斷與繪圖，極低負載)
    
    QUEUE_MAX: int = 5            # 影像緩衝區大小 (太大會造成延遲，太小會掉幀)
    
    # 設定要偵測的障礙物類別 (對應 COCO 資料集 ID)
    # 0:人, 1:腳踏車, 2:汽車, 3:機車, 5:公車, 7:卡車
    OBSTACLE_CLASSES: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    
    # --- 邏輯與時間控制參數 ---
    TIMEOUT_LOCK: float = 3.0           # 訊號鎖定時間：若紅綠燈閃爍或短暫被遮，3秒內仍視為有效
    TTS_INTERVAL: float = 2.0           # 倒數計時語音的最短間隔 (避免每秒都唸，太吵)
    TTS_RESET_INTERVAL: float = 8.0     # (輔助) 若超過 8 秒完全沒看到燈，重置語音記憶
    OBSTACLE_TTS_INTERVAL: float = 15.0 # 障礙物提醒冷卻時間：每 20 秒最多提醒一次 (避免一直報菜名)
    
    # --- [隧道視野] 過濾器參數 ---
    # 用於判斷「是否真的構成危險」，只有同時滿足以下條件才會語音警告
    MIN_OBS_HEIGHT_RATIO: float = 0.6   # 高度門檻：物體高度必須佔畫面 40% 以上 (代表很近)
    ROI_CENTER_MIN: float = 0.25        # 左邊界：畫面寬度的 25%
    ROI_CENTER_MAX: float = 0.75        # 右邊界：畫面寬度的 75%
                                        # 結論：只看畫面正中央 50% 區域且很近的物體
    
    # --- 通訊編碼 (模擬硬體訊號) ---
    CODE_GREEN: int = int("01000001", 2)      # 65 (ASCII 'A')
    CODE_RED: int = int("01000011", 2)        # 67 (ASCII 'C')
    CODE_COUNTDOWN: int = int("01000010", 2)  # 66 (ASCII 'B')

    # CNN 模型的輸入圖片大小
    IMG_SIZE: Tuple[int, int] = (28, 28)

# ==========================================
# 2. 語音處理模組 (TTS Worker)
# ==========================================
class TTSWorker(threading.Thread):
    """
    獨立執行緒負責語音輸出，避免 pyttsx3 卡住主程式的影像辨識
    """
    def __init__(self):
        super().__init__(daemon=True) # daemon=True 代表主程式結束時，此執行緒也會自動結束
        self.queue = queue.Queue()    # 存放要說的話
        self.stop_event = threading.Event()
        self.engine = None
        self.last_spoken = {}         # 記錄每種語音類型上次說話的時間 (用於冷卻)

    def run(self):
        # 初始化語音引擎
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 130)   # 語速
            self.engine.setProperty('volume', 2.0) # 音量
            self.engine.startLoop(False)           # 啟動非阻塞循環
        except Exception as e:
            print(f"[TTS Init Warning]: {e}")

        # 無窮迴圈，等待語音指令
        while not self.stop_event.is_set():
            try:
                self.engine.iterate() # 驅動引擎發聲
                try:
                    msg = self.queue.get_nowait()
                    if msg:
                        print(f"[Voice]: {msg}")
                        self.engine.say(msg)
                except queue.Empty:
                    pass
                time.sleep(0.05) # 短暫休眠避免吃滿 CPU
            except: time.sleep(0.1)
        
        try: self.engine.endLoop()
        except: pass

    def speak(self, msg: str, key: str = "general", interval: float = 0):
        """
        對外接口：請求說話
        key: 語音類別 (如 'obs_car', 'fast' 等)
        interval: 該類別的最小間隔時間 (冷卻時間)
        """
        now = time.time()
        last_time = self.last_spoken.get(key, 0)
        # 只有超過冷卻時間才允許加入佇列
        if now - last_time > interval:
            self.queue.put(msg)
            self.last_spoken[key] = now

    def stop(self):
        self.stop_event.set()

# ==========================================
# 3. 狀態管理模組 (TrafficStateManager)
# ==========================================
class TrafficStateManager:
    """
    核心大腦：負責整合偵測結果、判斷紅綠燈狀態、倒數計時邏輯、以及決定何時發出語音。
    """
    def __init__(self, config: Config, tts: TTSWorker):
        self.cfg = config
        self.tts = tts
        
        # 燈號狀態資料結構
        # state: 是否亮燈 (True/False)
        # last_seen: 最後一次偵測到的時間 (用於 Timeout 機制)
        # box: 畫面上的框座標 (用於快取繪圖，防閃爍)
        self.lights = {
            "green": {"state": False, "last_seen": 0, "box": None},
            "red":   {"state": False, "last_seen": 0, "box": None}
        }
        
        # 倒數計時器狀態
        self.countdown = {
            "active": False, "value": 0, "last_tick": 0, 
            "last_digit_seen": None, "box": None
        }
        
        self.cnn_enabled = True       # 是否啟用 CNN (倒數開始後會暫停 CNN 以省資源)
        self.prev_light_tts = None    # 記憶上次播報的燈號 (避免重複播報)
        self.last_active_time = time.time()

    def update_detection(self, detections: dict, obstacles: dict, traffic_boxes: List[Tuple]):
        """
        接收偵測結果 (可能是新的，也可能是快取的)，更新系統狀態
        """
        now = time.time()
        any_light_active = False 

        # --- 步驟 1: 更新框座標 (Box) 與 最後目擊時間 (Last Seen) ---
        # 這裡為了查找方便，先將 List 轉為 Dict
        current_boxes = {}
        for label, score, box, cid in traffic_boxes:
            if cid == 101: current_boxes["green"] = box
            elif cid == 102: current_boxes["red"] = box
            elif cid == 100: self.countdown["box"] = box 

        for key in ["green", "red"]:
            # 如果這一幀偵測器有看到 (detections[key] 為 True)
            if detections.get(key):
                self.lights[key]["last_seen"] = now           # 更新時間
                self.lights[key]["box"] = current_boxes.get(key) # 更新框的位置

        # --- 步驟 2: 狀態鎖定邏輯 (Timeout Lock) ---
        # 防止因樹葉遮擋或偵測跳動導致訊號忽開忽關
        for key in ["green", "red"]:
            is_detected = detections.get(key, False)
            current_state = self.lights[key]["state"]
            
            if is_detected:
                # 看到燈了 -> 立刻開啟狀態
                if not current_state:
                    self._set_light_state(key, True)
            else:
                # 沒看到燈 -> 檢查是否還在「鎖定時間」內
                if current_state:
                    if now - self.lights[key]["last_seen"] > self.cfg.TIMEOUT_LOCK:
                        # 超過 3 秒沒看到 -> 視為斷線，關閉狀態
                        self._set_light_state(key, False)
                        self.lights[key]["box"] = None 
            
            if self.lights[key]["state"]:
                any_light_active = True

        # --- 步驟 3: 離開路口判定 ---
        if any_light_active:
            self.last_active_time = now
        else:
            # 如果超過 8 秒完全沒訊號，強制重置語音記憶
            if now - self.last_active_time > self.cfg.TTS_RESET_INTERVAL:
                if self.prev_light_tts is not None:
                    self.prev_light_tts = None 
        
        # 清除過期的倒數框
        if not self.countdown["active"] and (now - self.countdown.get("last_tick", 0) > 2.0):
             self.countdown["box"] = None

        # --- 步驟 4: 執行子邏輯 ---
        self._handle_countdown(detections.get('digit'))
        self._trigger_tts(obstacles)

    def _set_light_state(self, key, new_state):
        """切換燈號狀態，並發送模擬訊號"""
        self.lights[key]["state"] = new_state
        code = self.cfg.CODE_GREEN if key == "green" else self.cfg.CODE_RED
        if new_state:
            print(f"SEND: {code} ({key})")
        else:
            # 關鍵邏輯：當燈號斷線 (變成 False) 時，清除語音記憶
            # 這樣下次再連上時，系統會視為新事件，再次播報語音
            if self.prev_light_tts == key:
                self.prev_light_tts = None

    def _handle_countdown(self, digit):
        """處理倒數計時邏輯"""
        # 1. 自動倒數 (每秒 -1)
        if self.countdown["active"]:
            if time.time() - self.countdown["last_tick"] >= 1.0:
                self.countdown["value"] -= 1
                self.countdown["last_tick"] = time.time()
                if self.countdown["value"] <= 0:
                    self._reset_countdown()

        # 2. 觸發倒數 (當偵測到數字從 11 變 10)
        if digit is not None:
            if self.countdown["last_digit_seen"] == 11 and digit == 10:
                print(f"SEND: {self.cfg.CODE_COUNTDOWN} (countdown_start)")
                self.countdown.update({"active": True, "value": 10, "last_tick": time.time()})
                self.cnn_enabled = False # 進入自動倒數，關閉 CNN 省效能
                
                # 重置 10秒提醒的冷卻，確保這次能播報
                if "fast" in self.tts.last_spoken: 
                    self.tts.last_spoken["fast"] = 0 
            self.countdown["last_digit_seen"] = digit
        
        if not self.countdown["active"]:
            self.cnn_enabled = True

    def _reset_countdown(self):
        self.countdown.update({"active": False, "value": 0})
        self.cnn_enabled = True
        self.countdown["last_digit_seen"] = None

    def _trigger_tts(self, obstacles):
        """決定要播放什麼語音"""
        current_light = "red" if self.lights["red"]["state"] else \
                        "green" if self.lights["green"]["state"] else None

        # A. 紅綠燈語音 (優先級最高)
        if current_light and current_light != self.prev_light_tts:
            # 如果正在緊急倒數 (剩餘 < 5秒)，先不報燈號，避免干擾
            if not self.countdown["active"] or self.countdown["value"] > 5:
                msg = "紅燈請停下" if current_light == "red" else "綠燈可以走"
                self.tts.speak(msg)
                self.prev_light_tts = current_light

        # B. 倒數 10 秒提醒
        if self.countdown["active"] and self.countdown["value"] == 10:
            prefix = "紅燈" if current_light == "red" else \
                     "綠燈" if current_light == "green" else ""
            self.tts.speak(f"{prefix}剩餘10秒", key="fast", interval=2.0)

        # C. 障礙物提醒 (優先級最低)
        # 條件：無倒數 (或倒數很久) 且 障礙物計數 > 0 (經過嚴格過濾後的計數)
        if not self.countdown["active"] or self.countdown["value"] > 8:
            if obstacles['vehicle'] > 0:
                self.tts.speak("前方有車輛", key="obs_car", interval=self.cfg.OBSTACLE_TTS_INTERVAL)
            elif obstacles['person'] > 0:
                self.tts.speak("小心行人", key="obs_person", interval=self.cfg.OBSTACLE_TTS_INTERVAL)

    def get_stable_boxes(self) -> List[Tuple]:
        """回傳「穩定」的紅綠燈框 (受 Timeout Lock 保護，防閃爍)"""
        boxes = []
        for key, info in self.lights.items():
            if info["state"] and info["box"] is not None:
                cid = 101 if key == "green" else 102
                boxes.append((key.capitalize(), 1.0, info["box"], cid))
        if self.countdown["box"] is not None:
             boxes.append(("CNT", 1.0, self.countdown["box"], 100))
        return boxes
    
    def get_display_info(self):
        return {"countdown_val": self.countdown["value"] if self.countdown["active"] else None}

# ==========================================
# 4. 影像偵測模組 (TrafficDetector)
# ==========================================
class TrafficDetector:
    """
    負責載入模型並執行推論。
    特點：將「紅綠燈偵測」與「障礙物偵測」拆分成兩個獨立函式，支援錯峰運算。
    """
    def __init__(self, config: Config):
        self.cfg = config
        self._init_models()
        # 限制執行緒數，避免與 PyTorch 搶資源
        for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]: os.environ[k] = "1"
        try: cv2.setNumThreads(1); import torch; torch.set_num_threads(1)
        except: pass

    def _init_models(self):
        print(f"Loading Traffic Model: {self.cfg.YOLO_MODEL_PATH}")
        self.yolo_traffic = YOLO(self.cfg.YOLO_MODEL_PATH)
        print(f"Loading Common Model: {self.cfg.COMMON_MODEL_PATH}")
        self.yolo_common = YOLO(self.cfg.COMMON_MODEL_PATH)
        print(f"Loading CNN Model: {self.cfg.CNN_MODEL_PATH}")
        self.cnn = load_model(self.cfg.CNN_MODEL_PATH)

    def crop_digits(self, img, max_digits=2, min_w=10):
        """影像處理工具：將畫面中的數字區域切割出來給 CNN 辨識"""
        if img is None or img.size == 0: return []
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉灰階
        _, th = cv2.threshold(g, 50, 255, cv2.THRESH_BINARY) # 二值化
        
        # 垂直投影法找數字位置
        col_sum = np.sum(th, axis=0)
        if np.all(col_sum == 0): return []
        l, r = np.argmax(col_sum > 0), len(col_sum) - np.argmax(col_sum[::-1] > 0)
        if r <= l: return []
        w = r - l
        num = min(max_digits, max(1, w // min_w))
        return [img[:, i * w // num : (i + 1) * w // num] for i in range(num)]

    # --- 任務 A: 偵測紅綠燈 (Frame 0) ---
    def detect_traffic(self, frame, cnn_enabled=True):
        detections = {"green": False, "red": False, "digit": None}
        traffic_boxes = []
        
        try:
            # 使用高解析度 (640)
            results = self.yolo_traffic(frame, imgsz=self.cfg.YOLO_IMGSZ, conf=self.cfg.CONF_THRESHOLD, verbose=False)
            for r in results:
                if not r.boxes: continue
                for box, cls, score in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box); c = int(cls)
                    label = ""
                    # 類別 0: 倒數讀秒
                    if c == 0: 
                        label = "cnt"
                        if cnn_enabled:
                            # 執行切割與 CNN 辨識
                            crop = frame[y1:y2, x1:x2]
                            digits = self.crop_digits(crop)
                            batch = [cv2.resize(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), (28,28))/255.0 for d in digits if np.mean(d)>50]
                            if batch:
                                p = self.cnn.predict(np.array(batch).reshape(-1,28,28,1), verbose=0)
                                res = [np.argmax(x) for x in p if np.max(x) >= self.cfg.CNN_MIN_CONF]
                                if res: detections["digit"] = int("".join(map(str, res))); label=f"Num:{detections['digit']}"
                        traffic_boxes.append((label, score, (x1, y1, x2, y2), 100)) # ID 100: 黃色
                    # 類別 1: 綠燈
                    elif c == 1: 
                        detections["green"]=True
                        traffic_boxes.append(("Green", score, (x1,y1,x2,y2), 101)) # ID 101: 綠色
                    # 類別 2: 紅燈
                    elif c == 2: 
                        detections["red"]=True
                        traffic_boxes.append(("Red", score, (x1,y1,x2,y2), 102)) # ID 102: 紅色
        except Exception: pass
        return detections, traffic_boxes

    # --- 任務 B: 偵測人車障礙物 (Frame 1) ---
    def detect_obstacles(self, frame):
        obstacles = {"person": 0, "vehicle": 0}
        obstacle_boxes = []
        h, w = frame.shape[:2]
        
        try:
            # 使用低解析度 (320) 以加速運算
            results_c = self.yolo_common(frame, imgsz=self.cfg.YOLO_COMMON_IMGSZ, conf=self.cfg.YOLO_MIN_CONF, classes=self.cfg.OBSTACLE_CLASSES, verbose=False)
            for r in results_c:
                if not r.boxes: continue
                for box, cls, score in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box); c = int(cls)
                    label_text = "Object"
                    color_id = 0
                    
                    # [過濾器] 計算物體高度與中心位置
                    obj_h = y2 - y1
                    obj_center_x = (x1 + x2) / 2
                    
                    # 條件 1: 夠大 (代表夠近)
                    is_close = (obj_h / h) > self.cfg.MIN_OBS_HEIGHT_RATIO
                    # 條件 2: 在畫面正中央
                    is_center = (obj_center_x > w * self.cfg.ROI_CENTER_MIN) and \
                                (obj_center_x < w * self.cfg.ROI_CENTER_MAX)
                    
                    # 同時滿足才觸發警告
                    trigger_alert = is_close and is_center

                    if c == 0: # Person
                        label_text = "Person"
                        color_id = 200
                        if trigger_alert: obstacles["person"] += 1 
                    elif c in [1, 2, 3, 5, 7]: # Vehicles
                        label_text = "Vehicle"
                        color_id = 201
                        if trigger_alert: obstacles["vehicle"] += 1 
                    
                    if trigger_alert: 
                        label_text += " (!)" # 視覺標記驚嘆號
                        color_id = 200 if c==0 else 201
                    obstacle_boxes.append((label_text, score, (x1, y1, x2, y2), color_id))
        except Exception: pass
        return obstacles, obstacle_boxes

# ==========================================
# 5. 主程式入口 (Main)
# ==========================================
def main():
    cfg = Config()
    
    # 初始化各模組
    tts = TTSWorker(); tts.start() # 啟動語音執行緒
    detector = TrafficDetector(cfg)
    state_manager = TrafficStateManager(cfg, tts)
    
    # 開啟影像串流
    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    frame_q = queue.Queue(maxsize=cfg.QUEUE_MAX)
    stop_event = threading.Event()

    # 獨立執行緒讀取影像 (避免 I/O 阻塞)
    def capture_loop():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret: frame_q.put(None); break
            try: frame_q.put(frame, timeout=0.1)
            except: frame_q.get_nowait(); frame_q.put(frame) # 若滿了則丟棄舊幀 (Drop frame)
        cap.release()
    threading.Thread(target=capture_loop, daemon=True).start()

    print(f"系統就緒。模式: 錯峰運算 (Interleaved)。請按 'q' 離開。")
    cv2.namedWindow("Traffic Eye", cv2.WINDOW_NORMAL)
    
    idx = 0
    
    # --- 快取變數 (Cache) ---
    # 因為我們採用錯峰運算，所以需要變數來記住「上一次」的結果
    # 這樣在沒有跑模型的那一幀，畫面框框才不會消失
    cache_traffic_dets = {"green": False, "red": False, "digit": None}
    cache_traffic_boxes = []
    cache_obstacles = {"person": 0, "vehicle": 0}
    cache_obstacle_boxes = []

    try:
        while True:
            # 從佇列取出影像
            try: frame = frame_q.get(timeout=0.5)
            except: continue
            if frame is None: break
            
            idx += 1
            # 計算目前是循環中的哪一幀 (0, 1, 2)
            cycle_idx = idx % cfg.PROCESS_CYCLE 
            
            # --- 核心優化：錯峰運算 (Load Balancing) ---
            
            # [Frame 0]: 執行紅綠燈偵測 (高解析度)
            if cycle_idx == cfg.FRAME_TRAFFIC:
                cache_traffic_dets, cache_traffic_boxes = detector.detect_traffic(frame, state_manager.cnn_enabled)
                
            # [Frame 1]: 執行人車偵測 (低解析度)
            elif cycle_idx == cfg.FRAME_COMMON:
                cache_obstacles, cache_obstacle_boxes = detector.detect_obstacles(frame)
            
            # [Frame 2]: 休息幀 (什麼都不做，直接用 Cache)
            
            # --- 更新狀態機 ---
            # 無論這一幀有沒有跑模型，都將目前的 Cache 資料餵給狀態機
            # 狀態機會負責處理時間 Timeout 和邏輯判斷
            state_manager.update_detection(cache_traffic_dets, cache_obstacles, cache_traffic_boxes)
            
            # --- 畫面繪圖 ---
            # 1. 取得穩定的紅綠燈框 (State Manager 會過濾掉閃爍的框)
            stable_traffic_boxes = state_manager.get_stable_boxes()
            
            # 2. 合併人車框 (直接用 Cache)
            # 因為人車偵測是錯峰的，所以框框可能會延遲 1-2 幀，但人眼幾乎看不出來
            all_boxes = stable_traffic_boxes + cache_obstacle_boxes
            
            # 3. 繪製 ROI 參考線 (灰色直線，顯示隧道視野範圍)
            h, w = frame.shape[:2]
            cv2.line(frame, (int(w*cfg.ROI_CENTER_MIN), 0), (int(w*cfg.ROI_CENTER_MIN), h), (100,100,100), 1)
            cv2.line(frame, (int(w*cfg.ROI_CENTER_MAX), 0), (int(w*cfg.ROI_CENTER_MAX), h), (100,100,100), 1)

            # 4. 畫出所有的框
            for lab, sc, (x1,y1,x2,y2), cid in all_boxes:
                if cid == 101: color = (0, 255, 0)     # 綠燈
                elif cid == 102: color = (0, 0, 255)   # 紅燈
                elif cid == 100: color = (0, 255, 255) # 倒數 (黃)
                elif cid == 200: color = (255, 100, 0) # 人 (藍橘)
                elif cid == 201: color = (200, 200, 200) # 車 (灰)
                else: color = (255, 255, 255)
                
                # 如果是有驚嘆號 (!) 的危險物體，框畫粗一點
                thickness = 3 if "(!)" in lab else 2
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,thickness)
                cv2.putText(frame,lab,(x1,y1-5),0,0.6,color,2)
            
            # 5. 顯示倒數數值
            inf = state_manager.get_display_info()
            if inf["countdown_val"]: 
                cv2.putText(frame,f"CNT: {inf['countdown_val']}",(30,80),0,2,(0,200,255),3)
            
            cv2.imshow("Traffic Eye", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            
    finally:
        # 程式結束前的清理工作
        stop_event.set()
        tts.stop()
        cv2.destroyAllWindows()
        print("系統已關閉。")

if __name__ == "__main__":
    main()