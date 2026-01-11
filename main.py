import os
import cv2
import numpy as np
import threading
import queue
import time
import traceback
import gc
import base64
import copy
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

# 設定環境變數
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ★★★ 引入必要套件 ★★★
try:
    import requests
    from signalrcore.hub_connection_builder import HubConnectionBuilder
except ImportError:
    print("❌ 錯誤：請先安裝必要套件！輸入指令: pip install requests signalrcore")
    sys.exit(1)

from ultralytics import YOLO
try:
    from tensorflow.keras.models import load_model 
    import tensorflow as tf
except ImportError:
    print("警告: 找不到 TensorFlow，將停用 CNN 數字辨識功能。")
    load_model = None

import pyttsx3

# ==========================================
# 1. 系統參數配置 (Config)
# ==========================================
@dataclass
class Config:
    # -----------------------------------------------------------
    # [1] 影像輸入 (ESP32 IP)
    # -----------------------------------------------------------
    CAMERA_SOURCE: any = 1  
    
    # -----------------------------------------------------------
    # [2] 自動登入設定 (依照範例檔修正)
    # -----------------------------------------------------------
    LOGIN_URL: str = "https://sbas.runasp.net/api/AccountApi/Login"
    
    # ★★★ 請填入你的帳號密碼 (程式會自動登入拿 Token) ★★★
    USER_EMAIL: str = "TestUser1@gm.chihlee.edu.tw"  # 請填入實際帳號
    USER_PASSWORD: str = "Aa000000!"   # 請填入實際密碼
    
    # -----------------------------------------------------------
    # [3] SignalR 連線設定 (依照範例檔修正)
    # -----------------------------------------------------------
    # Hub 網址
    HUB_URL: str = "https://sbas.runasp.net/chathub"
    
    # ★★★ 關鍵修改 1: 後端方法名稱 ★★★
    # 根據範例檔，後端接收的方法叫做 "SendLiveStream"
    HUB_METHOD_NAME: str = "SendLiveStream"
    
    # 是否啟用傳送功能
    ENABLE_STREAMING: bool = True
    
    NEED_ROTATION: bool = False 
    
    # 模型路徑
    MODEL_TRAFFIC: str = "Model/traffic&count.pt" 
    MODEL_ZEBRA: str = "Model/zebra_v3.pt"            
    MODEL_CNN: str = "Model/cnn_digit_model_new.h5" 
    
    # 解析度與效能
    IMGSZ_TRAFFIC: int = 640    
    IMGSZ_ZEBRA: int = 640      
    QUEUE_MAX: int = 1 
    
    # 信心門檻
    CONF_TRAFFIC: float = 0.8    
    CONF_ZEBRA: float = 0.5     
    CONF_CNN: float = 0.7        
    
    # 綠色人行道與斑馬線
    LOWER_GREEN: np.ndarray = field(default_factory=lambda: np.array([75, 40, 40]))   
    UPPER_GREEN: np.ndarray = field(default_factory=lambda: np.array([92, 255, 255])) 
    ROAD_ROI_TOP: float = 0.5 
    IGNORE_BOTTOM_RATIO: float = 0.1 
    
    # 導航邏輯
    PATH_DEVIATION_TH: int = 3  
    PATH_CENTER_RATIO: float = 0.15 
    ZEBRA_MIN_AREA: float = 1.5
    
    # 語音與時間
    STARTUP_GRACE_PERIOD: float = 5.0
    PATH_LOST_TIMEOUT: float = 3.0   
    ZEBRA_LOCK_TIMEOUT: float = 2.0 
    REMIND_INTERVAL: float = 5.0 
    TIMEOUT_LOCK: float = 3.0    
    HEARTBEAT_INTERVAL: float = 5.0
    CODE_GREEN: int = 65      
    CODE_RED: int = 67        
    CODE_COUNTDOWN: int = 66  

# ==========================================
# 2. 自動登入模組 (Auto Login)
# ==========================================
def get_auth_token(config: Config):
    """
    呼叫後端 API 進行登入，取得 JWT Token
    """
    print(f"🔑 正在嘗試登入: {config.LOGIN_URL} ...")
    
    # 根據範例檔，Payload 欄位確認為 Email / Password
    payload = {
        "Email": config.USER_EMAIL,
        "Password": config.USER_PASSWORD
    }
    
    try:
        response = requests.post(config.LOGIN_URL, json=payload, verify=True, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # 根據範例檔，Token 放在 "token" 欄位
            token = data.get("token")
            if token:
                print(f"✅ 登入成功！Token (前10碼): {token[:10]}...")
                return token
            else:
                print(f"❌ 登入成功但找不到 token 欄位。回應: {data}")
        else:
            print(f"❌ 登入失敗 (Status {response.status_code})：{response.text}")
            
    except Exception as e:
        print(f"❌ 連線錯誤 (Login): {e}")
    
    return None

# ==========================================
# 3. 語音處理模組 (TTS)
# ==========================================
class TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.engine = None
        self.last_spoken_time = {} 
        self.last_spoken_msg = {} 

    def run(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 140) 
            self.engine.setProperty('volume', 1.0)
            self.engine.startLoop(False)
        except: pass

        while not self.stop_event.is_set():
            try:
                self.engine.iterate() 
                try:
                    msg = self.queue.get_nowait()
                    self.engine.say(msg)
                except queue.Empty: pass
                time.sleep(0.05)
            except: time.sleep(0.1)
        try: self.engine.endLoop()
        except: pass

    def speak(self, msg: str, key: str = "general", interval: float = 0, force: bool = False, clear_queue: bool = False):
        now = time.time()
        last_time = self.last_spoken_time.get(key, 0)
        last_msg = self.last_spoken_msg.get(key, "")
        if force or (msg != last_msg) or (now - last_time > interval):
            if clear_queue:
                with self.queue.mutex: self.queue.queue.clear()
            self.queue.put(msg)
            self.last_spoken_time[key] = now
            self.last_spoken_msg[key] = msg

    def stop(self):
        self.stop_event.set()

# ==========================================
# 4. 非同步影像偵測模組
# ==========================================
class AsyncTrafficDetector(threading.Thread):
    def __init__(self, config: Config):
        super().__init__(daemon=True)
        self.cfg = config
        self.input_queue = queue.Queue(maxsize=1) 
        self.result_lock = threading.Lock()
        self.running = True
        
        self.latest_results = {
            "traffic": {"green": False, "red": False, "digit": None}, 
            "green_path": {"percentage": 0, "cx": 0, "contours": []},
            "zebra": {"percentage": 0, "cx": 0, "masks_list": [], "box": None},
        }
        self.model_traffic = None; self.model_zebra = None; self.cnn = None

    def update_input(self, frame, cnn_enabled):
        if self.input_queue.full():
            try: self.input_queue.get_nowait()
            except queue.Empty: pass
        self.input_queue.put((frame.copy(), cnn_enabled))

    def get_results(self):
        with self.result_lock: return copy.deepcopy(self.latest_results)

    def run(self):
        print("[AsyncDetector] 正在載入 AI 模型...")
        self.model_traffic = YOLO(self.cfg.MODEL_TRAFFIC)
        self.model_zebra = YOLO(self.cfg.MODEL_ZEBRA)
        try:
            if load_model: self.cnn = load_model(self.cfg.MODEL_CNN)
        except: self.cnn = None
        print("[AsyncDetector] 模型載入完成")

        task_counter = 0
        while self.running:
            try:
                frame, cnn_enabled = self.input_queue.get(timeout=0.1)
                traffic_res = self._detect_traffic(frame, cnn_enabled)
                with self.result_lock:
                    current_green = self.latest_results["green_path"]
                    current_zebra = self.latest_results["zebra"]

                if task_counter % 2 == 0:
                    zebra_res = self._detect_zebra(frame)
                    green_res = current_green 
                else:
                    zebra_res = current_zebra 
                    green_res = self._detect_green_path(frame)

                task_counter += 1
                with self.result_lock:
                    self.latest_results["traffic"] = traffic_res
                    self.latest_results["zebra"] = zebra_res
                    self.latest_results["green_path"] = green_res
                time.sleep(0.005)
            except queue.Empty: continue
            except Exception as e: traceback.print_exc()

    def _detect_traffic(self, frame, cnn_enabled):
        res = {"green": False, "red": False, "digit": None}
        try:
            results = self.model_traffic(frame, imgsz=self.cfg.IMGSZ_TRAFFIC, conf=self.cfg.CONF_TRAFFIC, verbose=False)
            for r in results:
                for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    c = int(cls); safe_box = [int(x) for x in box]
                    if c == 1: res["green"] = True; res["green_box"] = safe_box
                    elif c == 2: res["red"] = True; res["red_box"] = safe_box
                    elif c == 0: 
                        res["cnt_box"] = safe_box
                        if cnn_enabled and self.cnn:
                            x1,y1,x2,y2 = safe_box
                            h, w = frame.shape[:2]
                            x1, x2 = max(0, x1), min(w, x2)
                            y1, y2 = max(0, y1), min(h, y2)
                            if x2 > x1 and y2 > y1:
                                crop = frame[y1:y2, x1:x2]
                                digit = self._predict_digit(crop)
                                if digit: res["digit"] = digit
        except: pass
        return res

    def _predict_digit(self, img):
        if img.size == 0 or self.cnn is None: return None
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            resized = cv2.resize(th, (28, 28)) / 255.0
            input_tensor = resized.reshape(1, 28, 28, 1)
            pred_tensor = self.cnn(input_tensor, training=False)
            pred = pred_tensor.numpy()
            if np.max(pred) > self.cfg.CONF_CNN: return np.argmax(pred)
        except: pass
        return None

    def _detect_green_path(self, frame):
        res = {"percentage": 0, "cx": frame.shape[1]//2, "contours": []}
        h, w = frame.shape[:2]
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_green = cv2.inRange(hsv, self.cfg.LOWER_GREEN, self.cfg.UPPER_GREEN)
            mask_green[0:int(h * self.cfg.ROAD_ROI_TOP), :] = 0
            if self.cfg.IGNORE_BOTTOM_RATIO > 0:
                mask_green[int(h * (1 - self.cfg.IGNORE_BOTTOM_RATIO)):, :] = 0
            contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(max_cnt)
                res["percentage"] = (area / (w * h)) * 100
                M = cv2.moments(max_cnt)
                if M["m00"] != 0: res["cx"] = int(M["m10"] / M["m00"])
                res["contours"] = [c for c in contours if cv2.contourArea(c) > 500]
        except: pass
        return res

    def _detect_zebra(self, frame):
        res = {"percentage": 0, "cx": frame.shape[1]//2, "masks_list": [], "box": None}
        h, w = frame.shape[:2]
        try:
            ai_input = frame.copy()
            if self.cfg.IGNORE_BOTTOM_RATIO > 0:
                ai_input[int(h * (1 - self.cfg.IGNORE_BOTTOM_RATIO)):, :] = 0
            results = self.model_zebra(ai_input, imgsz=self.cfg.IMGSZ_ZEBRA, conf=self.cfg.CONF_ZEBRA, verbose=False, retina_masks=True)
            r = results[0]
            if r.masks is not None:
                all_masks_points = r.masks.xy
                total_area = 0; weighted_cx = 0
                for points in all_masks_points:
                    if len(points) > 0:
                        ys = points[:, 1]
                        if np.min(ys) < h * self.cfg.ROAD_ROI_TOP: continue 
                        pts = points.astype(np.int32)
                        area = cv2.contourArea(pts)
                        if area < 800: continue
                        res["masks_list"].append(pts)
                        total_area += area
                        M = cv2.moments(pts)
                        if M["m00"] != 0: cx = int(M["m10"] / M["m00"]); weighted_cx += cx * area
                if total_area > 0:
                    res["percentage"] = (total_area / (w * h)) * 100
                    res["cx"] = int(weighted_cx / total_area)
            if not res["masks_list"] and r.boxes is not None:
                max_area = 0; best_box = None
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    if y1 < h * self.cfg.ROAD_ROI_TOP: continue
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area: max_area = area; best_box = (x1, y1, x2, y2)
                if best_box:
                    res["box"] = best_box
                    res["percentage"] = (max_area / (w * h)) * 100
                    res["cx"] = (best_box[0] + best_box[2]) // 2
        except: pass
        return res

# ==========================================
# 5. 狀態管理模組
# ==========================================
class TrafficStateManager:
    def __init__(self, config: Config, tts: TTSWorker):
        self.cfg = config; self.tts = tts
        self.lights = {"green": {"state": False, "last_seen": 0, "box": None}, "red": {"state": False, "last_seen": 0, "box": None}}
        self.countdown = {"active": False, "value": 0, "last_tick": 0, "last_digit": None, "box": None}
        self.cnn_enabled = True; self.path_state = "NORMAL"; self.last_path_state = "NORMAL"
        self.last_path_seen_time = time.time(); self.system_start_time = time.time()
        self.last_heartbeat = time.time(); self.prev_light_tts = None
        self.guidance_source = "NONE"; self.smoothed_cx = None; self.last_zebra_time = 0 

    def update(self, det_traffic, det_green, det_zebra, frame_width):
        now = time.time()
        if now - self.last_heartbeat > self.cfg.HEARTBEAT_INTERVAL:
            self.tts.speak("滴", key="heartbeat", interval=0, force=True)
            self.last_heartbeat = now
            
        self._update_lights(det_traffic, now)
        self._handle_countdown(det_traffic.get('digit'))
        
        current_light = "red" if self.lights["red"]["state"] else "green" if self.lights["green"]["state"] else None
        if current_light != "red": self._handle_path_guidance(det_green, det_zebra, frame_width, now)
        self._trigger_tts(current_light)

    def _update_lights(self, detections, now):
        current_boxes = {"green": detections.get("green_box"), "red": detections.get("red_box")}
        self.countdown["box"] = detections.get("cnt_box")
        for key in ["green", "red"]:
            detected = detections.get(key, False)
            if detected:
                if not self.lights[key]["state"]:
                    self.lights[key]["state"] = True
                    print(f"SEND SIGNAL: {self.cfg.CODE_GREEN if key=='green' else self.cfg.CODE_RED}")
                self.lights[key]["last_seen"] = now
                self.lights[key]["box"] = current_boxes[key]
            else:
                if self.lights[key]["state"] and (now - self.lights[key]["last_seen"] > self.cfg.TIMEOUT_LOCK):
                    self.lights[key]["state"] = False
                    self.lights[key]["box"] = None
                    if self.prev_light_tts == key: self.prev_light_tts = None

    def _handle_countdown(self, digit):
        if self.countdown["active"]:
            if time.time() - self.countdown["last_tick"] >= 1.0:
                self.countdown["value"] -= 1
                self.countdown["last_tick"] = time.time()
                if self.countdown["value"] <= 0: self.countdown["active"] = False; self.cnn_enabled = True
        if digit is not None and self.countdown.get("last_digit") == 11 and digit == 10:
            self.countdown.update({"active": True, "value": 10, "last_tick": time.time()})
            self.cnn_enabled = False 
            print(f"SEND SIGNAL: {self.cfg.CODE_COUNTDOWN}")
        self.countdown["last_digit"] = digit

    def _handle_path_guidance(self, green_data, zebra_data, width, now):
        zebra_pct = zebra_data.get("percentage", 0); green_pct = green_data.get("percentage", 0)
        center_x = width // 2; new_guidance_source = "NONE"

        if zebra_pct >= self.cfg.ZEBRA_MIN_AREA: 
            new_guidance_source = "ZEBRA"; raw_target_cx = zebra_data.get("cx", center_x); self.last_zebra_time = now 
        elif (now - self.last_zebra_time) < self.cfg.ZEBRA_LOCK_TIMEOUT:
            new_guidance_source = "WAITING_ZEBRA"; raw_target_cx = self.smoothed_cx if self.smoothed_cx else center_x
        elif green_pct >= self.cfg.PATH_DEVIATION_TH:
            new_guidance_source = "GREEN"; raw_target_cx = green_data.get("cx", center_x)
        else:
            new_guidance_source = "NONE"; raw_target_cx = center_x

        self.guidance_source = new_guidance_source
        if new_guidance_source in ["ZEBRA", "GREEN"]:
            if self.smoothed_cx is None: self.smoothed_cx = raw_target_cx
            else: self.smoothed_cx = int(self.smoothed_cx * 0.4 + raw_target_cx * 0.6)
            target_cx = self.smoothed_cx
        elif new_guidance_source == "WAITING_ZEBRA":
             target_cx = self.smoothed_cx if self.smoothed_cx else center_x
        else: target_cx = center_x

        limit_pixel = width * self.cfg.PATH_CENTER_RATIO
        current_state = "NORMAL"; msg = ""

        if new_guidance_source in ["ZEBRA", "GREEN"]:
            self.last_path_seen_time = now
            if target_cx < center_x - limit_pixel: current_state = "SHIFT_LEFT"; msg = "請向左修正"
            elif target_cx > center_x + limit_pixel: current_state = "SHIFT_RIGHT"; msg = "請向右修正"
        else:
            if now - self.last_path_seen_time > self.cfg.PATH_LOST_TIMEOUT: current_state = "NO_SIGNAL"
            elif now - self.system_start_time < self.cfg.STARTUP_GRACE_PERIOD: current_state = "SEARCHING" 
            else:
                current_state = "OUT_OF_PATH"
                if new_guidance_source != "WAITING_ZEBRA":
                    if self.last_path_state == "NORMAL": msg = ""
                    elif self.last_path_state in ["SHIFT_LEFT", "SHIFT_RIGHT"]: msg = "警告，偏離路徑"

        self.path_state = current_state 
        state_changed = (current_state != self.last_path_state)
        if msg: self.tts.speak(msg, key="path_guidance", interval=self.cfg.REMIND_INTERVAL, force=state_changed, clear_queue=True)
        self.last_path_state = current_state

    def _trigger_tts(self, current_light):
        if current_light and current_light != self.prev_light_tts:
            if not self.countdown["active"] or self.countdown["value"] > 5:
                msg = "紅燈請停下" if current_light == "red" else "綠燈可以走"
                self.tts.speak(msg, key="light", force=True, clear_queue=True)
                self.prev_light_tts = current_light
        if self.countdown["active"] and self.countdown["value"] == 10:
            self.tts.speak("剩餘10秒", key="cnt_10", force=True, clear_queue=True)

    def get_draw_info(self):
        boxes = []
        for k, v in self.lights.items():
            if v["state"] and v["box"] is not None:
                boxes.append((k.capitalize(), v["box"], (0,255,0) if k=='green' else (0,0,255)))
        if self.countdown["box"] is not None:
            boxes.append(("CNT", self.countdown["box"], (0,255,255)))
        return boxes

# ==========================================
# 6. 影像接收模組 (Input)
# ==========================================
class CameraReceiver(threading.Thread):
    def __init__(self, source, frame_queue):
        super().__init__(daemon=True)
        self.source = source
        self.frame_queue = frame_queue
        self.running = True

    def run(self):
        print(f"📷 連線攝影機: {self.source}")
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(f"❌ 無法開啟攝影機 {self.source}")
            return

        print("✅ 影像來源已連線")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 讀取失敗，嘗試重連...")
                cap.release(); time.sleep(2); cap = cv2.VideoCapture(self.source); continue

            if self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except: pass
            self.frame_queue.put(frame)
        cap.release()

# ==========================================
# 7. SignalR 發送器 (修正 IndentationError 版)
# ==========================================
class SignalRSender(threading.Thread):
    def __init__(self, config: Config, send_queue, token):
        super().__init__(daemon=True)
        self.cfg = config
        self.send_queue = send_queue
        self.token = token
        self.running = True
        self.hub_connection = None
        self.is_connected = False 

    def build_connection(self):
        """建立一個全新的 SignalR 連線物件"""
        print(f"🔄 正在建立新連線: {self.cfg.HUB_URL}")
        token_factory = lambda: self.token
        
        # ★★★ 修正點：使用小括號 () 包覆，避免換行錯誤 ★★★
        hub = (HubConnectionBuilder()
            .with_url(self.cfg.HUB_URL, options={
                "access_token_factory": token_factory,
                "headers": {"User-Agent": "SmartCane-PC-Client"}
            })
            .configure_logging(logging_level=40) # 修正：註解現在安全了
            .build())
            
        hub.on_open(self.on_open)
        hub.on_close(self.on_close)
        hub.on_error(self.on_error)
        return hub

    def on_open(self):
        print("✅ SignalR 已連線! 通道暢通。")
        self.is_connected = True

    def on_close(self):
        print("❌ SignalR 已斷線。")
        self.is_connected = False

    def on_error(self, data):
        print(f"⚠️ SignalR 發生錯誤: {data}")
        self.is_connected = False

    def run(self):
        print(f"☁️ 啟動 SignalR 發送執行緒...")
        
        while self.running:
            # 1. 確保連線存在
            if self.hub_connection is None:
                try:
                    self.hub_connection = self.build_connection()
                    self.hub_connection.start()
                    # 給予一點時間進行握手 (Handshake)
                    for _ in range(20): # 等待最多 2 秒
                        if self.is_connected: break
                        time.sleep(0.1)
                except Exception as e:
                    print(f"🔥 連線建立失敗: {e}")
                    self.is_connected = False
                    time.sleep(3) # 失敗後休息 3 秒再試
                    continue

            # 2. 如果連線成功，開始傳送
            if self.is_connected:
                if not self.send_queue.empty():
                    frame = self.send_queue.get()
                    
                    # 降低畫質以減輕頻寬壓力
                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    
                    try:
                        self.hub_connection.send(
                            self.cfg.HUB_METHOD_NAME, 
                            [self.cfg.USER_EMAIL, jpg_as_text]
                        )
                    except Exception as e:
                        print(f"⚠️ 傳送失敗: {e}")
                        self.is_connected = False
                        try: self.hub_connection.stop() 
                        except: pass
                        self.hub_connection = None 

                    time.sleep(0.06) 
                else:
                    time.sleep(0.01)
            
            # 3. 如果斷線了
            else:
                print("⏳ 連線中斷，正在重置連線...")
                try: 
                    if self.hub_connection: self.hub_connection.stop()
                except: pass
                self.hub_connection = None 
                time.sleep(3) 

        if self.hub_connection:
            self.hub_connection.stop()

# ==========================================
# 8. 主程式入口
# ==========================================
def main():
    def emergency_speak(text):
        try:
            eng = pyttsx3.init(); eng.setProperty('rate', 150); eng.say(text); eng.runAndWait()
        except: pass

    try:
        cfg = Config()
        
        # 1. 執行自動登入 (取得 Token)
        token = get_auth_token(cfg)
        if not token:
            print("❌ 無法取得 Token，將無法傳送影像到網頁！")
            # 這裡不強制退出，讓本地端還是可以跑
        
        tts = TTSWorker(); tts.start()
        
        # AI 偵測器
        detector = AsyncTrafficDetector(cfg)
        detector.start()
        state_mgr = TrafficStateManager(cfg, tts)
        
        # 2. 啟動輸入
        input_queue = queue.Queue(maxsize=1)
        camera_receiver = CameraReceiver(cfg.CAMERA_SOURCE, input_queue)
        camera_receiver.start()
        
        # 3. 啟動 SignalR 輸出 (如果有 Token)
        output_queue = queue.Queue(maxsize=1)
        if cfg.ENABLE_STREAMING and token:
            streamer = SignalRSender(cfg, output_queue, token)
            streamer.start()

        print(f"系統啟動中... 視窗按 'q' 可離開")
        
        cv2.namedWindow("Smart Guide", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Guide", 1024, 768)
        
        idx = 0
        while True:
            try:
                frame = input_queue.get(timeout=0.1)
            except queue.Empty:
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'): break
                continue
            
            if cfg.NEED_ROTATION:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            if idx % 300 == 0: gc.collect()
            idx += 1

            # AI 處理
            detector.update_input(frame, state_mgr.cnn_enabled)
            ai_results = detector.get_results()
            state_mgr.update(ai_results["traffic"], ai_results["green_path"], ai_results["zebra"], frame.shape[1])
            
            # 畫面繪製
            draw_frame = frame.copy()
            try:
                if ai_results["green_path"]["contours"]:
                    cv2.drawContours(draw_frame, ai_results["green_path"]["contours"], -1, (0, 255, 0), 2)
                if ai_results["zebra"].get("masks_list"):
                    overlay = draw_frame.copy()
                    for mask_pts in ai_results["zebra"]["masks_list"]:
                        cv2.fillPoly(overlay, [mask_pts], (0, 165, 255))
                    cv2.addWeighted(overlay, 0.4, draw_frame, 0.6, 0, draw_frame)
                elif ai_results["zebra"].get("box") is not None:
                    bx1, by1, bx2, by2 = ai_results["zebra"]["box"]
                    cv2.rectangle(draw_frame, (bx1, by1), (bx2, by2), (255, 0, 255), 3)

                limit_pixel = int(draw_frame.shape[1] * cfg.PATH_CENTER_RATIO)
                center_x = draw_frame.shape[1] // 2
                cv2.line(draw_frame, (center_x - limit_pixel, 0), (center_x - limit_pixel, draw_frame.shape[0]), (0, 100, 255), 1)
                cv2.line(draw_frame, (center_x + limit_pixel, 0), (center_x + limit_pixel, draw_frame.shape[0]), (0, 100, 255), 1)

                if state_mgr.smoothed_cx is not None and state_mgr.guidance_source != "NONE":
                    cx = state_mgr.smoothed_cx
                    h, w = draw_frame.shape[:2]
                    screen_center_x = w // 2
                    guide_color = (0, 165, 255) if state_mgr.guidance_source == "ZEBRA" else (0, 255, 0)
                    cv2.line(draw_frame, (screen_center_x, h), (cx, h//2), guide_color, 4)
                    cv2.circle(draw_frame, (cx, h//2), 15, guide_color, -1)
                
                roi_y = int(draw_frame.shape[0] * cfg.ROAD_ROI_TOP)
                cv2.line(draw_frame, (0, roi_y), (draw_frame.shape[1], roi_y), (100, 100, 100), 1)

                for label, box, color in state_mgr.get_draw_info():
                    x1,y1,x2,y2 = map(int, box) 
                    x1=max(0,x1); y1=max(0,y1); x2=min(draw_frame.shape[1],x2); y2=min(draw_frame.shape[0],y2)
                    cv2.rectangle(draw_frame, (x1,y1), (x2,y2), color, 3)
                    cv2.putText(draw_frame, label, (x1,y1-10), 0, 0.7, color, 2)
                    
                if state_mgr.countdown["active"]:
                    cv2.putText(draw_frame, f"CNT: {state_mgr.countdown['value']}", (30,80), 0, 2, (0,255,255), 3)
                
                status_text = f"Path: {state_mgr.path_state}"
                if state_mgr.guidance_source != "NONE":
                    status_text += f" [{state_mgr.guidance_source}]"
                cv2.putText(draw_frame, status_text, (30, 40), 0, 0.8, (255,255,0), 2)
            except: pass

            cv2.imshow("Smart Guide", draw_frame)
            
            # ★★★ 傳送給 SignalR ★★★
            if cfg.ENABLE_STREAMING and token and not output_queue.full():
                small_frame = cv2.resize(draw_frame, (640, 480)) 
                output_queue.put(small_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                os._exit(0) 

        tts.stop()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"CRASHED: {e}")
        emergency_speak("系統錯誤") 
        os._exit(1)

if __name__ == "__main__":
    main()