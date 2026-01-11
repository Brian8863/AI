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
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

# 設定環境變數，隱藏 TensorFlow 煩人的初始化 log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from ultralytics import YOLO
try:
    from tensorflow.keras.models import load_model 
    import tensorflow as tf
except ImportError:
    print("警告: 找不到 TensorFlow，將停用 CNN 數字辨識功能。")
    load_model = None

import pyttsx3
import websocket

# ==========================================
# 1. 系統參數配置 (Config) - 參數說明區
# ==========================================
@dataclass
class Config:
    # -----------------------------------------------------------
    # [1] 連線與攝影機設定
    # -----------------------------------------------------------
    # 攝影機來源：
    #   - 填入數字 (如 0, 1) 代表 USB 攝影機編號
    #   - 填入網址 (如 "https://..." 或 "http://...") 代表 WebSocket 或 IP Cam 串流
    CAMERA_SOURCE: any = 1
    
    # 畫面旋轉：若鏡頭安裝角度是倒的或轉90度，設為 True 可修正
    NEED_ROTATION: bool = False 
    
    # AI 模型路徑設定
    MODEL_TRAFFIC: str = "Model/traffic&count.pt" # YOLO 紅綠燈與倒數計時框模型
    MODEL_ZEBRA: str = "Model/zebra_v3.pt"        # YOLO 斑馬線分割模型    
    MODEL_CNN: str = "Model/cnn_digit_model_new.h5" # CNN 數字辨識模型 (讀取倒數秒數)
    
    # -----------------------------------------------------------
    # [2] 解析度與效能設定
    # -----------------------------------------------------------
    # AI 推論時的圖片大小 (像素)。
    #   - 640: 標準大小，速度快 (FPS高)，適合即時導航。
    #   - 960/1280: 較清楚但速度慢，易造成延遲。建議維持 640。
    IMGSZ_TRAFFIC: int = 640    
    IMGSZ_ZEBRA: int = 640      
    
    # 影像緩衝區大小。
    #   - 設為 1 (關鍵)：只保留最新的一張圖，舊的直接丟棄。
    #   這能確保 AI 算出來的永遠是「當下」的畫面，不會有 1~2 秒的延遲。
    QUEUE_MAX: int = 1 
    
    # -----------------------------------------------------------
    # [3] 信心門檻 (Confidence Threshold)
    # -----------------------------------------------------------
    # AI 有多少把握才算是「偵測到」？(範圍 0.0 ~ 1.0)
    CONF_TRAFFIC: float = 0.8  # 紅綠燈要求高準確度，避免誤判紅燈變綠燈  
    CONF_ZEBRA: float = 0.5    # 斑馬線稍微低一點沒關係，能抓到比較重要
    CONF_CNN: float = 0.7      # 數字辨識的信心門檻  
    
    # -----------------------------------------------------------
    # [4] 綠色人行道與斑馬線設定
    # -----------------------------------------------------------
    # HSV 色彩空間中「綠色步道」的顏色範圍 (Lower=下限, Upper=上限)
    # 若現場光線變化大導致抓不到綠色步道，可調整這裡
    LOWER_GREEN: np.ndarray = field(default_factory=lambda: np.array([75, 40, 40]))   
    UPPER_GREEN: np.ndarray = field(default_factory=lambda: np.array([92, 255, 255])) 
    
    # 感興趣區域 (ROI) 設定：
    #   - 0.5 代表只看畫面「下半部 50%」，忽略上半部 (天空、建築)。
    #   - 防止誤判遠處的綠色招牌或紅綠燈。
    ROAD_ROI_TOP: float = 0.5 
    
    # 忽略畫面最底部的比例 (例如 0.1 = 底部 10%)
    #   - 用來避開拍攝者自己的腳或盲杖，以免被誤判成障礙物或斑馬線。
    IGNORE_BOTTOM_RATIO: float = 0.1 
    
    # -----------------------------------------------------------
    # [5] 導航邏輯 (靈敏度設定)
    # -----------------------------------------------------------
    # 只有當綠色步道或斑馬線佔畫面比例超過此值 (%)，才開始進行導航
    #   - 避免雜訊導致系統亂引導
    PATH_DEVIATION_TH: int = 3  
    
    # ★★★ 左右偏離警告的靈敏度 (關鍵) ★★★
    #   - 定義什麼叫做「路徑中心」。
    #   - 0.15 代表：畫面中心左右各 15% 的範圍是「安全區」。
    #   - 超過這個範圍 (使用者偏離中心 15%)，系統就會警告「請向左/右修正」。
    #   - 數值越小 = 越靈敏 (稍微偏一點就叫)。
    #   - 數值越大 = 越寬容 (快走出路了才叫)。
    PATH_CENTER_RATIO: float = 0.15 
    
    # 斑馬線最小面積佔比 (%)，大於此值才認為前方有斑馬線
    ZEBRA_MIN_AREA: float = 1.5
    
    # -----------------------------------------------------------
    # [6] 語音與時間控制 (Timing)
    # -----------------------------------------------------------
    # 系統剛啟動後的「免打擾時間」 (秒)。剛開機可能畫面不穩，先不講話。
    STARTUP_GRACE_PERIOD: float = 5.0
    
    # 路徑 (斑馬線/綠色步道) 消失多久後，才判定為「迷失/無訊號」 (秒)。
    PATH_LOST_TIMEOUT: float = 3.0   
    
    # 斑馬線消失後的「記憶時間」 (秒)。
    #   - 避免斑馬線因為光影斷斷續續，導致語音一直切換模式。
    ZEBRA_LOCK_TIMEOUT: float = 2.0 
    
    # 語音重複提醒的間隔 (秒)。
    #   - 例如「請向左修正」講完後，使用者若還沒走回來，隔 5 秒再講一次。
    REMIND_INTERVAL: float = 5.0 
    
    # 紅綠燈消失後的狀態保留時間 (秒)。
    #   - 避免燈號閃爍或被遮擋瞬間導致誤判燈號結束。
    TIMEOUT_LOCK: float = 3.0    
    
    # 系統「心跳聲 (滴)」的間隔 (秒)。讓使用者知道系統還活著。
    HEARTBEAT_INTERVAL: float = 5.0
    
    # 發送給硬體的訊號代碼 (若有連接 Arduino/ESP32震動馬達)
    CODE_GREEN: int = 65      # 'A'
    CODE_RED: int = 67        # 'C'
    CODE_COUNTDOWN: int = 66  # 'B'

# ==========================================
# 2. 語音處理模組 (TTS)
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
# 3. 非同步影像偵測模組
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
        
        self.model_traffic = None
        self.model_zebra = None
        self.cnn = None

    def update_input(self, frame, cnn_enabled):
        if self.input_queue.full():
            try: self.input_queue.get_nowait()
            except queue.Empty: pass
        self.input_queue.put((frame.copy(), cnn_enabled))

    def get_results(self):
        with self.result_lock:
            return copy.deepcopy(self.latest_results)

    def run(self):
        print("[AsyncDetector] 正在載入 AI 模型...")
        self.model_traffic = YOLO(self.cfg.MODEL_TRAFFIC)
        self.model_zebra = YOLO(self.cfg.MODEL_ZEBRA)
        try:
            if load_model: self.cnn = load_model(self.cfg.MODEL_CNN)
        except: self.cnn = None
        print("[AsyncDetector] 模型載入完成，開始推論循環")

        task_counter = 0

        while self.running:
            try:
                frame, cnn_enabled = self.input_queue.get(timeout=0.1)
                
                # 1. 交通號誌
                traffic_res = self._detect_traffic(frame, cnn_enabled)
                
                # 2. 斑馬線與綠色步道 (交錯運算以節省資源)
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

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncDetector Error] {e}")
                traceback.print_exc()

    def _detect_traffic(self, frame, cnn_enabled):
        res = {"green": False, "red": False, "digit": None}
        try:
            results = self.model_traffic(frame, imgsz=self.cfg.IMGSZ_TRAFFIC, conf=self.cfg.CONF_TRAFFIC, verbose=False)
            for r in results:
                for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    c = int(cls)
                    safe_box = [int(x) for x in box]
                    if c == 1: 
                        res["green"] = True
                        res["green_box"] = safe_box
                    elif c == 2: 
                        res["red"] = True
                        res["red_box"] = safe_box
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
                if M["m00"] != 0:
                    res["cx"] = int(M["m10"] / M["m00"])
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
            results = self.model_zebra(ai_input, imgsz=self.cfg.IMGSZ_ZEBRA, conf=self.cfg.CONF_ZEBRA, 
                                     verbose=False, retina_masks=True)
            r = results[0]
            if r.masks is not None:
                all_masks_points = r.masks.xy
                total_area = 0
                weighted_cx = 0
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
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            weighted_cx += cx * area
                if total_area > 0:
                    res["percentage"] = (total_area / (w * h)) * 100
                    res["cx"] = int(weighted_cx / total_area)
            if not res["masks_list"] and r.boxes is not None:
                max_area = 0
                best_box = None
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    if y1 < h * self.cfg.ROAD_ROI_TOP: continue
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2, y2)
                if best_box:
                    res["box"] = best_box
                    res["percentage"] = (max_area / (w * h)) * 100
                    res["cx"] = (best_box[0] + best_box[2]) // 2
        except: pass
        return res

# ==========================================
# 4. 狀態管理模組
# ==========================================
class TrafficStateManager:
    def __init__(self, config: Config, tts: TTSWorker):
        self.cfg = config
        self.tts = tts
        self.lights = {
            "green": {"state": False, "last_seen": 0, "box": None},
            "red":   {"state": False, "last_seen": 0, "box": None}
        }
        self.countdown = {"active": False, "value": 0, "last_tick": 0, "last_digit": None, "box": None}
        self.cnn_enabled = True
        self.path_state = "NORMAL"
        self.last_path_state = "NORMAL" 
        self.last_path_seen_time = time.time()
        self.system_start_time = time.time()
        self.last_heartbeat = time.time()
        self.prev_light_tts = None
        self.guidance_source = "NONE" 
        self.smoothed_cx = None
        self.last_zebra_time = 0 

    def update(self, det_traffic, det_green, det_zebra, frame_width):
        now = time.time()
        if now - self.last_heartbeat > self.cfg.HEARTBEAT_INTERVAL:
            self.tts.speak("滴", key="heartbeat", interval=0, force=True)
            self.last_heartbeat = now
            
        self._update_lights(det_traffic, now)
        self._handle_countdown(det_traffic.get('digit'))
        
        current_light = "red" if self.lights["red"]["state"] else \
                        "green" if self.lights["green"]["state"] else None
        
        if current_light != "red": 
            self._handle_path_guidance(det_green, det_zebra, frame_width, now)
        
        self._trigger_tts(current_light)

    def _update_lights(self, detections, now):
        current_boxes = {
            "green": detections.get("green_box"),
            "red": detections.get("red_box")
        }
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
                if self.countdown["value"] <= 0:
                    self.countdown["active"] = False
                    self.cnn_enabled = True
        if digit is not None and self.countdown.get("last_digit") == 11 and digit == 10:
            self.countdown.update({"active": True, "value": 10, "last_tick": time.time()})
            self.cnn_enabled = False 
            print(f"SEND SIGNAL: {self.cfg.CODE_COUNTDOWN}")
        self.countdown["last_digit"] = digit

    def _handle_path_guidance(self, green_data, zebra_data, width, now):
        zebra_pct = zebra_data.get("percentage", 0)
        green_pct = green_data.get("percentage", 0)
        center_x = width // 2
        new_guidance_source = "NONE"

        if zebra_pct >= self.cfg.ZEBRA_MIN_AREA: 
            new_guidance_source = "ZEBRA"
            raw_target_cx = zebra_data.get("cx", center_x)
            self.last_zebra_time = now 
        elif (now - self.last_zebra_time) < self.cfg.ZEBRA_LOCK_TIMEOUT:
            new_guidance_source = "WAITING_ZEBRA" 
            raw_target_cx = self.smoothed_cx if self.smoothed_cx else center_x
        elif green_pct >= self.cfg.PATH_DEVIATION_TH:
            new_guidance_source = "GREEN"
            raw_target_cx = green_data.get("cx", center_x)
        else:
            new_guidance_source = "NONE"
            raw_target_cx = center_x

        self.guidance_source = new_guidance_source
        if new_guidance_source in ["ZEBRA", "GREEN"]:
            if self.smoothed_cx is None: 
                self.smoothed_cx = raw_target_cx
            else: 
                # 平滑移動：讓準心不會抖動，但又能快速跟上新的位置
                self.smoothed_cx = int(self.smoothed_cx * 0.4 + raw_target_cx * 0.6)
                
            target_cx = self.smoothed_cx
        elif new_guidance_source == "WAITING_ZEBRA":
             target_cx = self.smoothed_cx if self.smoothed_cx else center_x
        else:
            target_cx = center_x

        limit_pixel = width * self.cfg.PATH_CENTER_RATIO
        current_state = "NORMAL"
        msg = ""

        if new_guidance_source in ["ZEBRA", "GREEN"]:
            self.last_path_seen_time = now
            if target_cx < center_x - limit_pixel:
                current_state = "SHIFT_LEFT"
                msg = "請向左修正"
            elif target_cx > center_x + limit_pixel:
                current_state = "SHIFT_RIGHT"
                msg = "請向右修正"
        else:
            if now - self.last_path_seen_time > self.cfg.PATH_LOST_TIMEOUT:
                current_state = "NO_SIGNAL"
            elif now - self.system_start_time < self.cfg.STARTUP_GRACE_PERIOD:
                current_state = "SEARCHING" 
            else:
                current_state = "OUT_OF_PATH"
                if new_guidance_source != "WAITING_ZEBRA":
                    if self.last_path_state == "NORMAL": msg = ""
                    elif self.last_path_state in ["SHIFT_LEFT", "SHIFT_RIGHT"]: msg = "警告，偏離路徑"

        self.path_state = current_state 
        state_changed = (current_state != self.last_path_state)
        if msg:
            self.tts.speak(msg, key="path_guidance", interval=self.cfg.REMIND_INTERVAL, force=state_changed, clear_queue=True)
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
# 5. 影像接收模組 (Local & WebSocket)
# ==========================================
class WebSocketReceiver(threading.Thread):
    def __init__(self, url, frame_queue):
        super().__init__(daemon=True)
        self.url = url.replace("https://", "wss://").replace("http://", "ws://")
        self.frame_queue = frame_queue
        self.ws = None

    def on_message(self, ws, message):
        try:
            if isinstance(message, str):
                if "base64," in message:
                    message = message.split("base64,")[1]
                img_data = base64.b64decode(message)
                np_arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif isinstance(message, bytes):
                np_arr = np.frombuffer(message, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                img = None

            if img is not None:
                if self.frame_queue.full():
                    try: self.frame_queue.get_nowait()
                    except: pass
                self.frame_queue.put(img)
        except: pass

    def on_error(self, ws, error):
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket 連線已關閉")

    def on_open(self, ws):
        print("✅ WebSocket 已連線到 Render Server")

    def run(self):
        while True:
            self.ws = websocket.WebSocketApp(self.url,
                                             on_open=self.on_open,
                                             on_message=self.on_message,
                                             on_error=self.on_error,
                                             on_close=self.on_close)
            self.ws.run_forever()
            time.sleep(2) 

class LocalCameraReceiver(threading.Thread):
    def __init__(self, source_id, frame_queue):
        super().__init__(daemon=True)
        self.source_id = source_id
        self.frame_queue = frame_queue
        self.running = True

    def run(self):
        print(f"📷 正在啟動本機攝影機 ID: {self.source_id} ...")
        cap = cv2.VideoCapture(self.source_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(f"❌ 無法開啟攝影機 {self.source_id}")
            return

        print("✅ 本機攝影機已啟動")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("❌ 讀取不到影像，嘗試重連...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(self.source_id)
                continue

            if self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except: pass
            self.frame_queue.put(frame)
            time.sleep(0.005) 
        cap.release()

# ==========================================
# 6. 主程式入口
# ==========================================
def main():
    def emergency_speak(text):
        try:
            eng = pyttsx3.init()
            eng.setProperty('rate', 150)
            eng.say(text)
            eng.runAndWait()
        except: pass

    try:
        cfg = Config()
        tts = TTSWorker(); tts.start()
        
        # 啟動非同步 AI 偵測器
        detector = AsyncTrafficDetector(cfg)
        detector.start()
        
        state_mgr = TrafficStateManager(cfg, tts)
        
        # 影像佇列 (Maxsize=1 確保低延遲)
        frame_q = queue.Queue(maxsize=1)
        
        # 判斷影像來源並啟動對應接收器
        receiver = None
        if isinstance(cfg.CAMERA_SOURCE, int):
            print(f"模式：本機攝影機 (ID: {cfg.CAMERA_SOURCE})")
            receiver = LocalCameraReceiver(cfg.CAMERA_SOURCE, frame_q)
        else:
            print(f"模式：遠端伺服器 ({cfg.CAMERA_SOURCE})")
            receiver = WebSocketReceiver(cfg.CAMERA_SOURCE, frame_q)
        receiver.start()

        print(f"系統啟動中... 等待影像訊號...")
        
        cv2.namedWindow("Smart Guide", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Guide", 1024, 768)
        
        idx = 0
        while True:
            try:
                frame = frame_q.get(timeout=0.01)
            except queue.Empty:
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'): break
                continue
            
            if cfg.NEED_ROTATION:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            if idx % 300 == 0: gc.collect()
            idx += 1

            # 1. 放入影像
            detector.update_input(frame, state_mgr.cnn_enabled)

            # 2. 取得結果
            ai_results = detector.get_results()

            # 3. 邏輯判斷
            state_mgr.update(ai_results["traffic"], ai_results["green_path"], ai_results["zebra"], frame.shape[1])
            
            # 4. 畫面繪製
            try:
                if ai_results["green_path"]["contours"]:
                    cv2.drawContours(frame, ai_results["green_path"]["contours"], -1, (0, 255, 0), 2)
                
                if ai_results["zebra"].get("masks_list"):
                    overlay = frame.copy()
                    for mask_pts in ai_results["zebra"]["masks_list"]:
                        cv2.fillPoly(overlay, [mask_pts], (0, 165, 255))
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                elif ai_results["zebra"].get("box") is not None:
                    bx1, by1, bx2, by2 = ai_results["zebra"]["box"]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 255), 3)
                
                # 畫出安全範圍線 (左右邊界)
                limit_pixel = int(frame.shape[1] * cfg.PATH_CENTER_RATIO)
                center_x = frame.shape[1] // 2
                cv2.line(frame, (center_x - limit_pixel, 0), (center_x - limit_pixel, frame.shape[0]), (0, 100, 255), 1)
                cv2.line(frame, (center_x + limit_pixel, 0), (center_x + limit_pixel, frame.shape[0]), (0, 100, 255), 1)

                if state_mgr.smoothed_cx is not None and state_mgr.guidance_source != "NONE":
                    cx = state_mgr.smoothed_cx
                    h, w = frame.shape[:2]
                    screen_center_x = w // 2
                    guide_color = (0, 165, 255) if state_mgr.guidance_source == "ZEBRA" else (0, 255, 0)
                    cv2.line(frame, (screen_center_x, h), (cx, h//2), guide_color, 4)
                    cv2.circle(frame, (cx, h//2), 15, guide_color, -1)
                
                roi_y = int(frame.shape[0] * cfg.ROAD_ROI_TOP)
                cv2.line(frame, (0, roi_y), (frame.shape[1], roi_y), (100, 100, 100), 1)

                for label, box, color in state_mgr.get_draw_info():
                    x1,y1,x2,y2 = map(int, box) 
                    x1=max(0,x1); y1=max(0,y1); x2=min(frame.shape[1],x2); y2=min(frame.shape[0],y2)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
                    cv2.putText(frame, label, (x1,y1-10), 0, 0.7, color, 2)
                    
                if state_mgr.countdown["active"]:
                    cv2.putText(frame, f"CNT: {state_mgr.countdown['value']}", (30,80), 0, 2, (0,255,255), 3)
                
                status_text = f"Path: {state_mgr.path_state}"
                if state_mgr.guidance_source != "NONE":
                    status_text += f" [{state_mgr.guidance_source}]"
                cv2.putText(frame, status_text, (30, 40), 0, 0.8, (255,255,0), 2)

                cv2.imshow("Smart Guide", frame)
            
            except Exception as draw_err:
                print(f"Drawing Error: {draw_err}") 
                traceback.print_exc()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                print("使用者手動退出。")
                os._exit(0) 

        tts.stop()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"CRASHED: {e}")
        emergency_speak("系統錯誤") 
        os._exit(1)

if __name__ == "__main__":
    main()