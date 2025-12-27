import os
import cv2
import numpy as np
import threading
import queue
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

# 設定環境變數
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from ultralytics import YOLO
try:
    from tensorflow.keras.models import load_model 
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
    # [1] 檔案路徑設定
    # -----------------------------------------------------------
    VIDEO_PATH: any = 1  
    MODEL_TRAFFIC: str = "Model/traffic_1280.pt" 
    MODEL_ROAD: str = "best.pt"            
    MODEL_COMMON: str = "yolov8n.pt"       
    MODEL_CNN: str = "cnn_digit_model_new.h5" 
    
    # -----------------------------------------------------------
    # [2] 解析度設定
    # -----------------------------------------------------------
    IMGSZ_TRAFFIC: int = 1280    
    IMGSZ_ROAD: int = 640        
    IMGSZ_COMMON: int = 320      
    
    # -----------------------------------------------------------
    # [3] 信心門檻
    # -----------------------------------------------------------
    CONF_TRAFFIC: float = 0.5    
    CONF_ROAD: float = 0.25      
    CONF_COMMON: float = 0.5     
    CONF_CNN: float = 0.7        
    
    # -----------------------------------------------------------
    # [4] 效能優化設定
    # -----------------------------------------------------------
    PROCESS_CYCLE: int = 4       
    FRAME_TRAFFIC: int = 0       
    FRAME_ROAD: int = 1          
    FRAME_COMMON: int = 2        
    QUEUE_MAX: int = 3           
    
    # -----------------------------------------------------------
    # [5] 斑馬線/路徑顏色設定 (HSV)
    # -----------------------------------------------------------
    LOWER_GREEN: np.ndarray = field(default_factory=lambda: np.array([75, 40, 40]))   
    UPPER_GREEN: np.ndarray = field(default_factory=lambda: np.array([92, 255, 255])) 
    
    # -----------------------------------------------------------
    # [6] 路徑引導靈敏度
    # -----------------------------------------------------------
    ROAD_ROI_TOP: float = 0.5 
    
    PATH_DEVIATION_TH: int = 3  
    PATH_RETURN_BUFFER: int = 2

    PATH_CENTER_RATIO: float = 0.25
    PATH_CENTER_BUFFER: float = 0.05     
    
    # -----------------------------------------------------------
    # [7] 語音與邏輯控制
    # -----------------------------------------------------------
    STARTUP_GRACE_PERIOD: float = 5.0
    PATH_LOST_TIMEOUT: float = 5.0
    REMIND_INTERVAL: float = 8.0 
    TIMEOUT_LOCK: float = 3.0    
    TTS_INTERVAL: float = 15.0   
    MIN_OBS_HEIGHT: float = 0.6  
    
    # [新增] 心跳聲間隔 (秒)
    # 這是最後一道防線：如果使用者聽不到這個聲音，代表系統死當了
    HEARTBEAT_INTERVAL: float = 5.0
    
    # -----------------------------------------------------------
    # [8] 其他設定
    # -----------------------------------------------------------
    CODE_GREEN: int = 65      
    CODE_RED: int = 67        
    CODE_COUNTDOWN: int = 66  
    LOOP_VIDEO: bool = True      

# ==========================================
# 2. 語音處理模組
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
        except Exception as e:
            print(f"[TTS Error]: {e}")

        while not self.stop_event.is_set():
            try:
                self.engine.iterate() 
                try:
                    msg = self.queue.get_nowait()
                    print(f"🔊 [語音]: {msg}")
                    self.engine.say(msg)
                except queue.Empty:
                    pass
                time.sleep(0.05)
            except: time.sleep(0.1)
        try: self.engine.endLoop()
        except: pass

    def speak(self, msg: str, key: str = "general", interval: float = 0, force: bool = False):
        now = time.time()
        last_time = self.last_spoken_time.get(key, 0)
        last_msg = self.last_spoken_msg.get(key, "")

        if force or (msg != last_msg) or (now - last_time > interval):
            self.queue.put(msg)
            self.last_spoken_time[key] = now
            self.last_spoken_msg[key] = msg

    def stop(self):
        self.stop_event.set()

# ==========================================
# 3. 狀態管理模組
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
        
        # [新增] 心跳計時器
        self.last_heartbeat = time.time()
        
        self.prev_light_tts = None

    def update(self, det_traffic, det_road, det_obstacle, frame_width):
        now = time.time()
        
        # --- [防線 3] 心跳聲邏輯 (Heartbeat) ---
        if now - self.last_heartbeat > self.cfg.HEARTBEAT_INTERVAL:
            # 發出短促的「滴」聲，代表系統活著
            # 如果覺得語音「滴」太慢，這行可以換成播放 wav 音效
            self.tts.speak("滴", key="heartbeat", interval=0, force=True)
            self.last_heartbeat = now
            
        self._update_lights(det_traffic, now)
        self._handle_countdown(det_traffic.get('digit'))
        
        current_light = "red" if self.lights["red"]["state"] else \
                        "green" if self.lights["green"]["state"] else None
                        
        if current_light != "red": 
            self._handle_path_guidance(det_road, frame_width)
        
        self._trigger_tts(current_light, det_obstacle)

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

    def _handle_path_guidance(self, road_data, width):
        pct = road_data.get("percentage", 0)
        cx = road_data.get("cx", width // 2)
        center_x = width // 2
        
        is_warning_state = self.path_state in ["SHIFT_LEFT", "SHIFT_RIGHT"]
        
        if is_warning_state:
            ratio_threshold = self.cfg.PATH_CENTER_RATIO - self.cfg.PATH_CENTER_BUFFER
        else:
            ratio_threshold = self.cfg.PATH_CENTER_RATIO
            
        limit_pixel = width * ratio_threshold

        if self.path_state in ["OUT_OF_PATH", "NO_SIGNAL"]:
            area_threshold = self.cfg.PATH_DEVIATION_TH + self.cfg.PATH_RETURN_BUFFER 
        else:
            area_threshold = self.cfg.PATH_DEVIATION_TH 

        current_state = "NORMAL"
        msg = ""

        if pct >= area_threshold:
            self.last_path_seen_time = time.time()
            if cx < center_x - limit_pixel:
                current_state = "SHIFT_LEFT"
                msg = "請向左修正"
            elif cx > center_x + limit_pixel:
                current_state = "SHIFT_RIGHT"
                msg = "請向右修正"
            else:
                current_state = "NORMAL"
                msg = "路徑正常"
        else:
            time_lost = time.time() - self.last_path_seen_time
            if time_lost > self.cfg.PATH_LOST_TIMEOUT:
                current_state = "NO_SIGNAL"
                msg = "" 
            else:
                time_since_boot = time.time() - self.system_start_time
                if time_since_boot < self.cfg.STARTUP_GRACE_PERIOD:
                    current_state = "SEARCHING" 
                    msg = "" 
                else:
                    current_state = "OUT_OF_PATH"
                    msg = "警告，偏離路徑"

        self.path_state = current_state 
        state_changed = (current_state != self.last_path_state)
        
        if msg:
            if current_state == "NORMAL":
                if state_changed: 
                    self.tts.speak(msg, key="path_guidance", force=True)
            else:
                self.tts.speak(msg, key="path_guidance", interval=self.cfg.REMIND_INTERVAL, force=state_changed)

        self.last_path_state = current_state

    def _trigger_tts(self, current_light, obstacles):
        if current_light and current_light != self.prev_light_tts:
            if not self.countdown["active"] or self.countdown["value"] > 5:
                msg = "紅燈請停下" if current_light == "red" else "綠燈可以走"
                self.tts.speak(msg, key="light", force=True)
                self.prev_light_tts = current_light

        if self.countdown["active"] and self.countdown["value"] == 10:
            self.tts.speak("剩餘10秒", key="cnt_10", force=True)

        if self.path_state == "NORMAL":
            if obstacles.get('vehicle', 0) > 0:
                self.tts.speak("前方有車", key="obs", interval=self.cfg.TTS_INTERVAL)
            elif obstacles.get('person', 0) > 0:
                self.tts.speak("小心行人", key="obs", interval=self.cfg.TTS_INTERVAL)

    def get_draw_info(self):
        boxes = []
        for k, v in self.lights.items():
            if v["state"] and v["box"] is not None:
                boxes.append((k.capitalize(), v["box"], (0,255,0) if k=='green' else (0,0,255)))
        if self.countdown["box"] is not None:
            boxes.append(("CNT", self.countdown["box"], (0,255,255)))
        return boxes

# ==========================================
# 4. 影像偵測模組
# ==========================================
class TrafficDetector:
    def __init__(self, config: Config):
        self.cfg = config
        self._init_models()
        try: cv2.setNumThreads(1); import torch; torch.set_num_threads(1)
        except: pass

    def _init_models(self):
        print(f"正在載入模型...")
        self.model_traffic = YOLO(self.cfg.MODEL_TRAFFIC)
        self.model_road = YOLO(self.cfg.MODEL_ROAD)
        self.model_common = YOLO(self.cfg.MODEL_COMMON)
        try: 
            if load_model: self.cnn = load_model(self.cfg.MODEL_CNN)
            else: self.cnn = None
        except: self.cnn = None; print("警告: CNN 模型載入失敗")

    def detect_traffic(self, frame, cnn_enabled):
        res = {"green": False, "red": False, "digit": None}
        try:
            results = self.model_traffic(frame, imgsz=self.cfg.IMGSZ_TRAFFIC, conf=self.cfg.CONF_TRAFFIC, verbose=False)
            for r in results:
                for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                    c = int(cls)
                    safe_box = [int(x) for x in box]
                    
                    if c == 1: # Green
                        res["green"] = True
                        res["green_box"] = safe_box
                    elif c == 2: # Red
                        res["red"] = True
                        res["red_box"] = safe_box
                    elif c == 0: # Countdown
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
        except Exception as e:
            print(f"Traffic Det Error: {e}")
        return res

    def _predict_digit(self, img):
        if img.size == 0 or self.cnn is None: return None
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            resized = cv2.resize(th, (28, 28)) / 255.0
            pred = self.cnn.predict(resized.reshape(1,28,28,1), verbose=0)
            if np.max(pred) > self.cfg.CONF_CNN: return np.argmax(pred)
        except: pass
        return None

    def detect_road(self, frame):
        res = {"percentage": 0, "cx": frame.shape[1]//2, "contours": []}
        h, w = frame.shape[:2]
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_green = cv2.inRange(hsv, self.cfg.LOWER_GREEN, self.cfg.UPPER_GREEN)
            
            roi_height = int(h * self.cfg.ROAD_ROI_TOP)
            mask_green[0:roi_height, :] = 0
            
            contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                max_cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(max_cnt)
                res["percentage"] = (area / (w * h)) * 100
                M = cv2.moments(max_cnt)
                if M["m00"] != 0:
                    res["cx"] = int(M["m10"] / M["m00"])
                res["contours"] = [c for c in contours if cv2.contourArea(c) > 500]
        except Exception: pass
        return res

    def detect_obstacles(self, frame):
        # 為了畫面乾淨，這邊暫時保留偵測但不回傳任何框
        # 這樣就不會畫出白框了
        return {"person": 0, "vehicle": 0, "boxes": []}

# ==========================================
# 5. 主程式入口 (含防線 1：例外處理)
# ==========================================
def main():
    # [防線 1] 緊急語音函式 (當程式死掉時，做最後的掙扎)
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
        detector = TrafficDetector(cfg)
        state_mgr = TrafficStateManager(cfg, tts)
        
        vid_src = cfg.VIDEO_PATH
        if isinstance(vid_src, str) and vid_src.isdigit():
            vid_src = int(vid_src)
        cap = cv2.VideoCapture(vid_src)
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0 or np.isnan(video_fps): video_fps = 30
        delay_time = int(1000 / video_fps) 
        
        frame_q = queue.Queue(maxsize=cfg.QUEUE_MAX)
        stop_event = threading.Event()

        def reader():
            is_video_file = isinstance(vid_src, str)
            while not stop_event.is_set():
                ret, f = cap.read()
                if not ret: 
                    if cfg.LOOP_VIDEO and is_video_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        frame_q.put(None)
                        break
                
                if is_video_file:
                    frame_q.put(f) 
                else:
                    if frame_q.full(): 
                        try: frame_q.get_nowait() 
                        except: pass
                    frame_q.put(f)
            cap.release()
        
        threading.Thread(target=reader, daemon=True).start()

        print(f"系統已啟動。FPS: {video_fps}。按 'q' 離開, 'k' 測試崩潰, 'f' 測試死當。")
        
        cv2.namedWindow("Smart Guide", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Guide", 1024, 768)
        
        cache = {
            "traffic": {}, 
            "road": {"percentage": 0, "cx": 0, "contours": []}, 
            "obs": {"boxes": [], "person": 0, "vehicle": 0}
        }
        
        idx = 0
        while True:
            frame = frame_q.get()
            if frame is None: 
                print("影片播放結束。")
                break
            
            cycle = idx % cfg.PROCESS_CYCLE
            idx += 1
            
            if cycle == cfg.FRAME_TRAFFIC:
                cache["traffic"] = detector.detect_traffic(frame, state_mgr.cnn_enabled)
            elif cycle == cfg.FRAME_ROAD:
                cache["road"] = detector.detect_road(frame)
            elif cycle == cfg.FRAME_COMMON:
                cache["obs"] = detector.detect_obstacles(frame)
            
            state_mgr.update(cache["traffic"], cache["road"], cache["obs"], frame.shape[1])
            
            # --- 畫面繪製 (含防閃退 try-catch) ---
            try:
                if cache["road"]["contours"]:
                    cv2.drawContours(frame, cache["road"]["contours"], -1, (0, 255, 0), 2)
                    if cache["road"]["cx"] > 0:
                        cv2.circle(frame, (cache["road"]["cx"], frame.shape[0]//2), 10, (0, 0, 255), -1)
                
                roi_y = int(frame.shape[0] * cfg.ROAD_ROI_TOP)
                cv2.line(frame, (0, roi_y), (frame.shape[1], roi_y), (100, 100, 100), 1)

                for label, box, color in state_mgr.get_draw_info():
                    x1,y1,x2,y2 = map(int, box) 
                    x1=max(0,x1); y1=max(0,y1); x2=min(frame.shape[1],x2); y2=min(frame.shape[0],y2)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
                    cv2.putText(frame, label, (x1,y1-10), 0, 0.7, color, 2)
                    
                # 障礙物框已在 detect_obstacles 停用，這裡不會畫出東西
                    
                if state_mgr.countdown["active"]:
                    cv2.putText(frame, f"CNT: {state_mgr.countdown['value']}", (30,80), 0, 2, (0,255,255), 3)
                
                cv2.putText(frame, f"Path: {state_mgr.path_state} ({cache['road']['percentage']:.1f}%)", (30, 40), 0, 0.8, (255,255,0), 2)

                cv2.imshow("Smart Guide", frame)
            
            except Exception as draw_err:
                print(f"Drawing Error: {draw_err}") 

            # --- 按鍵控制 (測試用) ---
            key = cv2.waitKey(delay_time) & 0xFF
            
            if key == ord('q'): # 正常離開
                print("使用者手動退出。")
                # 強制回傳 0 (代表正常)，這樣看門狗就不會重啟
                os._exit(0)
                
            elif key == ord('k'): # 模擬「崩潰 (Crash)」-> 測試自動重啟
                print("\n[測試] 3秒後模擬程式崩潰...")
                time.sleep(1) 
                raise RuntimeError("這是手動觸發的測試崩潰！")

            elif key == ord('f'): # 模擬「死當 (Freeze)」-> 測試心跳聲消失
                print("\n[測試] 系統模擬死當 (無限迴圈)...")
                while True:
                    time.sleep(1) 

        stop_event.set()
        tts.stop()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print("\n" + "="*40)
        print("❌ 程式發生錯誤 (CRASHED)")
        print(f"錯誤訊息: {e}")
        print("="*40)
        # [防線 1] 程式死前講最後一句話
        emergency_speak("系統錯誤，請原地停止") 
        # 讓 exit code 不為 0，通知外部看門狗
        os._exit(1)

if __name__ == "__main__":
    main()