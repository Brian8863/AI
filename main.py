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
    MODEL_TRAFFIC: str = "Model/traffic&count.pt" 
    
    # ★ 恢復斑馬線模型 (原本叫 MODEL_ROAD，現在正名為 MODEL_ZEBRA)
    MODEL_ZEBRA: str = "Model/zebra.pt"            
    
    MODEL_COMMON: str = "Model/yolov8n.pt"       
    MODEL_CNN: str = "Model/cnn_digit_model_new.h5" 
    
    # -----------------------------------------------------------
    # [2] 解析度設定 (維持高效能設定)
    # -----------------------------------------------------------
    IMGSZ_TRAFFIC: int = 640    # 紅綠燈 (降解析度求順暢)
    IMGSZ_ZEBRA: int = 640      # 斑馬線
    IMGSZ_COMMON: int = 320     # 障礙物
    
    # -----------------------------------------------------------
    # [3] 信心門檻
    # -----------------------------------------------------------
    CONF_TRAFFIC: float = 0.5    
    CONF_ZEBRA: float = 0.25     
    CONF_COMMON: float = 0.5     
    CONF_CNN: float = 0.7        
    
    # -----------------------------------------------------------
    # [4] 效能優化設定 (錯峰運算)
    # -----------------------------------------------------------
    # 維持 6 幀一個循環，讓 AI 運算分散開來
    PROCESS_CYCLE: int = 6       
    
    # [排程表]
    FRAME_TRAFFIC: int = 0      # AI: 紅綠燈
    FRAME_GREEN: int = 1        # HSV: 綠色人行道 (快)
    FRAME_COMMON: int = 2       # AI: 障礙物
    FRAME_ZEBRA: int = 3        # AI: 斑馬線 (重頭戲)
    FRAME_GREEN_2: int = 4      # HSV: 再補一次綠色 (確保流暢)
    
    QUEUE_MAX: int = 2 
    
    # -----------------------------------------------------------
    # [5] 綠色人行道設定 (HSV)
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

    def speak(self, msg: str, key: str = "general", interval: float = 0, force: bool = False, clear_queue: bool = False):
        now = time.time()
        last_time = self.last_spoken_time.get(key, 0)
        last_msg = self.last_spoken_msg.get(key, "")

        if force or (msg != last_msg) or (now - last_time > interval):
            if clear_queue:
                with self.queue.mutex:
                    self.queue.queue.clear()
            self.queue.put(msg)
            self.last_spoken_time[key] = now
            self.last_spoken_msg[key] = msg

    def stop(self):
        self.stop_event.set()

# ==========================================
# 3. 狀態管理模組 (支援雙路徑邏輯)
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
        self.guidance_source = "NONE" # 紀錄現在是跟著誰走

    def update(self, det_traffic, det_green, det_zebra, det_obstacle, frame_width):
        now = time.time()
        
        # 心跳聲 (維持有聲音)
        if now - self.last_heartbeat > self.cfg.HEARTBEAT_INTERVAL:
            self.tts.speak("滴", key="heartbeat", interval=0, force=True)
            self.last_heartbeat = now
            
        self._update_lights(det_traffic, now)
        self._handle_countdown(det_traffic.get('digit'))
        
        current_light = "red" if self.lights["red"]["state"] else \
                        "green" if self.lights["green"]["state"] else None
                        
        if current_light != "red": 
            # 傳入兩種路徑數據 (綠色 + 斑馬線)
            self._handle_path_guidance(det_green, det_zebra, frame_width)
        
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

    # ★★★ 雙路徑整合邏輯 ★★★
    def _handle_path_guidance(self, green_data, zebra_data, width):
        # 取得兩邊的信心度/面積
        zebra_pct = zebra_data.get("percentage", 0)
        green_pct = green_data.get("percentage", 0)
        
        target_cx = width // 2
        center_x = width // 2
        current_pct = 0
        
        # [優先級邏輯]
        # 如果有斑馬線，優先跟隨斑馬線 (因為路口引導更重要)
        if zebra_pct >= self.cfg.PATH_DEVIATION_TH:
            self.guidance_source = "ZEBRA"
            target_cx = zebra_data.get("cx", center_x)
            current_pct = zebra_pct
        elif green_pct >= self.cfg.PATH_DEVIATION_TH:
            self.guidance_source = "GREEN"
            target_cx = green_data.get("cx", center_x)
            current_pct = green_pct
        else:
            self.guidance_source = "NONE"
            current_pct = 0

        # --- 以下為通用引導邏輯 ---
        is_warning_state = self.path_state in ["SHIFT_LEFT", "SHIFT_RIGHT"]
        ratio_threshold = (self.cfg.PATH_CENTER_RATIO - self.cfg.PATH_CENTER_BUFFER) if is_warning_state else self.cfg.PATH_CENTER_RATIO
        limit_pixel = width * ratio_threshold
        
        area_threshold = (self.cfg.PATH_DEVIATION_TH + self.cfg.PATH_RETURN_BUFFER) if self.path_state in ["OUT_OF_PATH", "NO_SIGNAL"] else self.cfg.PATH_DEVIATION_TH 

        current_state = "NORMAL"
        msg = ""

        if current_pct >= area_threshold:
            self.last_path_seen_time = time.time()
            if target_cx < center_x - limit_pixel:
                current_state = "SHIFT_LEFT"
                msg = "請向左修正"
            elif target_cx > center_x + limit_pixel:
                current_state = "SHIFT_RIGHT"
                msg = "請向右修正"
            else:
                current_state = "NORMAL"
                msg = "" # 正常時閉嘴
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
            self.tts.speak(msg, key="path_guidance", interval=self.cfg.REMIND_INTERVAL, force=state_changed, clear_queue=True)

        self.last_path_state = current_state

    def _trigger_tts(self, current_light, obstacles):
        if current_light and current_light != self.prev_light_tts:
            if not self.countdown["active"] or self.countdown["value"] > 5:
                msg = "紅燈請停下" if current_light == "red" else "綠燈可以走"
                self.tts.speak(msg, key="light", force=True, clear_queue=True)
                self.prev_light_tts = current_light

        if self.countdown["active"] and self.countdown["value"] == 10:
            self.tts.speak("剩餘10秒", key="cnt_10", force=True, clear_queue=True)

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
# 4. 影像偵測模組 (整合 AI 與 HSV)
# ==========================================
class TrafficDetector:
    def __init__(self, config: Config):
        self.cfg = config
        self._init_models()
        # 不限制執行緒，讓效能全開

    def _init_models(self):
        print(f"正在載入模型...")
        self.model_traffic = YOLO(self.cfg.MODEL_TRAFFIC)
        
        # 載入斑馬線模型 (best.pt)
        self.model_zebra = YOLO(self.cfg.MODEL_ZEBRA) 
        
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

    # 功能 1: 綠色人行道 (HSV - 快速)
    def detect_green_path(self, frame):
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

    # 功能 2: 斑馬線 (AI - 準確但重)
    def detect_zebra(self, frame):
        res = {"percentage": 0, "cx": frame.shape[1]//2, "box": None}
        h, w = frame.shape[:2]
        try:
            # 呼叫 AI 模型 (best.pt)
            results = self.model_zebra(frame, imgsz=self.cfg.IMGSZ_ZEBRA, conf=self.cfg.CONF_ZEBRA, verbose=False)
            
            best_box = None
            max_area = 0
            
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2, y2)
            
            if best_box:
                x1, y1, x2, y2 = best_box
                res["percentage"] = (max_area / (w * h)) * 100
                res["cx"] = (x1 + x2) // 2
                res["box"] = best_box 

        except Exception as e:
            print(f"Zebra Det Error: {e}")
        return res

    def detect_obstacles(self, frame):
        return {"person": 0, "vehicle": 0, "boxes": []}

# ==========================================
# 5. 主程式入口
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
        
        # 快取空間 (含綠色與斑馬線)
        cache = {
            "traffic": {}, 
            "green_path": {"percentage": 0, "cx": 0, "contours": []},
            "zebra": {"percentage": 0, "cx": 0, "box": None},
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
            
            # [排程邏輯] 分散運算
            if cycle == cfg.FRAME_TRAFFIC:
                cache["traffic"] = detector.detect_traffic(frame, state_mgr.cnn_enabled)
                
            elif cycle == cfg.FRAME_GREEN or cycle == cfg.FRAME_GREEN_2:
                # 跑兩次綠色 (因為它快)
                cache["green_path"] = detector.detect_green_path(frame)
                
            elif cycle == cfg.FRAME_ZEBRA:
                # 跑一次斑馬線 (因為它重)
                cache["zebra"] = detector.detect_zebra(frame)
                
            elif cycle == cfg.FRAME_COMMON:
                cache["obs"] = detector.detect_obstacles(frame)
            
            state_mgr.update(cache["traffic"], cache["green_path"], cache["zebra"], cache["obs"], frame.shape[1])
            
            # --- 畫面繪製 ---
            try:
                # 1. 畫綠色人行道 (輪廓)
                if cache["green_path"]["contours"]:
                    cv2.drawContours(frame, cache["green_path"]["contours"], -1, (0, 255, 0), 2)
                
                # 2. 畫斑馬線 (粉紅框)
                if cache["zebra"]["box"] is not None:
                    zx1, zy1, zx2, zy2 = cache["zebra"]["box"]
                    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 255), 3) 
                    cv2.putText(frame, "Zebra", (zx1, zy1-10), 0, 0.7, (255, 0, 255), 2)
                
                # 顯示目前跟隨的目標點
                target_cx = 0
                if state_mgr.guidance_source == "ZEBRA":
                    target_cx = cache["zebra"]["cx"]
                    cv2.circle(frame, (target_cx, frame.shape[0]//2), 15, (255, 0, 255), -1) 
                elif state_mgr.guidance_source == "GREEN":
                    target_cx = cache["green_path"]["cx"]
                    cv2.circle(frame, (target_cx, frame.shape[0]//2), 10, (0, 255, 0), -1) 
                
                roi_y = int(frame.shape[0] * cfg.ROAD_ROI_TOP)
                cv2.line(frame, (0, roi_y), (frame.shape[1], roi_y), (100, 100, 100), 1)

                for label, box, color in state_mgr.get_draw_info():
                    x1,y1,x2,y2 = map(int, box) 
                    x1=max(0,x1); y1=max(0,y1); x2=min(frame.shape[1],x2); y2=min(frame.shape[0],y2)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
                    cv2.putText(frame, label, (x1,y1-10), 0, 0.7, color, 2)
                    
                if state_mgr.countdown["active"]:
                    cv2.putText(frame, f"CNT: {state_mgr.countdown['value']}", (30,80), 0, 2, (0,255,255), 3)
                
                # 顯示狀態
                status_text = f"Path: {state_mgr.path_state}"
                if state_mgr.guidance_source != "NONE":
                    status_text += f" [{state_mgr.guidance_source}]"
                cv2.putText(frame, status_text, (30, 40), 0, 0.8, (255,255,0), 2)

                cv2.imshow("Smart Guide", frame)
            
            except Exception as draw_err:
                print(f"Drawing Error: {draw_err}") 

            key = cv2.waitKey(delay_time) & 0xFF
            
            if key == ord('q'): 
                print("使用者手動退出。")
                os._exit(0) 
                
            elif key == ord('k'): 
                print("\n[測試] 3秒後模擬程式崩潰...")
                time.sleep(1) 
                raise RuntimeError("這是手動觸發的測試崩潰！")

            elif key == ord('f'): 
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
        emergency_speak("系統錯誤，請原地停止") 
        os._exit(1)

if __name__ == "__main__":
    main()