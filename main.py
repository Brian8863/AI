import os
import cv2
import numpy as np
import threading
import queue
import time
import traceback
import gc
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

# 設定環境變數：隱藏 TensorFlow 的除錯訊息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
    # [1] 檔案路徑設定
    # -----------------------------------------------------------
    # 0 或 1 代表 Webcam
    VIDEO_PATH: any = 2 
    
    # ★ 旋轉設定：直拍請設 True，橫拍請設 False
    NEED_ROTATION: bool = True 
    
    MODEL_TRAFFIC: str = "Model/traffic&count.pt" 
    MODEL_ZEBRA: str = "Model/zebra_v3.pt"            
    MODEL_CNN: str = "Model/cnn_digit_model_new.h5" 
    
    # -----------------------------------------------------------
    # [2] 解析度設定
    # -----------------------------------------------------------
    IMGSZ_TRAFFIC: int = 960    
    IMGSZ_ZEBRA: int = 640      
    
    # -----------------------------------------------------------
    # [3] 信心門檻
    # -----------------------------------------------------------
    CONF_TRAFFIC: float = 0.7    
    CONF_ZEBRA: float = 0.5     
    CONF_CNN: float = 0.7        
    
    # -----------------------------------------------------------
    # [4] 效能優化設定 (錯峰運算)
    # -----------------------------------------------------------
    PROCESS_CYCLE: int = 5       
    
    FRAME_TRAFFIC: int = 0      
    FRAME_GREEN: int = 1        
    FRAME_ZEBRA: int = 2        
    FRAME_GREEN_2: int = 3      
    FRAME_REST: int = 4         
    
    FRAME_COMMON: int = -1      
    
    QUEUE_MAX: int = 2 
    
    # -----------------------------------------------------------
    # [5] 綠色人行道設定 (HSV)
    # -----------------------------------------------------------
    LOWER_GREEN: np.ndarray = field(default_factory=lambda: np.array([75, 40, 40]))   
    UPPER_GREEN: np.ndarray = field(default_factory=lambda: np.array([92, 255, 255])) 
    
    # -----------------------------------------------------------
    # [6] 路徑引導靈敏度 (★ 關鍵修正區域)
    # -----------------------------------------------------------
    ROAD_ROI_TOP: float = 0.5 
    IGNORE_BOTTOM_RATIO: float = 0.1 
    
    PATH_DEVIATION_TH: int = 3  
    PATH_RETURN_BUFFER: int = 2

    # ★ 修正 1：放寬安全區 (原本 0.25 -> 改 0.40)
    # 讓中間 40% 的區域都算「直走」，減少左右修正的語音干擾
    PATH_CENTER_RATIO: float = 0.40
    PATH_CENTER_BUFFER: float = 0.05     
    
    # ★ 修正 2：新增「斑馬線最小面積門檻」
    # 原本是寫死 0.5%，現在改用參數控制，設為 1.5%
    # 只有看到夠大塊的斑馬線才開始導航，過濾小白點
    ZEBRA_MIN_AREA: float = 1.5
    
    # -----------------------------------------------------------
    # [7] 語音與邏輯控制
    # -----------------------------------------------------------
    STARTUP_GRACE_PERIOD: float = 5.0
    PATH_LOST_TIMEOUT: float = 3.0   
    ZEBRA_LOCK_TIMEOUT: float = 2.0 
    
    REMIND_INTERVAL: float = 8.0 
    TIMEOUT_LOCK: float = 3.0    
    
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
                    # print(f"🔊 [語音]: {msg}")
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
        self.last_heartbeat = time.time()
        self.prev_light_tts = None
        self.guidance_source = "NONE" 
        
        self.smoothed_cx = None
        self.last_zebra_time = 0 

    def update(self, det_traffic, det_green, det_zebra, frame_width):
        now = time.time()
        
        # 心跳聲
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
        
        target_cx = width // 2
        center_x = width // 2
        
        new_guidance_source = "NONE"

        # ★★★ 修正重點：使用 ZEBRA_MIN_AREA 取代原本的 0.5 ★★★
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
        
        # 平滑處理
        if new_guidance_source in ["ZEBRA", "GREEN"]:
            if self.smoothed_cx is None:
                self.smoothed_cx = raw_target_cx
            else:
                self.smoothed_cx = int(self.smoothed_cx * 0.7 + raw_target_cx * 0.3)
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
                    # 智慧靜音
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
# 4. 影像偵測模組
# ==========================================
class TrafficDetector:
    def __init__(self, config: Config):
        self.cfg = config
        self._init_models()

    def _init_models(self):
        print(f"正在載入模型...")
        self.model_traffic = YOLO(self.cfg.MODEL_TRAFFIC)
        self.model_zebra = YOLO(self.cfg.MODEL_ZEBRA) 
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
            
            # 使用 call/invoke 方式而非 .predict() 以避免 Graph 堆積
            input_tensor = resized.reshape(1, 28, 28, 1)
            pred_tensor = self.cnn(input_tensor, training=False)
            pred = pred_tensor.numpy()
            
            if np.max(pred) > self.cfg.CONF_CNN: return np.argmax(pred)
        except Exception as e:
            pass
        return None

    def detect_green_path(self, frame):
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
        except Exception: pass
        return res

    def detect_zebra(self, frame):
        res = {"percentage": 0, "cx": frame.shape[1]//2, "masks_list": [], "box": None}
        h, w = frame.shape[:2]
        try:
            ai_input = frame.copy()
            if self.cfg.IGNORE_BOTTOM_RATIO > 0:
                ai_input[int(h * (1 - self.cfg.IGNORE_BOTTOM_RATIO)):, :] = 0
            
            results = self.model_zebra(ai_input, imgsz=self.cfg.IMGSZ_ZEBRA, conf=self.cfg.CONF_ZEBRA, 
                                     verbose=False, retina_masks=True)
            r = results[0]
            
            # 模式 A: 分割 (Mask)
            if r.masks is not None:
                all_masks_points = r.masks.xy
                total_area = 0
                weighted_cx = 0
                for points in all_masks_points:
                    if len(points) > 0:
                        ys = points[:, 1]
                        if np.min(ys) < h * self.cfg.ROAD_ROI_TOP: continue 
                        pts = points.astype(np.int32)
                        rect = cv2.minAreaRect(pts)
                        (center), (r_w, r_h), angle = rect
                        if r_w == 0 or r_h == 0: continue
                        aspect_ratio = max(r_w, r_h) / min(r_w, r_h)
                        if aspect_ratio < 1.5: continue 
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
            
            # 模式 B: 偵測 (Box) - 沒Mask時才用
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

        except Exception as e:
            print(f"Zebra Det Error: {e}")
        return res

    def detect_obstacles(self, frame):
        return {"person": 0, "vehicle": 0}

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
                if frame_q.full():
                    time.sleep(0.01) 
                    try: frame_q.get_nowait() 
                    except: pass
                
                ret, f = cap.read()
                if not ret: 
                    if cfg.LOOP_VIDEO and is_video_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        frame_q.put(None)
                        break
                
                if cfg.NEED_ROTATION:
                    f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)

                try: frame_q.put(f, timeout=0.01)
                except: pass
            cap.release()
        
        threading.Thread(target=reader, daemon=True).start()

        print(f"系統已啟動。FPS: {video_fps}。按 'q' 離開。")
        
        cv2.namedWindow("Smart Guide", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Guide", 1024, 768)
        
        cache = {
            "traffic": {}, 
            "green_path": {"percentage": 0, "cx": 0, "contours": []},
            "zebra": {"percentage": 0, "cx": 0, "masks_list": [], "box": None},
        }
        
        idx = 0
        while True:
            frame = frame_q.get()
            if frame is None: 
                print("影片播放結束。")
                break
            
            cycle = idx % cfg.PROCESS_CYCLE
            idx += 1

            if idx % 300 == 0:
                gc.collect()
            
            if cycle == cfg.FRAME_TRAFFIC:
                cache["traffic"] = detector.detect_traffic(frame, state_mgr.cnn_enabled)
                
            elif cycle == cfg.FRAME_GREEN or cycle == cfg.FRAME_GREEN_2:
                cache["green_path"] = detector.detect_green_path(frame)
                
            elif cycle == cfg.FRAME_ZEBRA:
                cache["zebra"] = detector.detect_zebra(frame)
                
            elif cycle == cfg.FRAME_REST:
                pass
            
            state_mgr.update(cache["traffic"], cache["green_path"], cache["zebra"], frame.shape[1])
            
            # --- 畫面繪製 ---
            try:
                if cache["green_path"]["contours"]:
                    cv2.drawContours(frame, cache["green_path"]["contours"], -1, (0, 255, 0), 2)
                
                # 繪圖保護：先檢查 masks_list
                if cache["zebra"].get("masks_list"):
                    overlay = frame.copy()
                    for mask_pts in cache["zebra"]["masks_list"]:
                        cv2.fillPoly(overlay, [mask_pts], (0, 165, 255))
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                # 再檢查 box
                elif cache["zebra"].get("box") is not None:
                    bx1, by1, bx2, by2 = cache["zebra"]["box"]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 255), 3)
                
                if state_mgr.smoothed_cx is not None and state_mgr.guidance_source != "NONE":
                    cx = state_mgr.smoothed_cx
                    h, w = frame.shape[:2]
                    screen_center_x = w // 2
                    guide_color = (0, 165, 255) if state_mgr.guidance_source == "ZEBRA" else (0, 255, 0)
                    cv2.line(frame, (screen_center_x, h), (cx, h//2), guide_color, 4)
                    cv2.circle(frame, (cx, h//2), 15, guide_color, -1)
                    
                    # 畫出安全區 (除錯用)
                    limit = int(w * cfg.PATH_CENTER_RATIO)
                    cv2.line(frame, (screen_center_x - limit, h), (screen_center_x - limit, 0), (100,100,100), 1)
                    cv2.line(frame, (screen_center_x + limit, h), (screen_center_x + limit, 0), (100,100,100), 1)
                
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

            key = cv2.waitKey(delay_time) & 0xFF
            
            if key == ord('q'): 
                print("使用者手動退出。")
                os._exit(0) 

        stop_event.set()
        tts.stop()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"CRASHED: {e}")
        emergency_speak("系統錯誤") 
        os._exit(1)

if __name__ == "__main__":
    main()