# =============================================================================
# drone_feed.py 
# Phase 2 | Zero-Latency Hardware Integration (Multi-Hardware Support)
# =============================================================================

import os
import cv2
import threading
import time

# Apply strict experimental UDP flags globally (FFmpeg will only use them if we call it)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|strict;experimental"

class DroneVideoFeed:
    def __init__(self, source=0):
        print(f"[VideoEngine] Connecting to video source: {source}")
        self.source = source
        
        # Categorize the hardware source
        self.is_network = isinstance(source, str) and source.startswith(("rtsp", "http", "udp"))
        self.is_webcam = isinstance(source, int)
        self.is_live = self.is_network or self.is_webcam
        
        # Boot the correct OpenCV backend based on hardware
        if self.is_network:
            print("[VideoEngine] Detected Network Stream. Engaging FFmpeg zero-latency protocols...")
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        elif self.is_webcam:
            print("[VideoEngine] Detected USB/Local Camera. Engaging standard DirectShow protocols...")
            # Using standard initialization for webcams
            self.cap = cv2.VideoCapture(source)
        else:
            print("[VideoEngine] Detected Local File. Engaging playback protocols...")
            self.cap = cv2.VideoCapture(source)
            
        if not self.cap.isOpened():
            raise ValueError(f"[VideoEngine] FAILED: Could not connect to source: {source}")
            
        if self.is_live:
            # Force the hardware buffer to strictly hold only 1 frame
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or not self.fps:
            self.fps = 30.0
            
        self.running = True
        self.ret = False
        self.frame = None
        
        print(f"[VideoEngine] Feed opened. Target FPS: {self.fps}. Booting thread...")
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        time.sleep(1.0)

    def _update(self):
        target_delay = 1.0 / self.fps
        missed_frames = 0  
        
        while self.running:
            loop_start = time.time()
            
            if self.is_live:
                ret = self.cap.grab()
                if ret:
                    ret, frame = self.cap.retrieve()
                    if ret:
                        self.ret = ret
                        self.frame = frame
                        missed_frames = 0  
                    else:
                        missed_frames += 1
                else:
                    missed_frames += 1
                
                # Failsafe Trigger: 5 dropped packets = declare sensor dead
                if missed_frames > 5:
                    self.ret = False
                    self.frame = None
                    time.sleep(0.05) 
            else:
                # Playback for local .mp4 files
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = frame
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            
            # Artificial pacing for local .mp4 playback
            if not self.is_live:
                elapsed = time.time() - loop_start
                sleep_time = target_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def get_latest_frame(self):
        if not self.ret or self.frame is None:
            return None
        return self.frame.copy()

    def close_feed(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()
        print("[VideoEngine] Video feed closed.")