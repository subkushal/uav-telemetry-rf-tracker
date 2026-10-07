# RF SURVEILLANCE — TACTICAL DASHBOARD
## Setup & Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Configure video source
Edit `main_live.py` and set `video_source=0` (webcam) or your RTSP URL.

To scan available cameras:
```
python find_drone_camera.py
```

### 3. Set GPS coordinates
Edit `main_live.py` → `TEST_LAT` / `TEST_LON` to match your deployment location.

### 4. Run the server
```
python server.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## Dashboard Layout

```
┌─────────────────────────────────────┬──────────────┐
│           LIVE VIDEO FEED           │   SIDEBAR    │
│         (1280×720 MJPEG)            │  Source table│
│    OpenCV overlay rendered on top   │  Conf breakdown│
│                                     │  Event log   │
├──────────────┬──────────────────────┴──────────────┤
│  NODE TELEM  │   SIGNAL SPARKLINES   │  KF DIAG    │
└──────────────┴───────────────────────┴─────────────┘
```

## File Structure

```
surveillance_dashboard/
├── server.py                ← Flask server + JSON API  (START HERE)
├── main_live.py  
├── templates/
│   └── index.html           ← Military-grade HTML dashboard
├── kalman_filter.py         ← Adaptive Kalman Filter
├── signal_pipeline.py       ← RF source tracking pipeline
├── signal_model.py          ← Friis + Two-Ray RF model
├── confidence.py            ← Multi-factor confidence engine
├── projection_v2.py         ← Camera projection (Phase 2)
├── camera_model.py          ← Pinhole + Brown-Conrady distortion
├── overlay_v2.py            ← OpenCV overlay renderer
├── operator_interface.py    ← Alert + event log system
├── telemetry.py             ← Fixed node telemetry
├── geo_math.py              ← GPS → local NE math
├── gps_noise.py             ← GPS/TDOA noise simulation
├── drone_feed.py            ← Video capture wrapper
├── find_drone_camera.py     ← Camera discovery utility
├── targets.py               ← (Phase 1 legacy)
└── requirements.txt
```
-----------------------------------------------------------------------------------------------------------------
I. Core Integration & Networking
server.py — The Central Nervous System. The main Flask web server and multi-threaded integration hub. It ingests video, runs the data loops, commands the raycaster, and broadcasts the live Tactical Dashboard to the web browser.

main_live.py — The Bootstrapper. Initializes the camera feeds, backend infrastructure, and hardware links before handing control over to the main server loop.

motorola_bridge.py — The RF Hardware Link. Monitors the secondary display to extract live GPS coordinate broadcasts from the physical Motorola radios on the ground.

--------------------------------------------------------------------------------------------------------------------
II. Computer Vision & Neural Networks
yolo_detector.py — The Vision Core. Loads the custom neural network weights (best.pt or best.onnx) to detect humans in the live video feed and draw tracking bounding boxes.

ocr_engine.py — The Telemetry Engine. Reads the drone's live On-Screen Display (OSD) to extract critical flight parameters, or acts as a zero-overhead static anchor for fixed-node deployments.

----------------------------------------------------------------------------------------------------------------------
III. Spatial Geometry & Data Fusion
sensor_fusion.py — The Target Binder. Houses the Hungarian Algorithm. It evaluates the 50-meter mathematical locking gate to bind raycasted YOLO humans to live Motorola radio signals.

coordinate_projector.py — The 3D Raycaster. Takes the 2D pixel locations of the YOLO bounding boxes and raycasts them down into real-world geographic GPS coordinates.

projection_v2.py — Matrix Transformations. Handles the complex NED-to-gimbal matrix rotations required for advanced camera projection mapping.

camera_model.py — Lens Physics. Manages pinhole camera mathematics and Brown-Conrady distortion to ensure edge-to-edge pixel accuracy.

geo_math.py — Geographic Conversions. Handles the raw mathematical conversions from global GPS (Latitude/Longitude) to localized North-East coordinate vectors.

---------------------------------------------------------------------------------------------------------------------
IV. RF Physics & Tracking
signal_pipeline.py — The Tracking Matrix. Manages the lifecycle, spawning, and tracking memory of all active RF sources in the field.

kalman_filter.py — The Adaptive Physics Engine. Tracks signal velocity and smooths out camera micro-jitters using predictive state estimations.

signal_model.py — The RF Simulator. Computes physical link budgets, Friis path loss, and two-ray ground reflections for advanced signal degradation tracking.

confidence.py — The Scoring Engine. A multi-factor algorithm that evaluates signal strength, track age, and covariance to output a dynamic reliability score for each target.

gps_noise.py — Noise Management. Simulates and filters environmental GPS/TDOA signal variations.

--------------------------------------------------------------------------------------------------------------------
V. User Interface & Utilities
overlay_v2.py — The Augmented Reality (AR) Engine. Takes the raw video frame and draws the tactical HUD, confidence gauges, and Red/Blue target boxes directly onto the live stream.

operator_interface.py — The Alert System. Manages the visual warning banners, event logs, and tactical screen notifications based on changing threat levels.

telemetry.py — Data Structures. Defines the specific Python dataclasses used for passing fixed-node and dynamic drone flight variables between scripts.

drone_feed.py — Stream Ingestion. The video capture wrapper that handles the raw RTSP network stream or local .mp4 file processing.

find_drone_camera.py — Hardware Discovery. A lightweight utility script to probe the system and identify the correct OpenCV index for connected webcams or capture cards.

targets.py — Legacy Simulator. Contains the simulated ghost targets used during Phase 1 testing.

requirements.txt — The Environment Blueprint. Defines the exact versions of the external AI and math libraries (PyTorch, YOLO, EasyOCR) required to run the code without crashing.


----------------------------------------------------------------------------------------------------------------------
📦 1. Verify Deployment Package

Before installing, ensure your project folder (pendrive) contains the following:

All .py source code files (server.py, ocr_engine.py, sensor_fusion.py, etc.)

The YOLO neural network weights file (best.pt or best.onnx)

The requirements.txt file (exactly as provided)

💻 2. Standard Installation (Requires Internet for First Boot)

Follow these exact steps in your computer's terminal/command prompt to build the environment.

Step A: Open the Terminal

Navigate into the main project folder and open your command line or terminal.

Step B: Create the Virtual Environment

Create a clean, isolated Python environment so these AI libraries do not conflict with the rest of your computer.

python -m venv venv


Step C: Activate the Virtual Environment

You must turn the environment on. You will know it worked if you see (venv) appear at the beginning of your terminal line.

On Windows:

venv\Scripts\activate


On Mac/Linux:

source venv/bin/activate


Step D: Install the Tactical Stack

Upgrade your package manager and install the required AI libraries (PyTorch, YOLO, OpenCV, EasyOCR).

python -m pip install --upgrade pip
pip install -r requirements.txt


🚀 3. Launching the System

Once the installation is complete, boot the integration hub:

python server.py


Open your web browser and navigate to http://127.0.0.1:5000 to view the live tactical dashboard.

⚠️ CRITICAL: FIRST BOOT REQUIREMENT
The very first time you execute python server.py on a new machine, the computer MUST be connected to the internet. The EasyOCR engine requires a one-time automatic download of its English language detection weights (approximately 30MB).
If you attempt the first boot in an offline field environment, the system will throw a connection error and crash. Once this initial 30MB download is complete, the software can run completely offline in the field forever.

--------------------------------------------------------------------------------------------------------------------------------------------
🔒 4. (Optional) 100% Offline / Air-Gapped Installation

If the target system has zero internet access and cannot perform the first boot online, you must prepare the pendrive on an internet-connected computer first:

On the Internet-Connected PC:

Create a folder named offline_packages on your pendrive.

Download all library wheels into that folder:

pip download -r requirements.txt -d /path/to/pendrive/offline_packages


Manually download craft_mlt_25k.pth and english_g2.pth from the EasyOCR official repository and place them in an offline_models folder on the pendrive.

On the Offline Target PC:

Create and activate the virtual environment (Steps A, B, and C above).

Install directly from the pendrive folder without checking the internet:

pip install --no-index --find-links=./offline_packages -r requirements.txt
